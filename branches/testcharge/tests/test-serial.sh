#!/bin/bash

##############################################################################
#
#  Driver script for serial unit/smoke regression tests
#  These tests should run within a few minutes.
#
#  Intended to be invoked from the test directory.
#
##############################################################################

DIR_TST=`pwd`
DIR_MPI=`pwd`/../mpi_s
DIR_SRC=`pwd`/../src
DIR_REG=`pwd`/regression

# Unit tests

cd $DIR_MPI
make clean
make libc
make testc


cd $DIR_SRC
make clean
make serial

cd $DIR_TST
make clean
make do_tests
make clean

# Smoke tests
# The naming convention for the files is "serial-xxxx-xxx.inp"
# for the input and with extension ".log" for the reference
# output.

# We are going to run from the regression test directory

cd $DIR_REG

for f in ./serial*inp
do
    input=$f
    stub=`echo $f | sed 's/.inp//'`
    echo
    $DIR_SRC/Ludwig.exe $f > $stub.new

    # Get difference via the difference script
    $DIR_TST/test-diff.sh $stub.new $stub.log

    if [ $? -ne 0 ]
	then
	echo "    FAIL $f"
	$DIR_TST/test-diff.sh -v $stub.log $stub.new
	else
	echo "PASS     $f"
    fi
done

# Clean up all directories and finish

cd $DIR_SRC
make clean

cd $DIR_MPI
make clean

cd $DIR_TST
