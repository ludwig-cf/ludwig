#!/bin/bash

##############################################################################
#
#  Driver script for serial unit/smoke regression tests
#  These tests should run within a few minutes.
#
#  Intended to be invoked from the test directory.
#
##############################################################################  

# Unit tests

cd ../mpi_s
make clean
make libc
make testc

cd ../src
make clean
make lib

cd ../tests
make clean
make do_tests
make clean

# Smoke tests
# The naming convention for the files is "serial-xxxx-xxx.inp"
# for the input and with extension ".log" for the reference
# output.

cd ../src
make serial

# We are going to run from the regression test directory

cd ../tests/regression

for f in ./serial*inp
do
    input=$f
    stub=`echo $f | sed 's/.inp//'`
    echo
    ../../src/Ludwig.exe $f > $stub.new

    # Get difference via the difference script
    ../test-diff.sh $stub.new $stub.log

    if [ $? -ne 0 ]
	then
	echo "    FAIL $f"
	../test-diff.sh -v $stub.log $stub.new
	else
	echo "PASS     $f"
    fi
done

# Clean up all directories and finish

cd ../../src
make clean

cd ../mpi_s
make clean

cd ../tests
