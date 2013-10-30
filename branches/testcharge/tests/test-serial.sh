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
make libc
make testc

cd ../src
make clean
make lib

cd ../tests
make do_tests
make clean

# Smoke tests
# The naming convention for the files is "serial-xxxx-xxx.inp"
# for the input and with extension ".log" for the reference
# output.

cd ../src
make serial

for f in ../tests/regression/serial*inp
do
    input=$f
    stub=`echo $f | sed 's/.inp//'`
    echo
    ./Ludwig.exe $f > $stub.new

    # Get difference via the difference script
    ../tests/test-diff.sh $stub.new $stub.log

    if [ $? -ne 0 ]
	then
	echo "    FAIL $f"
	../tests/test-diff.sh -v $stub.log $stub.new
	else
	echo "PASS     $f"
    fi
done

# Clean up all directories and finish

make clean

cd ../mpi_s
make clean

cd ../tests
