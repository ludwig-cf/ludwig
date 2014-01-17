#!/bin/bash

##############################################################################
#
#  Parallel unit/short regression tests
#  All run on 8 MPI tasks; should not take more than 1 minute each.
#
##############################################################################

cd ../src
make clean
make libmpi

cd ../tests
make clean
make do_tests_mpi
make clean

cd ../src
make mpi

for f in ../tests/regression/pmpi08*inp
do
    input=$f
    stub=`echo $f | sed 's/.inp//'`
    echo
    mpirun -np 8 ./Ludwig.exe $f > $stub.new

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

cd ../tests
