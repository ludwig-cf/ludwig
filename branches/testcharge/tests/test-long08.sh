#!/bin/bash

# Longer parallel regression tests
# All run on 8 MPI tasks

cd ../src
make clean
make libmpi
make mpi

for f in ../tests/regression/long08*inp
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
