#!/bin/bash

##############################################################################
#
#  Parallel unit/short regression tests
#  All run on 8 MPI tasks; should not take more than 1 minute each.
#
##############################################################################

DIR_TST=`pwd`
DIR_MPI=`pwd`/../mpi_s
DIR_SRC=`pwd`/../src
DIR_REG=`pwd`/regression

# Unit tests (make sure there is no stub mpi hanging around first)

cd $DIR_MPI
make clean

cd $DIR_SRC
make clean
make mpi

cd $DIR_TST
make clean
make do_tests_mpi
make clean

cd $DIR_REG

for f in ./pmpi08*inp
do
    input=$f
    stub=`echo $f | sed 's/.inp//'`
    echo
    mpirun -np 8 $DIR_SRC/Ludwig.exe $f > $stub.new

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

cd $DIR_TST
