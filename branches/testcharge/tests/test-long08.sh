#!/bin/bash

###############################################################################
#
# Longer parallel regression tests (up to 4-5 minutes each)
#
# All run on 8 MPI tasks
#
###############################################################################

DIR_TST=`pwd`
DIR_SRC=`pwd`/../src
DIR_REG=`pwd`/regression


cd $DIR_SRC
make clean
make mpi

cd $DIR_REG

for f in ./long08*inp
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
