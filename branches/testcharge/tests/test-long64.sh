#!/bin/bash

###############################################################################
#
# Even longer parallel regression tests
# All run on 64 MPI tasks
#
###############################################################################

DIR_TST=`pwd`
DIR_SRC=`pwd`/../src
DIR_REG=`pwd`/regression

cd $DIR_SRC
make clean
make libmpi
make mpi

cd $DIR_REG

for f in ./long64*inp
do
    input=$f
    stub=`echo $f | sed 's/.inp//'`
    echo
    mpirun -np 64 $DIR_SRC/Ludwig.exe $f > $stub.new

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
