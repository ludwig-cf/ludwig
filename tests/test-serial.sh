##############################################################################
#
#  Driver script for serial unit/smoke regression tests
#  These tests should run within a few minutes.
#
#  Intended to be invoked from the test directory.
#
#  Use e.g., ./test-serial -u -r d2q9
#  to run unit tests and regreesion tests for d2q9 compilation
#
#  Edinburgh Soft Matter and Statisical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2014 The University of Edinburgh
#  Kevin Stratford (kevin@epcce.ed.ac.uk)
#
##############################################################################
#!/bin/bash

echo $0 $@
OPTIND=1

DIR_TST=`pwd`
DIR_MPI=`pwd`/../mpi_s
DIR_TARGETDP=`pwd`/../targetDP
DIR_SRC=`pwd`/../src
DIR_REG=`pwd`/regression
DIR_UNT=`pwd`/unit

if [ $# -lt 1 ]
then
    echo "Usage: $0 -r -u [d2q9 | d3q15 | ...]"
    exit -1
fi

##############################################################################
#
#  test_unit [d2q9 | d2q9r | d3q15 | d3q15r | d3q19 | d3q19r]
#
##############################################################################

function test_unit {

  echo "TEST --> unit tests serial $1"
  cd $DIR_MPI
  make clean
  make libc
  make testc

  cd $DIR_TARGETDP
  make clean
  make targetDP_C

  cd $DIR_SRC
  make clean
  make serial-$1

  cd $DIR_UNT
  make clean
  make serial-$1
  make run-serial
  make clean

  # Clean up all directories and finish

  cd $DIR_SRC
  make clean

  cd $DIR_MPI
  make clean

  cd $DIR_TST
}

##############################################################################
#
#  test_regr [d2q9 | ...]
#
##############################################################################

function test_regr {

  echo "TEST --> regression tests serial $1"
  cd $DIR_MPI
  make clean
  make libc
  make testc

  cd $DIR_SRC
  make clean
  make serial-$1

  # Smoke tests
  # The naming convention for the files is "serial-xxxx-xxx.inp"
  # for the input and with extension ".log" for the reference
  # output.

  # We are going to run from the regression test directory
  # for the appropriate argument

  cd $DIR_REG/$1

  for f in serial*inp
  do
    input=$f
    stub=`echo $input | sed 's/.inp//'`
    echo
    $DIR_SRC/Ludwig.exe $input > $stub.new

    # Get difference via the difference script
    $DIR_TST/test-diff.sh $stub.new $stub.log

    if [ $? -ne 0 ]
	then
	echo "    FAIL ./$1/$f"
	$DIR_TST/test-diff.sh -v $stub.log $stub.new
	else
	echo "PASS     ./$1/$f"
    fi
  done

  # Clean up all directories and finish

  cd $DIR_SRC
  make clean

  cd $DIR_MPI
  make clean

  cd $DIR_TST
}


# Run the tests

run_unit=0
run_regr=0

while getopts ru opt
do
case "$opt" in
    r)
	    run_regr=1
	    ;;
    u)
	    run_unit=1
	    ;;
esac
done

shift $((OPTIND-1))

if [ $run_unit -eq 1 ]
then
    test_unit $1
fi

if [ $run_regr -eq 1 ]
then
    test_regr $1
fi
