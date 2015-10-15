##############################################################################
#
#  test-serial.sh
#
#  Driver script for serial unit/smoke regression tests
#  These tests should run within a few minutes.
#
#  Intended to be invoked from the test directory.
#
#  Use e.g., ./test-serial.sh -u -r d3q19
#  to run unit tests and regreesion tests for d2q9 compilation
#
#  ./test-serial.sh -c d2q9
#  for compilation only (no clean-up at end)
#
#  ./test-serial -x -u -r d3q15
#  for run only (no compilation)
#
#
#  Edinburgh Soft Matter and Statisical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2014-2015 The University of Edinburgh
#  Contributing authors:
#  Kevin Stratford (kevin@epcce.ed.ac.uk)
#
##############################################################################
#!/bin/bash

DIR_TST=`pwd`
DIR_MPI=`pwd`/../mpi_s
DIR_TARGETDP=`pwd`/../targetDP
DIR_SRC=`pwd`/../src
DIR_REG=`pwd`/regression
DIR_UNT=`pwd`/unit

###############################################################################
#
#  main
#
###############################################################################

function main() {

  echo $0 $@
  OPTIND=1

  # Parse command line to determine which test(s) to run

  if [ $# -lt 1 ]; then
    echo "Usage: $0 [-c] -r -u [d2q9 | d3q15 | ...]"
    exit -1
  fi

  run_comp=1
  run_unit=0
  run_regr=0
  run_clean=1

  while getopts crux opt
  do
    case "$opt" in
      c)
	run_clean=0
	;;
      r)
	run_regr=1
	;;
      u)
	run_unit=1
	;;
      x)
	run_comp=0
	;;
    esac
  done

  shift $((OPTIND-1))

  [[ $run_comp -eq 1 ]]  && test_compile "$1"
  [[ $run_unit -eq 1 ]]  && test_unit "$1"
  [[ $run_regr -eq 1 ]]  && test_regr "$1"
  [[ $run_clean -eq 1 ]] && test_clean

  return
}

##############################################################################
#
#  test_compile [d2q9 | d2q9r | d3q15 | d3q15r | d3q19 | d3q19r]
#
##############################################################################

function test_compile {

  test_clean

  cd $DIR_MPI
  make libc
  make testc

  cd $DIR_TARGETDP
  make targetDP_C

  cd $DIR_SRC
  make serial-$1

  cd $DIR_UNT
  make serial-$1

}

##############################################################################
#
#  test_clean
#
##############################################################################

function test_clean {

  cd $DIR_MPI
  make clean

  cd $DIR_TARGETDP
  make clean

  cd $DIR_SRC
  make clean

  cd $DIR_UNT
  make clean

  return
}

##############################################################################
#
#  test_unit [d2q9 | d2q9r | d3q15 | d3q15r | d3q19 | d3q19r]
#
##############################################################################

function test_unit {

  echo "TEST --> unit tests serial $1"

  cd $DIR_UNT
  make run-serial

  return
}

##############################################################################
#
#  test_regr [d2q9 | ...]
#
##############################################################################

function test_regr {

  echo "TEST --> regression tests serial $1"

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

  return
}

# Run and exit

main "$@"
