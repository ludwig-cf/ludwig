###############################################################################
#
#  test-long64.sh
#
#  Even longer parallel regression tests for 64 MPI tasks
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2014-2015 The University of Edinburgh
#  Contributing authors:
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
###############################################################################
#!/bin/bash

DIR_TST=`pwd`
DIR_MPI=`pwd`/../mpi_s
DIR_TARGETDP=`pwd`/../targetDP
DIR_SRC=`pwd`/../src
DIR_REG=`pwd`/regression
DIR_UNT=`pwd`/unit

MPIRUN=mpirun
NPROCS=64

##############################################################################
#
#  main
#
##############################################################################

function main() {

  echo $0 $@
  OPTIND=1

  if [ $# -lt 2 ]; then
    echo "Usage: $0 -r [d2q9 | d3q15 | ...] test-stub"
    exit -1
  fi

  run_regr=0

  while getopts ru opt
  do
      case "$opt" in
	  r)
	      run_regr=1
	      ;;
      esac
  done

  shift $((OPTIND-1))

  [[ $run_regr -eq 1 ]] && test_regr $1 $2

  return
}

##############################################################################
#
#  test_regr [d2q9 | ...] test-stub
#
##############################################################################

function test_regr {

  echo "TEST --> regression tests parallel $1"
  cd $DIR_MPI
  make clean

  cd $DIR_TARGETDP
  make clean
  make targetDP_C

  cd $DIR_SRC
  make clean
  make mpi-$1

  # Smoke tests
  # The naming convention for the files is "test-stub-xxxx-xxx.inp"
  # for the input and with extension ".log" for the reference
  # output.

  # We are going to run from the regression test directory
  # for the appropriate argument

  cd $DIR_REG/$1

  for f in $2*inp
  do
    input=$f
    stub=`echo $input | sed 's/.inp//'`
    echo
    $MPIRUN -np $NPROCS $DIR_SRC/Ludwig.exe $input > $stub.new

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

# Run and exit

main "$@"
