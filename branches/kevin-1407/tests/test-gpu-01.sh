##############################################################################
#
#  test-gpu-01.sh
#
#  Only d3q19r regrssion tests at the moment
#  Serial only, until a parallel test machine is available.
#
##############################################################################
#!/bin/bash --login

DIR_TST=`pwd`
DIR_MPI=`pwd`/../mpi_s
DIR_TARGETDP=`pwd`/../targetDP
DIR_SRC=`pwd`/../src
DIR_REG=`pwd`/regression
DIR_UNT=`pwd`/unit

##############################################################################
#
#  main
#
##############################################################################

function main() {

  test_compile
  test_regr
  test_clean

  cd $DIR_TST

  return
}

##############################################################################
#
#  test_compile
#
##############################################################################

function test_compile() {

  test_clean

  cd $DIR_TARGETDP
  make targetDP_CUDA

  cd $DIR_MPI
  make

  cd $DIR_SRC
  make USEGPU=1 serial-d3q19r

  return
}

##############################################################################
#
#  test_clean
#
##############################################################################

function test_clean() {

  cd $DIR_TARGETDP
  make clean

  cd $DIR_MPI
  make clean

  cd $DIR_SRC
  make clean

  return
}

##############################################################################
#
#  test_regr
#
##############################################################################

function test_regr() {

  echo "TEST --> regression tests serial GPU d3q19r"

  # We are going to run from the regression test directory

  cd $DIR_REG/d3q19

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
        echo "    FAIL ./d3q19r/$f"
        $DIR_TST/test-diff.sh -v $stub.log $stub.new
        else
        echo "PASS     ./d3q19r/$f"
    fi
  done

  return
}

# Run and exit

main "$@"
