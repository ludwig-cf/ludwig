#!/bin/bash

##############################################################################
#
#  test.sh
#
#  Run a regression test.
#
#  ./test.sh input.inp "serial launch command" "parallel launch command"
#
#  The executable may be launched in parallel, but the test-diff
#  script must be run in serial. Hence the requirement for both
#  methods to be provided as two of the three arguments.
#
#  The launch method may be an empty string in serial.
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

###############################################################################
#
#  main
#
###############################################################################

function main() {

  # We are going to run this from, e.g.,  regression/d3q15
  # hence the rather long relative paths
  executable=../../../src/Ludwig.exe
  test_diff=../../test-diff.sh

  # Arguments (zero checking!)
  input="$1"
  launch_serial="$2"
  launch_mpi="$3"

  # The naming convention for the files is "serial-xxxx-xxx.inp"
  # for the input and with extension ".log" for the reference
  # output.

  stub=`echo $input | sed 's/.inp//'`
  echo
  ${launch_mpi} ${executable} $input > $stub.new

  # Get difference via the difference script
  ${launch_serial} ${test_diff} $stub.new $stub.log

  if [ $? -ne 0 ]
  then
      echo "    FAIL ./$input"
      ${launch_serial} ${test_diff} -v $stub.log $stub.new
      # ok, exit with zero
     exit 0
  else
      echo "PASS     ./$input"
  fi

  return
}

# Run and exit

main "$@"
