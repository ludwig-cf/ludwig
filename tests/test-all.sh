##############################################################################
#
#  test-all.sh
#
#  Just a convenience to run all tests...
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2014-2015 The University of Edinburgh
#
##############################################################################
#!/bin/bash

make compile-run-serial-d2q9
make compile-run-serial-d2q9r
make compile-run-serial-d3q15
make compile-run-serial-d3q15r
make compile-run-serial-d3q19
make compile-run-serial-d3q19r

make compile-run-mpi-d2q9
make compile-run-mpi-d2q9r
make compile-run-mpi-d3q15
make compile-run-mpi-d3q15r
make compile-run-mpi-d3q19
make compile-run-mpi-d3q19r


#. ./test-mpix08.sh -u d3q19 long08
#. ./test-long64.sh -r d3q19 long64
