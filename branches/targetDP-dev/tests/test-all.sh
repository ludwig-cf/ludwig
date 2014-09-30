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
#  (c) 2014 The University of Edinburgh
#
##############################################################################
#!/bin/bash

. ./test-serial.sh -u d2q9
. ./test-serial.sh -u d2q9r
. ./test-serial.sh -u -r d3q15
. ./test-serial.sh -u d3q15r
. ./test-serial.sh -u -r d3q19
. ./test-serial.sh -u d3q19r

. ./test-mpix08.sh -u d3q15 pmpi08
. ./test-mpix08.sh -u -r d3q19 pmpi08
. ./test-mpix08.sh -u d3q19 long08

. ./test-long64.sh -r d3q19 long64
