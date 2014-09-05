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

. ./test-serial.sh
. ./test-pmpi08.sh
. ./test-long08.sh
. ./test-long64.sh
