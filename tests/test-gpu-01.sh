#!/bin/bash 
##############################################################################
#
#  test-gpu-01.sh
#
#  Only d3q19r regrssion tests at the moment
#  Serial only, until a parallel test machine is available.
#
##############################################################################

# This is intended for the nightly test, so this copy must be here

echo "TEST --> GPU serial"
cp ../config/lunix-nvcc-default.mk ../config.mk
make clean
make compile-serial-d3q19r
make run-serial-regr-d3q19r
