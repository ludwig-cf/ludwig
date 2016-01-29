#!/bin/bash 
##############################################################################
#
#  test-simdvl.sh
#
#  Serial tests only.
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2016 The University of Edinburgh
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
##############################################################################

# This is intended for the nightly test, so this copy must be here

echo "TEST --> SIMD VECTOR LENGTH TWO"
cp ../config/lunix-nvcc-simdvl2.mk ../config.mk
make clean
make compile-run-serial-d3q15
make compile-run-serial-d3q19

