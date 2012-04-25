#############################################################################
#
#  trycalibration.sh
#
#  This runs the tests of the colloid hydrodynamic radii.
#  This is a function of the input radius, the viscosity,
#  and the LB relaxation.
#
#  Note running all of these will take some time, as detailed
#  below. Some are therefore commented out.
#
#  $Id$
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2011 The University of Edinburgh
#
#############################################################################
#!/bin/bash

cd ../src
make clean
make mpi
cd ../tests/calibration

run="mpirun -np 8 ../../src/Ludwig.exe"

$run ./input_a125_eta0.1666 > d3q19_a125_eta0.1666.log
$run ./input_a125_eta0.0100 > d3q19_a125_eta0.0100.log
$run ./input_a125_eta0.0010 > d3q19_a125_eta0.0010.log

# About 2hr on 8 cpu

$run ./input_a230_eta0.1666 > d3q19_a230_eta0.1666.log
$run ./input_a230_eta0.0100 > d3q19_a230_eta0.0100.log
$run ./input_a230_eta0.0010 > d3q19_a230_eta0.0010.log

# About 40 hr on 8 cpu

$run ./input_a477_eta0.1666 > d3q19_a477_eta0.1666.log
$run ./input_a477_eta0.0100 > d3q19_a477_eta0.0100.log
#$run ./input_a477_eta0.0010 > d3q19_a477_eta0.0010.log

##############################################################################
#
#  Repeat for two relaxation time (trt) scheme
#
##############################################################################

#$run ./input_trt_a125_eta0.1666 > d3q19_trt_a125_eta0.1666.log
#$run ./input_trt_a125_eta0.0100 > d3q19_trt_a125_eta0.0100.log
#$run ./input_trt_a125_eta0.0010 > d3q19_trt_a125_eta0.0010.log

# About 2hr on 8 cpu

#$run ./input_trt_a230_eta0.1666 > d3q19_trt_a230_eta0.1666.log
#$run ./input_trt_a230_eta0.0100 > d3q19_trt_a230_eta0.0100.log
#$run ./input_trt_a230_eta0.0010 > d3q19_trt_a230_eta0.0010.log

# About 40 hr on 8 cpu

#$run ./input_trt_a477_eta0.1666 > d3q19_trt_a477_eta0.1666.log
#$run ./input_trt_a477_eta0.0100 > d3q19_trt_a477_eta0.0100.log
#$run ./input_trt_a477_eta0.0010 > d3q19_trt_a477_eta0.0010.log

cd ../../src
make clean
