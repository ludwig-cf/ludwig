# script to automate submission for PFP coursework
# should be executed from the directory containing the code version to be run
# each directory contains a symlink to the main file in the MD directory 

#!/bin/bash


qsub -o output/ -e error/ fftludwig.pbs
