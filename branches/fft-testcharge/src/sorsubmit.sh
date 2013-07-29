# script to automate submission for PFP coursework
# should be executed from the directory containing the code version to be run
# each directory contains a symlink to the main file in the MD directory 

#!/bin/bash

small=2048
medium=4096
large=16384

if [ $1 = 26 ]; then
  queue=nodes128-dev
elif [ $1 -le $small ]; then
  queue=nodes128
elif [ $1 -le $medium ]; then
  queue=nodes256
else
  queue=nodes1024
fi

if [ $1 != 0 ]; then
  qsub -o output/ -q $queue sorludwig.pbs -v arg1=$1
else
  echo "Please supply the number of tasks you wish to run on"
fi
