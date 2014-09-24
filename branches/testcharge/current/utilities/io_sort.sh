#!/bin/bash
#PBS -l mppwidth=1
#PBS -l mppnppn=1
#PBS -l walltime=12:00:00
#PBS -A e73-oh

cd $PBS_O_WORKDIR
export NPROC=`qstat -f $PBS_JOBID | grep mppwidth | awk '{print $3}'`
export NTASK=`qstat -f $PBS_JOBID | grep mppnppn  | awk '{print $3}'`

python ./io_sort.py filelist 16 > stdout
