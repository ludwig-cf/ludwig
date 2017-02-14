#!/bin/bash

#PBS -N qscript
#PBS -l nodes=1
#PBS -l walltime=00:20:00
#PBS -A csc206

module swap PrgEnv-cray PrgEnv-intel

cd $PBS_O_WORKDIR
#export KMP_AFFINITY=disabled

export OMP_WAIT_POLICY=active
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true

export OMP_NUM_THREADS=8
pwd

aprun -n 1 hostname
aprun -n 2 -S 1 -d 8 -cc numa_node ./Ludwig.exe
