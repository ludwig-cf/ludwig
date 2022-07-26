#!/bin/bash
#
#SBATCH --job-name=zzh
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu

NVHPC_VERSION=22.2
module load intel-20.4/compilers
module load nvidia/nvhpc/$NVHPC_VERSION

date
echo $LD_LIBRARY_PATH

mpiexec ./src/Ludwig.exe ./tests/performance/epcc-cirrus-sc16/input

date
