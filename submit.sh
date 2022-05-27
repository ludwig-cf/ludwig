#!/bin/bash
#
#SBATCH --job-name=zzh
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu

#module load intel-mpi-19
#module load intel-comiplers-19
#module load nvidia/cuda-11.2
NVHPC_VERSION=22.2
module load nvidia/nvhpc/$NVHPC_VERSION
module load intel-20.4/compilers
module load intel-20.4/mpi
module load mpt

date
echo $LD_LIBRARY_PATH

srun ./src/Ludwig.exe ./tests/performance/epcc-cirrus-sc16/input

date
