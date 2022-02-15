#!/bin/bash
#
#SBATCH --job-name=zzh
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-cascade
#SBATCH --reservation=gputeachmsc
#SBATCH --qos=gpu

#module load intel-mpi-19
#module load intel-comiplers-19
#module load nvidia/cuda-11.2
NVHPC_VERSION=21.2
module load nvidia/nvhpc/$NVHPC_VERSION

date
echo $LD_LIBRARY_PATH

srun ./src/Ludwig.exe ./tests/performance/epcc-cirrus-sc16/input
