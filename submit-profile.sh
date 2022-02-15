#!/bin/bash
#
#SBATCH --job-name=profiler
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu

NVHPC_VERSION=21.2
module load nvidia/nvhpc/$NVHPC_VERSION

date

nsys profile -o profiler-${SLURM_JOB_ID} ./src/Ludwig.exe ./tests/performance/epcc-cirrus-sc16/input

# For unknown reasons this isn't on the PATH
$NVHPC/Linux_x86_64/$NVHPC_VERSION/profilers/Nsight_Systems/host-linux-x64/QdstrmImporter profiler-${SLURM_JOB_ID}.qdstrm
