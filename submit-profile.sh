#!/bin/bash
#
#SBATCH --job-name=ludwig-profiler
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu

NVHPC_VERSION=22.2
module load nvidia/nvhpc/$NVHPC_VERSION
module load intel-20.4/compilers
module load intel-20.4/mpi
module load mpt

echo $LD_LIBRARY_PATH

date

nsys profile -o profiler-${SLURM_JOB_ID} mpirun -np 1 ./src/Ludwig.exe ./tests/performance/epcc-cirrus-sc16/input

# For unknown reasons this isn't on the PATH
$NVHPC/Linux_x86_64/$NVHPC_VERSION/profilers/Nsight_Systems/host-linux-x64/QdstrmImporter profiler-${SLURM_JOB_ID}.qdstrm

date
