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

nsys profile -t cuda,mpi -o profiler-${SLURM_JOB_ID} mpiexec ./src/Ludwig.exe ./input256

# For unknown reasons this isn't on the PATH
$NVHPC/Linux_x86_64/$NVHPC_VERSION/profilers/Nsight_Systems/host-linux-x64/QdstrmImporter profiler-${SLURM_JOB_ID}.qdstrm

date
