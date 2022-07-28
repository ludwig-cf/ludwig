#!/bin/bash
#
#SBATCH --job-name=zzh
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu

NVHPC_VERSION=22.2
module load nvidia/nvhpc-nompi/$NVHPC_VERSION
# module load intel-20.4/compilers

module load gcc
module load openmpi/4.1.2-cuda-11.6

module list

export OMP_NUM_THREADS=1
export OMPI_MCA_pml=ob1

date
echo $LD_LIBRARY_PATH

srun --ntasks=8 --tasks-per-node=4 --cpus-per-task=2 --hint=nomultithread \
    --distribution=block:block nsys profile -t cuda,mpi -o profiler-${SLURM_JOB_ID} ./src/Ludwig.exe ./input256

# For unknown reasons this isn't on the PATH
# $NVHPC/Linux_x86_64/$NVHPC_VERSION/profilers/Nsight_Systems/host-linux-x64/QdstrmImporter profiler-${SLURM_JOB_ID}.qdstrm

date
