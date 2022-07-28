#!/bin/bash
#
#SBATCH --job-name=zzh
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu

# module load intel-20.4/compilers
NVHPC_VERSION=22.2
module load nvidia/nvhpc-nompi/$NVHPC_VERSION
module load gcc
module load openmpi/4.1.2-cuda-11.6
# export LD_LIBRARY_PATH=/mnt/lustre/indy2lfs/work/dc134/dc134/s2225484/sw/openmpi/4.1.4-cuda-11.6/lib:$LD_LIBRARY_PATH

ip addr
module list

export OMP_NUM_THREADS=1
# export OMPI_MCA_btl_tcp_if_include=10.0.0.0/8
# export OMPI_MCA_btl_tcp_endpoint_complete_connect=1
# export OMPI_MCA_opal_common_ucx_opal_mem_hooks=1
export OMPI_MCA_pml=ob1
# export OMPI_MCA_btl=^uct
# export UCX_TLS=rc,sm,cuda_copy,gdr_copy,cuda_ipc
# export UCX_NET_DEVICES=mlx5_0:1
# export OMPI_MCA_pml_ucx_verbose=100

date
echo $LD_LIBRARY_PATH

ldd ./src/Ludwig.exe

srun --ntasks=8 --tasks-per-node=4 --cpus-per-task=2 --hint=nomultithread \
    --distribution=block:block ./src/Ludwig.exe ./input256

date
