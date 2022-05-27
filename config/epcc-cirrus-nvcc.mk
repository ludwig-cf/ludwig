###############################################################################
#
#  nvcc build
#
#  NVHPC_VERSION=22.2
#  module load nvidia/nvhpc/$NVHPC_VERSION
#  module load intel-20.4/compilers
#  module load intel-20.4/mpi
#  module load mpt
#
#  Host   As epcc-cirrus-intel.mk
#  Device NVIDIA Tesla V100-SXM2
#
###############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_

CC     = nvcc
CFLAGS = -ccbin=icpc -DADDR_SOA -DNDEBUG -arch=sm_70 -x cu -dc -Xcompiler -fast -Xcompiler -qopenmp

AR = ar
ARFLAGS = -cr
LDFLAGS= -ccbin=icpc -arch=sm_70 -liomp5

# MPI_HOME     = /lustre/sw/intel/compilers_and_libraries_2019.0.117/linux/mpi/intel64
# MPI_HOME     = /scratch/sw/intel/compilers_and_libraries_2019.0.117/linux/mpi/intel64
# MPI_HOME	 = /opt/hpe/hpc/mpt/mpt-2.25
MPI_HOME 	 = #/scratch/sw/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64
MPI_INC_PATH = #-I$(MPI_HOME)/include
MPI_LIB_PATH = -lmpi #-L$(MPI_HOME)/lib -lmpi

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD = mpirun
MPIRUN_NTASK_FLAG = -np

