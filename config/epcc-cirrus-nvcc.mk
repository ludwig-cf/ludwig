###############################################################################
#
#  nvcc build
#
#  module load intel-mpi-17
#  module load intel-comiplers-17
#  module load cuda/9.1
#
#  Host   As epcc-cirrus-intel.mk
#  Device NVIDIA Tesla V100-SXM2
#
###############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_

CC     = nvcc
MPICC  = nvcc
CFLAGS = -ccbin=icpc -DADDR_SOA -DNDEBUG -arch=sm_70 -x cu -dc -Xcompiler -fast

AR = ar
ARFLAGS = -cr
LDFLAGS= -ccbin=icpc -arch=sm_70

MPI_HOME     = /lustre/sw/intel/compilers_and_libraries_2017.2.174/linux/mpi
MPI_INC_PATH = -I$(MPI_HOME)/include64
MPI_LIB_PATH = -L$(MPI_HOME)/lib64 -lmpi

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD = mpirun
MPIRUN_NTASK_FLAG = -np

