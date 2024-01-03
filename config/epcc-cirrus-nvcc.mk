###############################################################################
#
#  nvcc build
#
#  There are a number of ways forward here... see further comments below
#
#  E.g., serial with stub MPI
#  module load nvidia/nvhpc-nompi/22.11
#
#  E.g., parallel using NVHPC with MPI ...
#  module load nvidia/nvhpc/22.11
#
###############################################################################

BUILD  = parallel
MODEL  = -D_D3Q19_

CC     = nvcc
CFLAGS = -g -DADDR_SOA -O2 -arch=sm_70 -x cu -dc

# PTX assembler extra information:  -Xptxas -v
# Alternative compiler, e.g., Intel: -ccbin=icpc -Xcompiler -fast

AR = ar
ARFLAGS = -cr
LDFLAGS = -arch=sm_70

# nvhpc (mpicc is present but drives nvc not nvcc) so use nvcc still ... but
MPI_HOME     = ${NVHPC_ROOT}/comm_libs/mpi
MPI_INC_PATH = -I$(MPI_HOME)/include
MPI_LIB_PATH = -L$(MPI_HOME)/lib -lmpi

# NVHPC bundled MPI must use mpirun supplied ...
LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD = mpirun
MPIRUN_NTASK_FLAG = -np
