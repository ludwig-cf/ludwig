###############################################################################
#
#  nvcc build
#
#  Here with GPU-aware MPI via specific OpenMPI build, which has
#  slurm support.
#
#  module load gcc
#  module load openmpi/4.1.6-cuda-12.4 
#  module load nvidia/nvhpc-nompi/24.5
#
###############################################################################

BUILD  = parallel
MODEL  = -D_D3Q19_
TARGET = nvcc

CC     = nvcc
CFLAGS = -g -DADDR_SOA -O3 -arch=sm_70 -x cu -dc -DHAVE_OPENMPI_ # -DNDEBUG

# PTX assembler extra information:  -Xptxas -v
# Alternative compiler, e.g., Intel: -ccbin=icpc -Xcompiler -fast

AR = ar
ARFLAGS = -cr
LDFLAGS = -arch=sm_70

# MPI_HOME is provided by the OpenMPI module

MPI_INC_PATH = -I${MPI_HOME}/include
MPI_LIB_PATH = -L${MPI_HOME}/lib -lmpi

# ... and has slurm support ...

LAUNCH_MPIRUN_CMD = srun --ntasks=1
