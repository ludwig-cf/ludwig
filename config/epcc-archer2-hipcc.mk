##############################################################################
#
#  epcc-archer2-hipcc.mk
#
#  ROCM 5.2.3.
#
#  module load PrgEnv-amd
#
#  -fgpu-rdc  is equivalent of nvcc -dc for relocatable device code
#             [allows multiple translation units]
#  --hip-link required at link time
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_
TARGET  = hipcc

CFLAGS_EXTRA = -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false

CC      = cc

CFLAGS  = -x hip -fgpu-rdc -O2 -DADDR_SOA $(CFLAGS_EXTRA) --offload-arch=gfx90a

AR      = ar
ARFLAGS = -cr
LDFLAGS = -fgpu-rdc --hip-link --offload-arch=gfx90a

MPI_HOME     =
MPI_INC_PATH =
MPI_LIB_PATH = -L${HIP_LIB_PATH} -lamdhip64

LAUNCH_MPIRUN_CMD =

