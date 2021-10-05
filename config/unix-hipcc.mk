##############################################################################
#
#  unix-hipcc.mk
#
#  ROCM 4.3.
#
#  -fgpu-rdc  is equivalent of nvcc -dc for relocatable device code
#             [allows multiple translation units]
#  --hip-link required at link time
#
##############################################################################

BUILD   = serial
MODEL   = -D_D3Q19_

CC      = hipcc
CFLAGS  = -x hip -fgpu-rdc -O2

AR      = ar
ARFLAGS = -cr
LDFLAGS = -fgpu-rdc --hip-link

MPI_HOME     =
MPI_INC_PATH =
MPI_LIB_PATH =

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD =
MPIRUN_NTASK_FLAG =

