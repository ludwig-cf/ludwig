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
TARGET  = hipcc

CC      = hipcc
CFLAGS  = -x hip -fgpu-rdc -O2

AR      = ar
ARFLAGS = -cr
LDFLAGS = -fgpu-rdc --hip-link

LAUNCH_MPIRUN_CMD =

