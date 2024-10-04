##############################################################################
#
#  unix-nvcc-default.mk
#
#  This example uses "nvcc -ccbin=mpicc" to drive the compilation.
#  Further include/library information may be required...
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_
TARGET  = nvcc

CC     = nvcc
CFLAGS = -ccbin=mpicc -O2 -DADDR_SOA -arch=sm_61 -x cu -dc

AR = ar
ARFLAGS = -cr
LDFLAGS= -arch=sm_61

MPI_INC_PATH=-I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB_PATH=-L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD = mpirun
MPIRUN_NTASK_FLAG = -np
