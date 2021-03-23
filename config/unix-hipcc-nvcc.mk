##############################################################################
#
#  unix-hipcc-nvcc.mk
#
#  Here we have a serial hipcc only via nvcc at the moment.
#
#  export CUDA_PATH=/usr/local/cuda-10.1
#  Note HIP version 3.3 not working with cuda-10.2 on gpulab2
#
##############################################################################

BUILD   = serial
MODEL   = -D_D3Q19_

CC      = hipcc
CFLAGS  = -O2 -arch=sm_35 -dc -D__HIP_PLATFORM_NVCC__

AR      = ar
ARFLAGS = -cru
LDFLAGS = -arch=sm_35

MPI_INC_PATH  = ./mpi_s
MPI_LIB_PATH  = ./mpi_s
MPI_LIB       = -lmpi
