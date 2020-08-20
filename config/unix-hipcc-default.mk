BUILD   = serial
MODEL   = -D_D3Q19_

CC      =hipcc
CFLAGS = -ccbin=g++ -DADDR_SOA -D__HIP_PLATFORM_NVCC__ -I${HIP_DIR}/include -dc 

AR      = ar
ARFLAGS = -cr
LDFLAGS = -ccbin=g++

MPI_INC_PATH      =
MPI_LIB_PATH      =
MPI_LIB           =

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD =

