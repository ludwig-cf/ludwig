##############################################################################
#
#  lunix-nvcc-default.mk
#
#  If CUDA is required use nvcc and
#    - CFLAGS should contain appropriate flags to allow nvcc to identify
#      C source files with extension .c
#
#  If MPI is required in addition to CUDA
#     - MPICC should be set the nvcc
#     - The true location of relevant MPI header files and libraries needs
#       to be identified and set in MPI_INCL and MPI_LIBS respectively
#     - nvcc will be also used at link stage.
#
#  Running the tests requires
#     - an MPI launch command (often "mpirun")
#     - the identity of the switch which controls the number of MPI tasks
#     - a serial "launch command" (can be useful for platforms requiring
#       cross-compiled)
#       e.g., "aprun -n 1" on Cray systems. Leave blank if none is required.
#
##############################################################################
BUILD   = parallel
MODEL   = -D_D3Q19_

CC     = nvcc
CFLAGS = -ccbin=mpicc -O2 -DNDEBUG -arch=sm_61 -x cu -dc

AR = ar
ARFLAGS = -cr
LDFLAGS= -arch=sm_61

MPI_INC_PATH=-I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB_PATH=-L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD = mpirun
MPIRUN_NTASK_FLAG = -np
