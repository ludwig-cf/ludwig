##############################################################################
#
#  unix-mpicc-default.mk
#
#  Compile for parallel execution assuming an mpi compiler
#  wrapper "mpicc" is available.
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_

CC      = mpicc
CFLAGS  = -O -g


MPI_INC_PATH      =
MPI_LIB_PATH      =
MPI_LIB           =

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD = mpirun
MPIRUN_NTASK_FLAG = -np
