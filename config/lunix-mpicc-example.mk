##############################################################################
#
#  lunix-mpi-example.mk
#
#  This is a parallel build using "mpicc" as the compiler wrapper
#  where we do not expect to have to set explicitly include and
#  library directories etc.
#
#  The mpirun run command e.g., "mpirun -np 4" is set in three
#  parts.
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_

CC      = mpicc
MPICC   = mpicc
CFLAGS  = -O -g -Wall

AR      = ar
ARFLAGS = -cru
LDFLAGS =

MPI_INC_PATH  =
MPI_LIB_PATH  =
MPI_LIB       =

MPIRUN_CMD          = mpirun
MPIRUN_NTASKS_FLAG  = -np
MPIRUN_NTASKS_ARG   = 4
