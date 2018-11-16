##############################################################################
#
#  lunix-gcc-default.mk
#
#  A typical Unix-like system will use:
#    gcc   Gnu C compiler
#    mpicc Wrapper to the local MPI C compiler
#
#  Options:
#    BUILD is either "serial" or "parallel"
#    MODEL is either "-D_D2Q9_", "-D_D3Q15_"" or "-D_D3Q19_"
#
#  Compiler switches
#    Use e.g., -DSIMBVL=4 to set the targt vector length to 4
#    Use -fopnemp for OpenMP
#
##############################################################################

BUILD   = serial
MODEL   = -D_D3Q19_

CC      = gcc
MPICC   = gcc
CFLAGS  = -O -g -Wall -fopenmp

AR      = ar
ARFLAGS = -cru
LDFLAGS = -fopenmp

MPI_INC_PATH  = ./mpi_s
MPI_LIB_PATH  = ./mpi_s
MPI_LIB       = -lmpi
