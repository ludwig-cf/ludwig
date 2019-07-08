##############################################################################
#
#  lunix-gcc-travis.mk
#
#  gcc, no OpenMP
#
##############################################################################

BUILD   = serial
MODEL   = -D_D3Q19_

CC      = gcc
MPICC   = gcc
CFLAGS  = -O -g -Wall

AR      = ar
ARFLAGS = -cru
LDFLAGS =

MPI_INC_PATH  = ./mpi_s
MPI_LIB_PATH  = ./mpi_s
MPI_LIB       = -lmpi
