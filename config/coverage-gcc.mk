##############################################################################
#
#  coverage-gcc.mk
#
#  One for coverage
#
##############################################################################

BUILD   = serial
MODEL   = -D_D3Q27_

GCOV    = -ftest-coverage -fprofile-arcs

CC      = gcc
CFLAGS  = -fopenmp $(GCOV) -O2 -g -Wall

AR      = ar
ARFLAGS = -cru
LDFLAGS = -fopenmp $(GCOV)

MPI_INC_PATH      = ./mpi_s
MPI_LIB_PATH      = ./mpi_s
MPI_LIB           = -lmpi

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD =
