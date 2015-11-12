##############################################################################
#
#  Makefile.mk
#
#  Define here platform-dependent information.
#  There are no targets.
#
##############################################################################

CC=nvcc
MPICC=dont
CFLAGS=-O2 -arch=sm_35 -x cu -DCUDA -DCUDAHOST -dc

OPTS=-DNP_D3Q6

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
