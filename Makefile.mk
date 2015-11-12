##############################################################################
#
#  Makefile.mk
#
#  Define here platform-dependent information.
#  There are no targets.
#
##############################################################################

CC=gcc
MPICC=mpicc
CFLAGS=-O2

OPTS=-DNP_D3Q6

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
