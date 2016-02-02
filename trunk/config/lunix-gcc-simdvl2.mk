##############################################################################
#
#  lunix-gcc-simdvl2.mk
#
#  Sets SIMD vector length to 2. This may or may not be optimal
#  for any given system. Intended for nightly tests.
#
##############################################################################

CC=gcc
MPICC=mpicc
CFLAGS=-O3 -DVVL=2

AR = ar
ARFLAGS = -cru

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
