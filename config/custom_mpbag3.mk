##############################################################################
#
#  lunix-gcc-default.mk
#
#  A typical Unix-like system will use:
#    gcc   Gnu C compiler
#    mpicc Wrapper to the local MPI C compiler
#
#  Running the tests requires
#     - an MPI launch command (often "mpirun")
#     - the identity of the switch which controls the number of MPI tasks
#     - a serial "launch command" (can be useful for platforms requiring
#       cross-compiled)
#       e.g., "aprun -n 1" on Cray systems. Leave blank if none is required.
#
##############################################################################

CC=gcc
MPICC=mpicc
CFLAGS=-O2 -Wall -DKEEPHYDROONTARGET -DKEEPFIELDONTARGET

AR = ar
ARFLAGS = -cru

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
