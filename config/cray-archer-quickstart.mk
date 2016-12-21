##############################################################################
#
#  config.mk
#
#  Cray Archer with VVL=4 and minimal target <--> host data movement
#
##############################################################################
  # Intel
  # -w2  gives errors and warnings
  # -w3  adds remarks (very verbose)

CC=cc
MPICC=cc
CFLAGS=-O2 -w2 -strict-ansi -std=c99 -openmp -DKEEPFIELDONTARGET -DKEEPHYDROONTARGET -DNDEBUG -DVVL=4 -DAOS


LDFLAGS= -openmp

LAUNCH_SERIAL_CMD=aprun -q -n 1
LAUNCH_MPI_CMD=aprun -q
LAUNCH_MPI_NP_SWITCH=-n
