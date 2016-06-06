##############################################################################
#
# Intel Xeon Phi
#
##############################################################################

CC=icc
MPICC=mpiicc
CFLAGS=-O2 -openmp -mmic -DNDEBUG -DKEEPHYDROONTARGET -DKEEPFIELDONTARGET -DVVL=8 -DAOSOA -opt-streaming-cache-evict=0 -opt-streaming-stores always -qopt-report
LDFLAGS= -openmp -mmic
AR = ar
ARFLAGS = -cru

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
