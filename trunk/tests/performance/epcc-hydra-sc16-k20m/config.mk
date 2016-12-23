##############################################################################
#
#  config.mk
#
#  EPCC hydra system
#
#  Kepler0: host   Intel Xeon E5-2670 2.60 GHz 2x16 cores
#           device 4x NVDIA K20m
#
##############################################################################

CC=nvcc
MPICC=nvcc
CFLAGS=-O2 -arch=sm_35 -x cu -dc -DADDR_MODEL_R -DNDEBUG

AR = ar
ARFLAGS = -cru
LDFLAGS=-arch=sm_35

MPI_INCL=-I/opt/intel/impi/5.0.3.048/intel64/include
MPI_LIBS=-L/opt/intel/impi/5.0.3.048/intel64/lib -lmpi

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
