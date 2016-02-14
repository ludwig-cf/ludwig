
CC=cc
MPICC=cc
CFLAGS=-O2 -fopenmp -DKEEPFIELDONTARGET -DKEEPHYDROONTARGET #-DVVL=4 -DAOSOA

AR = ar
ARFLAGS = -cru
LDFLAGS= -fopenmp

MPI_INCL=-I/opt/cray/mpt/default/gni/mpich2-CRAY64/8.3/include

#cray mpich
MPI_LIBS=
#dependencies of mpich
MPI_LIBS+=

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
