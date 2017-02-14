

CC=nvcc
MPICC=nvcc
CFLAGS=-O2 -arch=sm_60 -x cu -dc -Xcompiler -fopenmp  -Xptxas -v  -DKEEPFIELDONTARGET -DKEEPHYDROONTARGET -DFASTCOLLISION -DOVERLAP

AR = ar
ARFLAGS = -cru
LDFLAGS=-arch=sm_60 -Xcompiler -fopenmp

MPI_INCL=-I/opt/ibmhpc/pecurrent/mpich/gnu/include64
MPI_LIBS=-L/opt/ibmhpc/pecurrent/mpich/gnu/lib64 -lmpi -lgomp

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np


