###############################################################################
#
#  NVIDIA DGX-1 (Atos) "jade.hartree.stfc.ac.uk"
#  8 x P100 per node
#
#  Serial:
#    module load cuda/9.0
#    module load pgi/17.4
#
#  Parallel; in addition, set:
#    module load openmpi/1.10.2/2017
#  set MPI_HOME etc, and use nvcc -ccbin=pgc++
#
###############################################################################

CC=nvcc
MPICC=nvcc

# -Xcompiler for specific pgi flags
# -Xptxas -v for verbose output from ptx assmbler

CFLAGS= -ccbin=pgc++ -g -O3 -DADDR_SOA -DNDEBUG -arch=sm_60 -x cu -dc

AR = ar
ARFLAGS = -cr
LDFLAGS=-ccbin=pgc++ -arch=sm_60

MPI_HOME = /jmain01/apps/pgi/17.4/linux86-64/2017/mpi/openmpi-1.10.2
MPI_INCL = -I$(MPI_HOME)/include
MPI_LIBS = -L$(MPI_HOME)/lib -lmpi

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np


