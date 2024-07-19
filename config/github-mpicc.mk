##############################################################################
#
#  github-mpicc.mk
#
#  We expect mpicc driving gcc; tests run on two mpi processes.
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D2Q9_

CC      = mpicc -fopenmp
CFLAGS  = -g -Wall -O2

AR      = ar
ARFLAGS = -cru
LDFLAGS =

LAUNCH_MPIRUN_CMD = mpirun -np 2
