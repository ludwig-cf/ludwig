##############################################################################
#
#  travis-mpicc.mk
#
#  Parallel unit tests
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_

CC      = mpicc -fopenmp
CFLAGS  = -O2 -g -Wall -Werror

AR      = ar
ARFLAGS = -cru
LDFLAGS =

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD = mpirun --oversubscribe
MPIRUN_NTASK_FLAG = -np

# Unit tests only
MPIRUN_NTASK_UNIT = 4

