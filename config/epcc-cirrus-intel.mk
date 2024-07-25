##############################################################################
#
#  cirrus-mpi.mk
#
#  Intel Xeon E5-2695 v4 2.1 GhZ ("Broadwell").
#  2 sockets x 18 cores (2 hardware threads per core).
#
#  module load intel-mpi-18
#  module load intel-compilers-18
#
#  Note -fast is significantly quicker than -O2 for
#  broadwell than seen on some other platforms.
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_

CC     = mpiicc -qopenmp
CFLAGS = -fast -DNDEBUG -DNSIMDVL=4

AR = ar
ARFLAGS = -cru

LAUNCH_MPIRUN_CMD = mpirun -np 1
