##############################################################################
#
#  cirrus-mpi.mk
#
#  Intel Xeon E5-2695 v4 2.1 GhZ ("Broadwell").
#  2 sockets x 18 cores (2 hardware threads per core).
#
#  module load mpt
#  module load intel-compilers-16
#
#  Note -fast is significantly more quick than -O2 for
#  broadwell than seen on some other platforms.
#
##############################################################################

CC=icc
MPICC=mpicc -cc=icc -qopenmp
CFLAGS=-fast -DNDEBUG -DVVL=4

AR = ar
ARFLAGS = -cru

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
