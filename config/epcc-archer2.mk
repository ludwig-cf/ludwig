##############################################################################
#
#  archer2
#  https://www.archer2.ac.uk
#
#  PrgEnv-cray
#    - Use cce >= 12 to avoid problem in openmp
#    - "module load cce/12.0.3"
#
#  Same compiler options for all PrgEnv are available.
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_

CC      = cc -fopenmp
CFLAGS  = -g -O3 -Wall -DNSIMDVL=2 -DNDEBUG

MPI_INC_PATH      =
MPI_LIB_PATH      =
MPI_LIB           =

LAUNCH_SERIAL_CMD =
LAUNCH_MPIRUN_CMD = srun
MPIRUN_NTASK_FLAG = -n
