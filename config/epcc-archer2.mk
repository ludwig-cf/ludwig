##############################################################################
#
#  archer2
#  https://www.archer2.ac.uk
#
#  PrgEnv-cray
#  PrgEnv-gnu
#
#  But prefer PrgEnv-aocc
#    - CFLAGS = -DADDR_SOA -Ofast -DNDSIMDVL=1 ...
#      for just about the best performance
#
##############################################################################

BUILD   = parallel
MODEL   = -D_D3Q19_
TARGET  =

CC      = cc -fopenmp
CFLAGS  = -g -Ofast -Wall -DNSIMDVL=1 -DADDR_AOS -DNDEBUG

LAUNCH_MPIRUN_CMD = srun --ntasks=1
