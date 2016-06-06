##############################################################################
#
#  lunix-nvcc-default.mk
#
#  If CUDA is required use nvcc and
#    - CFLAGS should contain appropriate flags to allow nvcc to identify
#      C source files with extension .c
#
#  If MPI is required in addition to CUDA
#     - MPICC should be set the nvcc
#     - The true location of relevant MPI header files and libraries needs
#       to be identified and set in MPI_INCL and MPI_LIBS respectively
#     - nvcc will be also used at link stage.
#
#  Running the tests requires
#     - an MPI launch command (often "mpirun")
#     - the identity of the switch which controls the number of MPI tasks
#     - a serial "launch command" (can be useful for platforms requiring
#       cross-compiled)
#       e.g., "aprun -n 1" on Cray systems. Leave blank if none is required.
#
##############################################################################

CC=nvcc
MPICC=nvcc
CFLAGS=-O2 -arch=sm_35 -x cu -dc -Xcompiler -fopenmp  -Xptxas -v  -DKEEPFIELDONTARGET -DKEEPHYDROONTARGET -DVERBOSE_PERF_REPORT  #-maxrregcount=127 #-maxrregcount=255
#CFLAGS=-O2 -arch=sm_35 -x cu -dc  -Xptxas -v  -DKEEPFIELDONTARGET -DKEEPHYDROONTARGET -DVERBOSE_PERF_REPORT  #-maxrregcount=127 #-maxrregcount=255


AR = ar
ARFLAGS = -cru
LDFLAGS=-arch=sm_35 -Xcompiler -fopenmp

MPI_INCL=-I/opt/intel/impi/5.0.3.048/intel64/include
MPI_LIBS=-L/opt/intel/impi/5.0.3.048/intel64/lib -lmpi /opt/rh/devtoolset-3/root/usr/lib/gcc/x86_64-redhat-linux/4.9.1/libgomp.a
#-L/opt/rh/devtoolset-3/root/usr/lib/gcc/x86_64-redhat-linux/4.9.1/ -lgomp


LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
