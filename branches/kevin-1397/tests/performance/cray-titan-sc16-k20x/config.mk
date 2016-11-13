##############################################################################
#
#  config.mk
#
#  Cray Titan GPU
#
#  NVIDIA Kepler K20x with 14 streaming multiprocessors; transfers
#  via PCI express interface.
#
#  (host) AMD Opteron 6274 (Interlagos) 2.2 GHz 16 core per node
#
##############################################################################

CC=nvcc
MPICC=nvcc
CFLAGS=-O2 -arch=sm_35 -x cu -dc -DNDEBUG -DADDR_MODEL_R -Xptxas -v

AR = ar
ARFLAGS = -cru
LDFLAGS=-arch=sm_35

MPI_DIR = /opt/cray/mpt/default/gni/mpich-CRAY64/8.3

MPI_INCL=-I${MPI_DIR}/include

MPI_LIBS=${MPI_DIR}/lib/libmpich.a

# dependencies of mpich
MPI_LIBS+= /opt/cray/pmi/default/lib64/libpmi.a /opt/cray/alps/default/lib64/libalps.a /opt/cray/alps/default/lib64/libalpslli.a /opt/cray/alps/default/lib64/libalpsutil.a /opt/cray/dmapp/default/lib64/libdmapp.a /opt/cray/ugni/default/lib64/libugni.a /opt/cray/xpmem/default/lib64/libxpmem.a /opt/cray/wlm_detect/default/lib64/libwlm_detect.a /opt/cray/udreg/default/lib64/libudreg.a

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
