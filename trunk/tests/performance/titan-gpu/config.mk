
CC=nvcc
MPICC=nvcc
CFLAGS=-O2 -arch=sm_35 -x cu -dc -DKEEPFIELDONTARGET -DKEEPHYDROONTARGET -DOVERLAP

AR = ar
ARFLAGS = -cru
LDFLAGS=-arch=sm_35

MPI_INCL=-I/opt/cray/mpt/default/gni/mpich2-CRAY64/8.3/include

#cray mpich
MPI_LIBS=/opt/cray/mpt/default/gni/mpich2-CRAY64/8.3/lib/libmpich.a
#dependencies of mpich
MPI_LIBS+= /opt/cray/pmi/default/lib64/libpmi.a /opt/cray/alps/default/lib64/libalps.a /opt/cray/alps/default/lib64/libalpslli.a /opt/cray/alps/default/lib64/libalpsutil.a /opt/cray/dmapp/default/lib64/libdmapp.a /opt/cray/ugni/default/lib64/libugni.a /opt/cray/xpmem/default/lib64/libxpmem.a /opt/cray/wlm_detect/default/lib64/libwlm_detect.a /opt/cray/udreg/default/lib64/libudreg.a

LAUNCH_SERIAL_CMD=
LAUNCH_MPI_CMD=mpirun
LAUNCH_MPI_NP_SWITCH=-np
