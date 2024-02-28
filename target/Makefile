##############################################################################
#
#  Makefile
#
#  If the compiler is nvcc, the CUDA version will be built; otherwise
#  the serial or OpenMP version is built.
#
#  Targets include:
#
#  make test
#  make testmpi
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2018-2023 The University of Edinburgh
#
#  Contributing authors:
#  Alan Gray (alang@epcc.ed.ac.uk)
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
##############################################################################

ifneq ("","$(wildcard ../Makefile.mk)") #use global settings

include ../Makefile.mk

else
# can have specific settings here
endif

.SUFFIXES:
.SUFFIXES: .c  .o


TARGETSRC = target_x86.c

ifeq ($(TARGET),nvcc)
TARGETSRC = target_cuda.c
endif

ifeq ($(TARGET),hipcc)
TARGETSRC = target_hip.c
endif

TARGETOBJ = ${TARGETSRC:.c=.o}

default:
	$(CC) $(CFLAGS) -c $(TARGETSRC)
	$(AR) $(ARFLAGS) libtarget.a $(TARGETOBJ)

test:
	$(MAKE)
	$(CC) $(CFLAGS) -c test.c
	$(CC) $(LDFLAGS) test.o libtarget.a -lm
	./a.out

testmpi:
	$(MAKE)
	$(MPICC) $(MPI_INC_PATH) $(CFLAGS) -c testmpi.c
	$(MPICC) $(LDFLAGS) testmpi.o libtarget.a $(MPI_LIB_PATH) -lmpi

clean:
	rm -f *.o *.a a.out
	rm -f *gcno *gcda
