###############################################################################
#
#  Makefile for MPI serial stub library
#
#  Targets:
#    make lib
#    make test
#
###############################################################################

include ../Makefile.mk

LIBNAME = libmpi.a
INCLUDE = -I.

###############################################################################
#
#  Targets
#
###############################################################################

libmpi.a: lib

lib:	mpi_serial.c Makefile
	$(CC) $(INCLUDE) $(CFLAGS) -c mpi_serial.c
	$(AR) $(ARFLAGS) $(LIBNAME) mpi_serial.o

test:	mpi_tests.c $(LIBNAME) Makefile
	$(CC) $(INCLUDE) $(CFLAGS) -c mpi_tests.c
	$(CC) $(LDFLAGS) mpi_tests.o -L. -lmpi
	./a.out

clean:
	rm -rf mpi_serial.o $(LIBNAME) a.out
