###############################################################################
#
#  Makefile for MPI serial stub library
#
#  Targets:
#    make libc     (C compiler)
#    make libcpp   (C++ compiler)
#
#  Note there is only one source file mpi_serial.c
#  At least some compilers require a cpp extension to behave, so there
#  is a link mpi_serial.cpp -> mpi_serial.c
#
###############################################################################

LIBNAME = libmpi.a

###############################################################################
#
#  BlueGene
#
###############################################################################

#CC = xlc_r
#CXX = xlC_r
#CFLAGS = -O2 -qhalt=w
#CXXFLAGS = $(CFLAGS)
#INCLUDE  = -I.
#AR = ar -cru

CC = gcc -Wall
INCLUDE = -I.
AR = ar -cru

#options specific to AG's development environment
ifeq ($(MYHOSTNAME),mbpag3)
CC= gcc-4.6
endif

###############################################################################
#
#  Targets
#
###############################################################################

libc:	mpi_serial.c Makefile
	$(CC) $(INCLUDE) $(CFLAGS) -c mpi_serial.c
	$(AR) $(LIBNAME) mpi_serial.o

testc:	mpi_tests.c $(LIBNAME) Makefile
	$(CC) $(INCLULE) $(CLFAGS) mpi_tests.c -L. -lmpi
	./a.out

libcpp:	mpi_serial.c Makefile
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c mpi_serial.cpp
	$(AR) $(LIBNAME) mpi_serial.o
clean:
	rm -rf mpi_serial.o libmpi.a a.out
