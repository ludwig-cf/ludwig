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

CC = gcc -Wall
CXX = 
INCLUDE = -I.
AR = ar -cru

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
