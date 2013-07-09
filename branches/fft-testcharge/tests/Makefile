###############################################################################
#
#  Makefile
#  Test programs for Ludwig (Serial and MPI)
#
#  make do_tests            compiles and runs all tests in serial
#  make do_tests_mpi        compiles and runs all tests on NPROCS processors
#
#  Or you can compile and run each test separately. You need to set
#  -D_D3Q15_ or -D_D3Q19_ to be consistent with the choice in ../src
#  in COPTS.
#
#  Compilers etc will depend on your platform
#
#  $Id: Makefile,v 1.14 2010-11-02 17:51:22 kevin Exp $
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2010 The University of Edinburgh
#
###############################################################################

#------------------------------------------------------------------------------
# Compilation options, etc.
#------------------------------------------------------------------------------

CC     = gcc -g
MPICC  = mpicc
MPIRUN = mpirun -np
NPROCS = 8

SRC     = ../src
INCLUDE = -I$(SRC) -I../../../../p3dfft/include
OPTS    = -D_D3Q19_
#CFLAGS  = -Wall

MPI_STUB_INCLUDE = -I../mpi_s
MPI_STUB_LIB = -L../mpi_s -lmpi

FFTLIBS = -L../../../../p3dfft/lib -L../../../../fftw/lib -lp3dfft -lfftw3 -pgf90libs

CLIBS  = -lm $(FFTLIBS)
MPILIB = -lmpi


#------------------------------------------------------------------------------
# Files
#------------------------------------------------------------------------------

TESTSOURCES = test_decomp.c test_psi_fft.c

TESTS = ${TESTSOURCES:.c=}
TESTOBJECTS = ${TESTSOURCES:.c=.o}

#------------------------------------------------------------------------------
#  Rules
#------------------------------------------------------------------------------

base: $(TESTOBJECTS) tests.o test_coords_field.o
	test_sources='$(TESTS)'; \
	for file in $$test_sources; \
	do $(CC) $$file.o -o $$file tests.o test_coords_field.o \
	-L$(SRC) -lludwig $(CLIBS); \
	done

tests:	
	$(MAKE) base "INCLUDE = $(INCLUDE) $(MPI_STUB_INCLUDE)" \
	"CLIBS=$(CLIBS) $(MPI_STUB_LIB) $(LBLIBS)"

tests_mpi:
	$(MAKE) base "CC=$(MPICC)" "CLIBS=$(CLIBS) $(LBLIBS)"


do_tests: tests
	checks='$(TESTS)'; \
	for file in $$checks; do ./$$file; echo "Test was $$file"; done

do_tests_mpi: tests_mpi
	checks='$(TESTS)'; \
	for file in $$checks; do $(MPIRUN) $(NPROCS) ./$$file; echo; done


clean:
	$(RM) core *.o $(TESTS)

#------------------------------------------------------------------------------
# Other dependencies
#------------------------------------------------------------------------------

tests.o:	Makefile

#------------------------------------------------------------------------------
#  Implicit Rules
#------------------------------------------------------------------------------

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CC) $(OPTS) $(CFLAGS) $(INCLUDE) -c $*.c

