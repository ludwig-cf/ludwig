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

CC     = gcc
MPICC  = mpicc
MPIRUN = mpirun -np
NPROCS = 8

SRC     = ../src
INCLUDE = -I$(SRC)
OPTS    = -D_D3Q19_
CFLAGS  = -Wall

MPI_STUB_INCLUDE = -I../mpi_s
MPI_STUB_LIB = -L../mpi_s -lmpi

CLIBS  = -lm
MPILIB = -lmpi

#------------------------------------------------------------------------------
# Files
#------------------------------------------------------------------------------

TESTSOURCES = test_assumptions.c test_pe.c test_timer.c \
              test_runtime.c test_random.c \
              test_coords.c test_le.c test_io.c test_prop.c \
              test_model.c test_halo.c test_site_map.c \
	      test_phi.c test_le.c test_magnetic.c \
	      test_ewald.c test_polar_active.c test_phi_ch.c \
              test_colloid.c test_colloids.c test_colloids_halo.c \
              test_colloid_sums.c test_blue_phase.c

TESTS = ${TESTSOURCES:.c=}
TESTOBJECTS = ${TESTSOURCES:.c=.o}

#------------------------------------------------------------------------------
#  Rules
#------------------------------------------------------------------------------

base: $(TESTOBJECTS) tests.o
	test_sources='$(TESTS)'; \
	for file in $$test_sources; \
	do $(CC) $$file.o -o $$file tests.o \
	-L$(SRC) -lludwig $(CLIBS); \
	done

tests:	
	$(MAKE) base "INCLUDE = $(INCLUDE) $(MPI_STUB_INCLUDE)" \
	"CLIBS=$(CLIBS) $(MPI_STUB_LIB) $(LBLIBS)"

tests_mpi:
	$(MAKE) base "CC=$(MPICC) -D_MPI_" \
	"CLIBS=$(CLIBS) $(MPILIB) $(LBLIBS)"


do_tests: tests
	checks='$(TESTS)'; \
	for file in $$checks; do ./$$file; echo; done

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

