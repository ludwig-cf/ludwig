###############################################################################
#
#  Makefile
#  Unit tests for Ludwig (Serial and MPI)
#
#  make do_tests            compiles and runs all tests in serial
#  make do_tests_mpi        compiles and runs all tests on NPROCS processors
#
#  Compilers etc will depend on your platform
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2010-2014 The University of Edinburgh
#
###############################################################################

#------------------------------------------------------------------------------
# Compilation options, etc.
#------------------------------------------------------------------------------

CC     = gcc -g
MPICC  = mpicc
MPIRUN = mpirun -np
NPROCS = 8

SRC     = ../../src
INCLUDE = -I$(SRC)
CFLAGS  = -Wall

MPI_STUB_INCLUDE = -I../../mpi_s
MPI_STUB_LIB = -L../../mpi_s -lmpi

CLIBS  = -lm
MPILIB = -lmpi

#------------------------------------------------------------------------------
# Files
#------------------------------------------------------------------------------

TESTSOURCES = test_assumptions.c test_pe.c test_timer.c \
              test_runtime.c test_random.c \
              test_coords.c test_le.c test_io.c test_prop.c \
              test_model.c test_halo.c \
	      test_map.c \
	      test_ewald.c test_polar_active.c test_phi_ch.c \
              test_colloid.c test_colloids.c test_colloids_halo.c \
              test_colloid_sums.c test_blue_phase.c \
              test_psi.c test_psi_sor.c test_hydro.c \
              test_field.c test_field_grad.c test_nernst_planck.c \
              test_fe_electro.c test_fe_electro_symm.c test_be.c \
              test_noise.c test_build.c test_bonds.c test_lubrication.c \
              test_pair_lj_cut.c test_pair_ss_cut.c test_pair_yukawa.c \
              test_angle_cosine.c test_bond_fene.c

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
	for file in $$checks; do ./$$file; done

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

