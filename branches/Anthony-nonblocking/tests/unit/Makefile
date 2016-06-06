###############################################################################
#
#  Makefile
#
#  Unit tests for Ludwig (Serial and MPI)
#
#  The targets follow those in the src directory and the appropriate
#  model library must be available. In addition:
#
#  make run-serial
#  make run-mpi
#
#  with compile and run the tests (MPI using NPROCS set below)
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2010-2015 The University of Edinburgh
#
###############################################################################

include ../../Makefile.mk

#------------------------------------------------------------------------------
# Compilation options, etc.
#------------------------------------------------------------------------------

NPROCS = 8
MPIRUN = ${LAUNCH_MPI_CMD} ${LAUNCH_MPI_NP_SWITCH} $(NPROCS)

SRC     = ../../src
INCLUDE = -I$(SRC) -I../../targetDP

MPI_STUB_INCLUDE = -I../../mpi_s
MPI_STUB_LIB = -L../../mpi_s -lmpi

CLIBS  = -lm -L../../targetDP -ltarget
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
              test_angle_cosine.c test_bond_fene.c test_util.c

TESTS = ${TESTSOURCES:.c=}
TESTOBJECTS = ${TESTSOURCES:.c=.o}

#------------------------------------------------------------------------------
#  Rules
#------------------------------------------------------------------------------

base-me: $(TESTOBJECTS) tests.o test_coords_field.o
	$(CC) $(LDFLAGS) -L$(SRC) $(TESTOBJECTS) tests.o test_coords_field.o \
        -L$(SRC) -lludwig $(CLIBS)

serial:
	$(MAKE) serial-d3q19

serial-d2q9:
	$(MAKE) serial-test "LB=-D_D2Q9_"
serial-d3q15:
	$(MAKE) serial-test "LB=-D_D3Q15_"
serial-d3q19:
	$(MAKE) serial-test "LB=-D_D3Q19_"

serial-d2q9r:	serial-d2q9
serial-d3q15r:	serial-d3q15
serial-d3q19r:	serial-d3q19

serial-test:	
	$(MAKE) base-me "INCLUDE = $(INCLUDE) $(MPI_STUB_INCLUDE)" \
	"CLIBS=$(CLIBS) $(MPI_STUB_LIB) $(LBLIBS)"

mpi:	
	$(MAKE) mpi-d3q19

mpi-d2q9:
	$(MAKE) mpi-test "LB=-D_D2Q9_"
mpi-d3q15:
	$(MAKE) mpi-test "LB=-D_D3Q15_"
mpi-d3q19:
	$(MAKE) mpi-test "LB=-D_D3Q19_"

mpi-d2q9r:	mpi-d2q9
mpi-d3q15r:	mpi-d3q15
mpi-d3q19r:	mpi-d3q19

mpi-test:
	$(MAKE) base-me "CC=$(MPICC)" "CLIBS=$(CLIBS) $(LBLIBS)"


run-serial:
	$(LAUNCH_SERIAL_CMD) ./a.out

run-mpi:
	$(MPIRUN) ./a.out

clean:
	$(RM) core *.o a.out

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
	$(CC) $(LB) $(OPTS) $(CFLAGS) $(INCLUDE) -c $*.c

