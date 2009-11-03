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
###############################################################################

#------------------------------------------------------------------------------
# Compilation options, etc.
#------------------------------------------------------------------------------

CC     = gcc
MPICC  = mpicc
MPIRUN = mprun -np
NPROCS = 8

SRC    = ../src
INCLUDE = -I$(SRC) $(MPI_INCLUDE)
#COPTS  = -Xc -v -errwarn -D_D3Q19_
COPTS = -Wall -D_D3Q19_

MPI_STUB_INCLUDE = -I../mpi_s
MPI_STUB_LIB = -L../mpi_s -lmpi


CLINK  =
CLIBS  = -lm
MPILIB = -lmpi
LAUNCH  =

COMPILE = $(CC) $(INCLUDE) $(CLINK) $(COPTS)
LINK    = $(CC) $(CLINK)

#------------------------------------------------------------------------------
# Files
#------------------------------------------------------------------------------

TESTSOURCES = test_assumptions.c test_pe.c test_timer.c \
              test_runtime.c test_random.c \
              test_coords.c test_io.c test_prop.c \
              test_model.c test_halo.c test_site_map.c \
	      test_phi.c

TESTS = ${TESTSOURCES:.c=}

TEST_PE_OBJ = tests.o $(SRC)/pe.o
TEST_TIMER_OBJ = tests.o $(SRC)/pe.o $(SRC)/timer.o
TEST_RUNTIME_OBJ = tests.o $(SRC)/pe.o $(SRC)/runtime.o
TEST_RANDOM_OBJ = tests.o $(SRC)/pe.o $(SRC)/runtime.o $(SRC)/ran.o
TEST_COORDS_OBJ = tests.o $(SRC)/pe.o $(SRC)/runtime.o $(SRC)/coords.o
TEST_IO_OBJ     = tests.o $(SRC)/pe.o $(SRC)/runtime.o $(SRC)/coords.o \
		$(SRC)/io_harness.o $(SRC)/util.o $(SRC)/leesedwards.o \
		$(SRC)/control.o
TEST_MODEL_OBJ  = $(TEST_COORDS_OBJ) $(SRC)/model.o $(SRC)/control.o \
		$(SRC)/d3q15.o $(SRC)/d3q19.o $(SRC)/timer.o \
                $(SRC)/io_harness.o $(SRC)/util.o $(SRC)/leesedwards.o
TEST_PROP_OBJ	= $(TEST_MODEL_OBJ) $(SRC)/propagation.o
TEST_EWALD_OBJ  = tests.o $(SRC)/pe.o $(SRC)/coords.o $(SRC)/colloids.o \
                  $(SRC)/ewald.o
TEST_SITE_MAP_OBJ = $(TEST_IO_OBJ) $(SRC)/site_map.o
TEST_PHI_OBJ      = $(TEST_MODEL_OBJ) $(SRC)/site_map.o $(SRC)/phi.o

#------------------------------------------------------------------------------
#  Rules
#------------------------------------------------------------------------------

test_assumptions: ../src/util.o test_assumptions.o
		$(LINK) -o $@ ../src/util.o $@.o $(CLIBS)
test_pe:	$(TEST_PE_OBJ) test_pe.o
		$(LINK) -o $@ $(TEST_PE_OBJ) $@.o $(CLIBS)
test_timer:	$(TEST_TIMER_OBJ) test_timer.o
		$(LINK) -o $@ $(TEST_TIMER_OBJ) $@.o $(CLIBS)
test_runtime:	$(TEST_RUNTIME_OBJ) test_runtime.o
		$(LINK) -o $@ $(TEST_RUNTIME_OBJ) $@.o $(CLIBS)
test_random:	$(TEST_RANDOM_OBJ) test_random.o
		$(LINK) -o $@ $(TEST_RANDOM_OBJ) $@.o $(CLIBS)
test_coords:	$(TEST_COORDS_OBJ) test_coords.o
		$(LINK) -o $@ $(TEST_COORDS_OBJ) $@.o $(CLIBS)
test_io:	$(TEST_IO_OBJ) test_io.o
		$(LINK) -o $@ $(TEST_IO_OBJ) $@.o $(CLIBS)
test_model:	$(TEST_MODEL_OBJ) test_model.o
		$(LINK) -o $@ $(TEST_MODEL_OBJ) $@.o $(CLIBS)
test_halo:	$(TEST_MODEL_OBJ) test_halo.o
		$(LINK) -o $@ $(TEST_MODEL_OBJ) $@.o $(CLIBS)
test_prop:	$(TEST_PROP_OBJ) test_prop.o
		$(LINK) -o $@ $(TEST_PROP_OBJ) $@.o $(CLIBS)
test_site_map:	$(TEST_SITE_MAP_OBJ) test_site_map.o
		$(LINK) -o $@ $(TEST_SITE_MAP_OBJ) $@.o $(CLIBS)
test_phi:	$(TEST_PHI_OBJ) test_phi.o
		$(LINK) -o $@ $(TEST_PHI_OBJ) $@.o $(CLIBS)

test_sigma:     test_sigma.c
		$(COMPILE) -I../mpi_s -c test_sigma.c
		$(LINK)  \
	test_sigma.o $(MPI_STUB_LIB) $(CLIBS)

test_ewald:	$(TEST_EWALD_OBJ) test_ewald.o
		$(LINK) -o $@ $(TEST_EWALD_OBJ) $@.o $(CLIBS)

base: $(TESTS)

tests:	
	$(MAKE) base "INCLUDE = $(INCLUDE) $(MPI_STUB_INCLUDE)" \
	"CLIBS=$(CLIBS) $(MPI_STUB_LIB) $(LBLIBS)"

tests_mpi:
	$(MAKE) base "CC=$(MPICC) -D_MPI_" \
	"CLIBS=$(CLIBS) $(MPILIB) $(LBLIBS)"


do_tests: tests
	checks='$(TESTS)'; \
	for file in $$checks; do $(LAUNCH) ./$$file; echo; done

do_tests_mpi: tests_mpi
	$(MAKE) do_tests "LAUNCH=$(MPIRUN) $(NPROCS)" 

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
	$(COMPILE) -c $*.c

