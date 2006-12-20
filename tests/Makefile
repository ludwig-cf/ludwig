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
###############################################################################

#------------------------------------------------------------------------------
# Compilation options, etc.
#------------------------------------------------------------------------------

CC     = cc
MPICC  = tmcc
MPIRUN = mprun -np
NPROCS = 8

SRC    = ../src
INCLUDE = -I$(SRC)
COPTS  = -Xc -v -errwarn -D_D3Q19_

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
              test_coords.c test_model.c
TESTS = ${TESTSOURCES:.c=}

TEST_PE_OBJ = tests.o $(SRC)/pe.o
TEST_TIMER_OBJ = tests.o $(SRC)/pe.o $(SRC)/timer.o
TEST_RUNTIME_OBJ = tests.o $(SRC)/pe.o $(SRC)/runtime.o
TEST_RANDOM_OBJ = tests.o $(SRC)/pe.o $(SRC)/runtime.o $(SRC)/ran.o
TEST_COORDS_OBJ = tests.o $(SRC)/pe.o $(SRC)/runtime.o $(SRC)/coords.o
TEST_MODEL_OBJ  = $(TEST_COORDS_OBJ) $(SRC)/model.o \
		$(SRC)/d3q15.o $(SRC)/d3q19.o $(SRC)/ran.o $(SRC)/timer.o

#------------------------------------------------------------------------------
#  Rules
#------------------------------------------------------------------------------

test_assumptions: test_assumptions.o
		$(LINK) -o $@ $@.o $(CLIBS)
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
test_model:	$(TEST_MODEL_OBJ) test_model.o
		$(LINK) -o $@ $(TEST_MODEL_OBJ) $@.o $(CLIBS)

base: $(TESTS)

tests:	
	$(MAKE) base "CLIBS=$(CLIBS) $(LBLIBS)"

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

