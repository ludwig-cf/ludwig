##############################################################################
#
#  Makefile for test directory
#
#  Drives compilation and running of tests for various options.
#
#  Quick start:
#    make compile-run-serial     for serial tests (default D3Q19)
#    make compile-run-mpi        for parallel
#
#  The basic controls are of the form:
#
#    make compile-serial-d3q15
#    make compile-mpi-d3q15
#
#    make run-serial-unit-d3q15
#    make run-mpi-regr-d3q15
#
#  and so on. Various consolidated targets are available for convenience.
#
#  Compilation and executaion are separated to allow use on
#  platforms where cross compilation is required.
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2015-2018 The University of Edinburgh
#  Contributing authors:
#  Kevin Stratford (kevinAepcc.ed.ac.uk)
#
##############################################################################

include ../Makefile.mk

default:
	$(MAKE) build

build:
	$(MAKE) -C unit

test:
	$(MAKE) -C unit test
	$(MAKE) d3q19-short

verbose:
	@echo \
	"\t Testing options include:\n" \
	"\t   make compile-run-serial\n" \
	"\t   make compile-run-mpi\n"  \
	"\t See the Makefile for further model-specific targets\n" \
	"\t Note: it is assumed make is GNU make\n"

d3q19-short:
	$(MAKE) -C regression/d3q19-short

# Compilation only

.PHONY:	compile-serial compile-mpi compile-serial-base compile-mpi-base

compile-mpi-base:
	$(MAKE) -C ../target
	$(MAKE) -C ../src $(LBMODEL)
	$(MAKE) -C unit $(LBMODEL)

compile-serial-base:
	$(MAKE) -C ../mpi_s
	$(MAKE) compile-mpi-base

compile-serial:
	$(MAKE) compile-serial-base "LBMODEL=serial"
compile-serial-%:
	@echo "TEST --> compile serial $*"
	$(MAKE) compile-serial-base "LBMODEL=serial-$*"

compile-mpi:
	$(MAKE) compile-mpi-base "LBMODEL=mpi"
compile-mpi-%:
	@echo "TEST --> compile mpi $*"
	$(MAKE) compile-mpi-base "LBMODEL=mpi-$*"

# Execution only

# Unit tests:        run whatever has been compiled
# Regression tests:  run whatever ... against the requested directory
#                                     which ignores any postfix "r"
#
# That means there should be no further targets ending in "r".

.PHONY:	run-serial-unit run-mpi-unit

run-serial-unit:
	$(MAKE) -k -C unit run-serial
run-serial-regr-%r:
	$(MAKE) -s run-serial-regr-$*
run-serial-regr-%:
	$(MAKE) -k -C regression/$* serial

run-mpi-unit:
	$(MAKE) -k -C unit run-mpi
run-mpi-regr-%r:
	$(MAKE) -s run-mpi-regr-$*
run-mpi-regr-%:
	$(MAKE) -k -C regression/$* mpix08

# Clean

.PHONY:	clean test-clean

clean:
	$(MAKE) -C unit clean
	$(MAKE) -C regression clean


# Compile and run tests (targets of convenience...)
# "-i" to ignore errors and let the tests continue

.PHONY:	compile-run-serial compile-run-mpi

compile-run-serial-%:
	$(MAKE) clean
	@echo "TEST --> compile-serial-$*"
	$(MAKE) compile-serial-$*
	@echo "TEST --> run-serial-unit-$*"
	$(MAKE) -i run-serial-unit
	@echo "TEST --> run-serial-regr-$*"
	$(MAKE) -k run-serial-regr-$*

compile-run-mpi-%:
	$(MAKE) clean
	@echo "TEST --> compile-mpi-$*"
	$(MAKE) compile-mpi-$*
	@echo "TEST --> run-mpi-unit-$*"
	$(MAKE)	-i run-mpi-unit
	@echo "TEST --> run-mpi-regr-$*"
	$(MAKE) -k run-mpi-regr-$*

compile-run-serial:
	make compile-run-serial-d3q19

compile-run-mpi:
	make compile-run-mpi-d3q19

# Targets to run all the tests in various configurations
# Expects a default config.mk file to set CFLAGS

.PHONY:	all all-serial all-mpi

AOS4 = -DVVL=4
SOA1 = -DADDR_SOA
AOSOA4 = -DADDR_AOSOA -DVVL=4

all:
	$(MAKE) all-serial
	$(MAKE) all-serial "CFLAGS_TEST = $(AOS4)"
	$(MAKE) all-serial "CFLAGS_TEST = $(SOA1)"
	$(MAKE) all-serial "CFLAGS_TEST = $(AOSOA4)"

all-serial:
	$(MAKE) compile-run-serial-d2q9  "CFLAGS = $(CFLAGS) $(CFLAGS_TEST)"
	$(MAKE) compile-run-serial-d3q15 "CFLAGS = $(CFLAGS) $(CFLAGS_TEST)"
	$(MAKE) compile-run-serial-d3q19 "CFLAGS = $(CFLAGS) $(CFLAGS_TEST)"

all-mpi:
	$(MAKE) compile-run-mpi-d3q19 "CFLAGS = $(CFLAGS) $(CFLAGS_TEST)"

