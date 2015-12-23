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
#  (c) 2015 The University of Edinburgh
#  Contributing authors:
#  Kevin Stratford (kevinAepcc.ed.ac.uk)
#
##############################################################################

include ../Makefile.mk

default:
	$(MAKE) -s verbose

verbose:
	@echo \
	"\t Testing options include:\n" \
	"\t   make compile-run-serial\n" \
	"\t   make compile-run-mpi\n"  \
	"\t See the Makefile for further model-specific targets\n"

# Compilation only

compile-mpi-base:
	$(MAKE) -C ../targetDP
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

clean:
	$(MAKE) -C ../targetDP clean
	$(MAKE) -C ../mpi_s    clean
	$(MAKE) -C ../src      clean
	$(MAKE) -C unit clean

test-clean:
	$(MAKE) -C regression


# Compile and run tests (targets of convenience...)

compile-run-serial-%:
	make clean
	make compile-serial-$*
	make run-serial-unit
	make run-serial-regr-$*

compile-run-mpi-%:
	make clean
	make compile-mpi-$*
	make run-mpi-unit
	make run-mpi-regr-$*

compile-run-serial:
	make compile-run-serial-d3q19

compile-run-mpi:
	make compile-run-mpi-d3q19
