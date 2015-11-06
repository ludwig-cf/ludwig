##############################################################################
#
#  Makefile for test directory
#
#  Drives compilation and running of tests for various options.
#
#  Compilation and executaion are separated to allow use on
#  platforms where cross compilation is required.
#
##############################################################################

include ../Makefile.mk

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
	$(MAKE) compile-serial-base "LBMODEL=serial-$*"

compile-mpi:
	$(MAKE) compile-mpi-base "LBMODEL=mpi"
compile-mpi-%:
	$(MAKE) compile-mpi-base "LBMODEL=mpi-$*"

# Execution only

# Unit tests:        run whatever has been compiled
# Regression tests:  run whatever ... against the requested directory
#                                     which ignores any postfix "r"

run-serial-unit:
	echo "run serial unit"
run-serial-regr-%r:
	$(MAKE) -s run-serial-regr-$*
run-serial-regr-%:
	echo "Run serial regr/$*"

run-serial-%r:
	$(MAKE) -s run-serial-$*
run-serial-%:
	$(MAKE) -s run-serial-unit
	$(MAKE) -s run-serial-regr-$*

run-mpi-%r:
	$(MAKE) -s run-mpi-$*
run-mpi-%:
	$(MAKE) -C unit run-mpi
	$(MAKE) -C regression/$* run-mpi-$*

clean:
	$(MAKE) -C ../targetDP clean
	$(MAKE) -C ../mpi_s    clean
	$(MAKE) -C ../src      clean
	$(MAKE) -C unit clean

