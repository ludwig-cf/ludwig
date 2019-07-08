##############################################################################
#
#  Makefile.mk
#
#  (c) 2015-2018 The University of Edinburgh
#
#  Contributing authors:
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
##############################################################################

include ./Makefile.mk

default:
	$(MAKE) build

# With the current compiler build the current target

serial:
	$(MAKE) -C mpi_s test

build:
	@echo "Build ${MY_BUILD} -> ${BUILD}"
	$(MAKE) -C target
	$(MAKE) -C src
	$(MAKE) -C tests
	$(MAKE) -C util

test:
	$(MAKE) -C target test
	$(MAKE) -C tests test

clean:
	$(MAKE) -C mpi_s clean
	$(MAKE) -C target clean
	$(MAKE) -C src clean
	$(MAKE) -C tests clean
	$(MAKE) -C util clean