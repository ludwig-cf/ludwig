##############################################################################
#
#  Makefile for test directory
#
#  Targets include:
#
#    default:         builds unit tests
#    test:            runs unit tests and d3q19-short (default test)
#    d3q19-short      a batch of shorter tests
#    d3q19-extra      a extra batch of longer tests
#    d3q19-io         a batch of tests with file I/O
#    d3q19-elec       a batch of tests for electrokinetics
#
#    d3q19-short-gpu  unit tests and d3q19-short-gpu
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2015-2019 The University of Edinburgh
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
	"\t See the Makefile list of targets\n" \
	"\t Note: it is assumed make is GNU make\n"

# Specific tests

d3q19-short:
	$(MAKE) -C regression/d3q19-short

d3q19-extra:
	$(MAKE) -C regression/d3q19-extra

d3q19-io:
	$(MAKE) -C regression/d3q19-io

d3q19-elec:
	$(Make) -C regression/d3q19-elec

# GPU target

d3q19-short-gpu:
	$(MAKE) -C regression/d3q19-short-gpu

# Clean

.PHONY:	clean

clean:
	$(MAKE) -C unit clean
	$(MAKE) -C regression clean
