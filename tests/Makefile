##############################################################################
#
#  Makefile for test directory
#
#  Targets include:
#
#    default:         builds unit tests
#    test:            runs unit tests and d3q19-short (default test)
#    d2q9             D2Q9 tests
#    d3q15            D3Q15 tests
#    d3q19-short      a batch of shorter tests
#    d3q27            D3Q27  tests
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2015-2024 The University of Edinburgh
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

d2q9:
	$(MAKE) -C regression/d2q9

d3q15:
	$(MAKE) -C regression/d3q15

d3q19-short:
	$(MAKE) -C regression/d3q19-short

d3q27:
	$(MAKE) -C regression/d3q27

# Clean

.PHONY:	clean

clean:
	$(MAKE) -C unit clean
	$(MAKE) -C regression clean
