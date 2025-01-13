###############################################################################
#
#  Makefile
#
###############################################################################

include ../../../Makefile.mk

PAR=${LAUNCH_MPIRUN_CMD}

SOURCES = $(wildcard *.inp)
LOGS    = ${SOURCES:.inp=.new}

test:
	$(MAKE) -s clean
	$(MAKE) -s logs
	@echo End of tests.

logs:	$(LOGS)

%.new:	%.inp
	../../test.sh $< "" "${PAR}"
clean:
	rm -f *new test-diff* *meta *001-001
