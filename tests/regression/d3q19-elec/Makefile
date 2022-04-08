###############################################################################
#
#  Makefile
#
#  D3Q19 electrokinetic tests. These are separate a) because they
#  are quite slow and b) because they typically require a lower
#  tolerance than other tests.
#
###############################################################################

include ../../../Makefile.mk

ifeq (${BUILD},parallel)
MPIRUN_NTASKS=1
endif

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPIRUN_CMD} ${MPIRUN_NTASK_FLAG} ${MPIRUN_NTASKS}

SOURCES = $(wildcard *.inp)
LOGS    = ${SOURCES:.inp=.new}

test:
	$(MAKE) -s clean
	$(MAKE) -s logs
	@echo End of tests.

# Run a test

%.new:  %.inp
	../../test.sh $< "${SER}" "${PAR}"

logs:	$(LOGS)

# Rule to ensure restart test runs in correct order

serial-rest-ec2.new:    serial-rest-ec1.new

clean:
	rm -f *new test-diff*
