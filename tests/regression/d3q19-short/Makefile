###############################################################################
#
#  Makefile
#
#  D3Q19 short regression tests. "Short" is not more than 10 seconds
#  on a single thread (with assertions).
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

logs:	$(LOGS)

%.new:	%.inp
	../../test.sh $< "${SER}" "${PAR}"

# Restart tests must be in the right order

serial-rest-c02.new:	serial-rest-c01.new

# Generate initial conditions for polymer test before running

serial-poly-st1.new: serial-poly-st1.pre

serial-poly-st1.pre:
	${SER} ${PAR} ../../../util/multi_poly_init

clean:
	rm -f *new test-diff* *meta *001-001
