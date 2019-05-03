###############################################################################
#
#  Makefile
#
#  D3Q19 short regression tests. "Short" is not more than 10 seconds
#  on a single thread (with assertions).
#
###############################################################################

include ../../../Makefile.mk

NTASKS=1

SER=${LAUNCH_SERIAL_CMD}
PAR=${MPI_LAUNCH_CMD} ${MPI_NTASKS_ARG} ${NTASKS}

test:
	@echo "TEST --> regression tests serial"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "${SER}" "${PAR}"; done

clean:
	rm -f *new test-diff* *meta *001-001
