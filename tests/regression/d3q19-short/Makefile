###############################################################################
#
#  Makefile
#
#  D3Q19 short regression tests. "Short" is not more than 10 seconds
#  on a single thread (with assertions).
#
###############################################################################

include ../../../Makefile.mk

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPI_CMD} ${LAUNCH_MPI_NP_SWITCH}

test:
	@echo "TEST --> regression tests serial"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "" ""; done

clean:
	rm -f *new test-diff* *meta *001-001
