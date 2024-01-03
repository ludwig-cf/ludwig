###############################################################################
#
#  Makefile
#
#  D3Q15 regression tests
#
###############################################################################

include ../../../Makefile.mk

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPI_CMD} ${LAUNCH_MPI_NP_SWITCH}

serial:
	@echo "TEST --> regression tests serial"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "" ""; done
clean:
	rm -f *new test-diff* input
