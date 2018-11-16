###############################################################################
#
#  Makefile
#
#  D3Q19 extra serial regression tests. These take too long to
#  be consdiered "short".
#
###############################################################################

include ../../../Makefile.mk

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPI_CMD} ${LAUNCH_MPI_NP_SWITCH}

test:
	@echo "TEST --> regression tests d3q19-extra"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "" ""; done

clean:
	rm -f *new test-diff* *meta *001-001
