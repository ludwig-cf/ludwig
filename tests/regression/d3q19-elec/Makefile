###############################################################################
#
#  Makefile
#
#  D3Q19 electrokinetic tests. These are separate a) because they
#  are quite slow and b) because they typically require a lower
#  tolerance than other terts.
#
###############################################################################

include ../../../Makefile.mk

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPI_CMD} ${LAUNCH_MPI_NP_SWITCH}

test:
	@echo "TEST --> regression tests d3q19-elec"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "" ""; done

clean:
	rm -f *new test-diff* *meta *001-001
