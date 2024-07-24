###############################################################################
#
#  Makefile
#
#  D3Q15 regression tests
#
###############################################################################

include ../../../Makefile.mk

PAR=${LAUNCH_MPIRUN_CMD}

serial:
	@echo "TEST --> regression tests (d3q15)"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "" "$(PAR)"; done
clean:
	rm -f *new test-diff* input
