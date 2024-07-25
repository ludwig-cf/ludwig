###############################################################################
#
#  Makefile
#
#  D2Q9 regression tests
#
###############################################################################

include ../../../Makefile.mk

PAR=${LAUNCH_MPIRUN_CMD}

d2q9:
	@echo "TEST --> regression tests (d2q9)"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "" "$(PAR)"; done
clean:
	rm -f *new test-diff* input
