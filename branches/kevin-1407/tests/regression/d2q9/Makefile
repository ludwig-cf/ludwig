###############################################################################
#
#  Makefile
#
#  D2Q9 regression tests
#
###############################################################################

include ../../../Makefile.mk

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPI_CMD} ${LAUNCH_MPI_NP_SWITCH}

serial:
	@echo "TEST --> regression tests serial d2q9"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "" ""; done

mpix01:
	@echo "TEST --> regression tests mpi d2q9 (1 mpi task)"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "$(SER)" "$(PAR) 1"; done

mpix08:
	@echo "TEST --> regression tests mpi d2q9 (8 mpi tasks)"

mpix64:
	@echo "TEST --> regression test mpi d2q9 (64 mpi tasks)"

clean:
	rm -f *new test-diff*
