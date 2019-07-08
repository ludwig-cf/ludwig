###############################################################################
#
#  Makefile
#
#  D3Q19 extra MPI regression tests
#
###############################################################################

include ../../../Makefile.mk

MPIRUN_NTASKS=8

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPIRUN_CMD} ${MPIRUN_NTASK_FLAG} ${MPIRUN_NTASKS} ${MPIRUN_EXTRA}

default:
	@echo "TEST --> regression d3q19-mpi-short"
	inputs='*inp'; \
	for file in $$inputs; do ../../test.sh $$file "$(SER)" "${PAR}"; done

clean:
	rm -f *new test-diff*
