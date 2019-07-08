###############################################################################
#
#  Makefile
#
#  D3Q19 regression tests
#
###############################################################################

include ../../../Makefile.mk

ifeq (${BUILD},parallel)
MPIRUN_NTASKS=1
endif

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPIRUN_CMD} ${MPIRUN_NTASK_FLAG} ${MPIRUN_NTASKS}

mpix64:
	@echo "TEST --> regression test mpi (64 mpi tasks)"
	inputs='pmpi64*inp'; \
	for file in $$inputs; do ../../test.sh $$file "$(SER)" "$(PAR) 64"; done

clean:
	rm -f *new test-diff* *meta \
	psi-00000020.001-001 vel-00000020.001-001 dist-00000020.001-001 \
	config.cds00000020.001-001
