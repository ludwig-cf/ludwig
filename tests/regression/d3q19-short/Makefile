###############################################################################
#
#  Makefile
#
#  D3Q19 short regression tests. "Short" is not more than 10 seconds
#  on a single thread (with assertions).
#
###############################################################################

include ../../../Makefile.mk

ifeq (${BUILD},parallel)
MPIRUN_NTASKS=1
endif

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPIRUN_CMD} ${MPIRUN_NTASK_FLAG} ${MPIRUN_NTASKS}


test:
	@echo "TEST --> regression tests serial"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "${SER}" "${PAR}"; done

clean:
	rm -f *new test-diff* *meta *001-001
