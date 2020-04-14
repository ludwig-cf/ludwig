###############################################################################
#
#  Makefile
#
#  D3Q19 electrokinetic tests. These are separate a) because they
#  are quite slow and b) because they typically require a lower
#  tolerance than other tests.
#
###############################################################################

include ../../../Makefile.mk

ifeq (${BUILD},parallel)
MPIRUN_NTASKS=1
endif

SER=${LAUNCH_SERIAL_CMD}
PAR=${LAUNCH_MPIRUN_CMD} ${MPIRUN_NTASK_FLAG} ${MPIRUN_NTASKS}

test:
	@echo "TEST --> regression tests d3q19-elec"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "${SER}" "${PAR}"; done

clean:
	rm -f *new test-diff*
