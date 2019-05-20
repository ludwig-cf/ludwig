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
PAR=${MPIRUN_CMD} ${MPIRUN_NTASKS_FLAG} ${MPIRUN_NTASKS_ARG}

test:
	@echo "TEST --> regression tests d3q19-extra"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "${SER}" "${PAR}"; done

clean:
	rm -f *new test-diff* *meta *001-001
