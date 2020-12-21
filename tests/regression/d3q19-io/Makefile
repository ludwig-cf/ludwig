###############################################################################
#
#  Makefile
#
#  Various tests involving input from file
#
###############################################################################

include ../../../Makefile.mk

ifeq (${BUILD},parallel)
MPIRUN_NTASKS1=1
MPIRUN_NTASKS2=2
MPIRUN_NTASKS4=4
endif

SER=${LAUNCH_SERIAL_CMD}
PAR1=${LAUNCH_MPIRUN_CMD} ${MPIRUN_NTASK_FLAG} ${MPIRUN_NTASKS1}
PAR2=${LAUNCH_MPIRUN_CMD} ${MPIRUN_NTASK_FLAG} ${MPIRUN_NTASKS2}
PAR4=${LAUNCH_MPIRUN_CMD} ${MPIRUN_NTASK_FLAG} ${MPIRUN_NTASKS4}

test:
	@echo "TEST --> regression tests serial"
	inputs='serial*inp'; \
	for file in $$inputs; do ../../test.sh $$file "${SER}" "${PAR1}"; done
	make mpi1
	make mpi2
	make mpi4

mpi1:
	mpi1='iodrop-mpi1*inp'; \
	for file in $$mpi1; do ../../test.sh $$file "${SER}" "${PAR1}"; done
mpi2:
	mpi2='iodrop-mpi2*inp'; \
	for file in $$mpi2; do ../../test.sh $$file "${SER}" "${PAR2}"; done
mpi4:
	mpi4='iodrop-mpi4*inp'; \
	for file in $$mpi4; do ../../test.sh $$file "${SER}" "${PAR4}"; done

clean:
	rm -f *new test-diff*
	rm -f dist*001 dist*002 phi*001 vel*001
	rm -t *.meta