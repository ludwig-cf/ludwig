###############################################################################
#
#  Makefile
#  Test programs for Ludwig (Serial and MPI)
#
#  make do_tests            compiles and runs all tests in serial
#  make do_tests_mpi        compiles and runs all tests on NPROCS processors
#
#  Or you can compile and run each test separately. You need to set
#  -D_D3Q15_ or -D_D3Q19_ to be consistent with the choice in ../src
#  in COPTS.
#
#  Compilers etc will depend on your platform
#
#  $Id: Makefile,v 1.14 2010-11-02 17:51:22 kevin Exp $
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2010 The University of Edinburgh
#
###############################################################################

#------------------------------------------------------------------------------
# Compilation options, etc.
#------------------------------------------------------------------------------

CC     = gcc -g
MPICC  = mpicc
MPIRUN = mpirun -np
NPROCS = 8

SRC     = ../src
INCLUDE = -I$(SRC) -I../../../../p3dfft/include
OPTS    = -D_D3Q19_
#CFLAGS  = -Wall

MPI_STUB_INCLUDE = -I../mpi_s
MPI_STUB_LIB = -L../mpi_s -lmpi

#FFTLIBS = -L../../../../p3dfft/lib -L../../../../fftw/lib -lp3dfft -lfftw3 -pgf90libs

#MORAR
MPICC = mpicc
INC = -I. -I../../../../p3dfft/include/
FFTLIBS = -lm ../../../../p3dfft/lib/libp3dfft.a ../../../../fftw/lib/libfftw3.a -pgf90libs

#HECToR
#MPICC = cc
#INC = -I. -I../../build/p3dfft-2.5/include/
#FFTLIBS = -lm ../../build/p3dfft-2.5/lib/libp3dfft.a /opt/fftw/3.3.0.1/interlagos/lib/libfftw3.a -pgf90libs

#BlueGene/Q
#MPICC = mpixlc_r
#INC = -I. -I${HOME}/p3dfft/include/
#FFTLIBS = -lm  ${HOME}/p3dfft/lib/libp3dfft.a /opt/ibmmath/lib64/libesslbg.a -L/bgsys/drivers/V1R1M2/ppc64/comm/xl/lib -L/bgsys/drivers/V1R1M2/ppc64/comm/sys/lib -L/bgsys/drivers/V1R1M2/ppc64/spi/lib -L/opt/ibmcmp/xlsmp/bg/3.1/bglib64 -L/opt/ibmcmp/xlmass/bg/7.3/bglib64 -L/opt/ibmcmp/xlf/bg/14.1/bglib64 -R/opt/ibmcmp/lib64/bg/bglib64 -L/bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6 -L/bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc -L/bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6/../../../../powerpc64-bgq-linux/lib -lmpich -lopa -lmpl -lpami -lSPI -lSPI_cnk -lpthread -lrt -lstdc++ -lmpichf90 -lxlf90_r -lxlopt -lxlomp_ser -lxl -lxlfmath -ldl -lm -lnss_files -lnss_dns -lresolv

CLIBS  = -lm $(FFTLIBS)
MPILIB = -lmpi


#------------------------------------------------------------------------------
# Files
#------------------------------------------------------------------------------

TESTSOURCES = test_decomp.c test_psi_fft.c \
							test_assumptions.c test_pe.c test_timer.c \
              test_runtime.c test_random.c \
              test_coords.c test_le.c test_io.c test_prop.c \
              test_model.c test_halo.c \
							test_magnetic.c test_map.c \
							test_ewald.c test_polar_active.c test_phi_ch.c \
              test_colloid.c test_colloids.c test_colloids_halo.c \
              test_colloid_sums.c test_blue_phase.c test_fluctuations.c \
              test_collision.c test_psi.c test_psi_sor.c test_hydro.c \
              test_field.c test_field_grad.c test_nernst_planck.c \
              test_fe_electro.c test_fe_electro_symm.c

TESTS = ${TESTSOURCES:.c=}
TESTOBJECTS = ${TESTSOURCES:.c=.o}

#------------------------------------------------------------------------------
#  Rules
#------------------------------------------------------------------------------

base: $(TESTOBJECTS) tests.o test_coords_field.o
	test_sources='$(TESTS)'; \
	for file in $$test_sources; \
	do $(CC) $$file.o -o $$file tests.o test_coords_field.o \
	-L$(SRC) -lludwig $(CLIBS); \
	done

tests:	
	$(MAKE) base "INCLUDE = $(INCLUDE) $(MPI_STUB_INCLUDE)" \
	"CLIBS=$(CLIBS) $(MPI_STUB_LIB) $(LBLIBS)"

tests_mpi:
	$(MAKE) base "CC=$(MPICC)" "CLIBS=$(CLIBS) $(LBLIBS)"


do_tests: tests
	checks='$(TESTS)'; \
	for file in $$checks; do ./$$file; echo "Test was $$file"; done

do_tests_mpi: tests_mpi
	checks='$(TESTS)'; \
	for file in $$checks; do $(MPIRUN) $(NPROCS) ./$$file; echo; done


clean:
	$(RM) core *.o $(TESTS)

#------------------------------------------------------------------------------
# Other dependencies
#------------------------------------------------------------------------------

tests.o:	Makefile

#------------------------------------------------------------------------------
#  Implicit Rules
#------------------------------------------------------------------------------

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CC) $(OPTS) $(CFLAGS) $(INCLUDE) -c $*.c

