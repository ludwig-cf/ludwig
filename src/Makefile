###############################################################################
#
# Makefile for LUDWIG
#
# Preprocessor flags.
#
#  One of:
# _D_D3Q15_              Use D3Q15 or
# -D_D3Q19_              Use D3Q19
#
# -D_NOISE_              Switch on fluctuations
# -D_SINGLE_FLUID_       Use single fluid only
#
# Temporary (to be replaced by run-time options)
# -D_SUBGRID_            Sub-grid particles
# -D_ACTIVE2_            "Active2" particles
#
# Please use the following for any production run
#
# -DNDEBUG               Switch off standard C assert()
#
# make serial            for serial code
# make mpi               for parallel code
#
# Compiler flags.
#
# You will need to set the appropriate compiler flags for your
# local compiler/system of choice.
#
# CC        C compiler (make serial)
# MPICC     C compiler (make mpi)
# LDFLAGS   Link time flags [-lm included by default]
#
###############################################################################


OBJS= d3q15.o d3q19.o model.o propagation.o collision.o \
      physics.o advection.o active.o blue_phase.o \
      blue_phase_beris_edwards.o brownian.o \
      cio.o cmd.o ccomms.o \
      colloids.o bbl.o build.o communicate.o coords.o control.o \
      eigen3.o ewald.o free_energy.o interaction.o io_harness.o\
      magnetic_field.o magnetic_field_rt.o \
      model_le.o potential.o \
      phi.o phi_cahn_hilliard.o phi_force.o phi_gradients.o \
      phi_lb_coupler.o phi_stats.o \
      lattice.o leesedwards.o pe.o ran.o regsteer.o runtime.o \
      site_map.o subgrid.o \
      stats_rheology.o stats_surfactant.o stats_turbulent.o \
      test.o timer.o util.o utilities.o wall.o main.o

EXE= Ludwig.exe

###############################################################################
#
#  Compilation
#
#  Note that for the serial version you will need the dummy MPI library
#  ../mpi_s from the cvs.
#
#  You need to set, as a minimum:
#
#  CC                    Serial C compiler
#  MPICC                 MPI C compiler
#  CFLAGS                Sum total of flags given to compiler
#  LDFLAGS               Link options
#
###############################################################################

CC=cc
MPICC=cc
OPTS = -O3
LDFLAGS=
CFLAGS=  -D_D3Q19_ $(OPTS) $(LDFLAGS)

INC= -I.
LIBS= -lm

###############################################################################
#
#  Targets
#
#  Typing "make" will give some instructions.
#
###############################################################################

default:
	$(MAKE) -s verbose

verbose:
	echo -e "LUDWIG make options ...\n"\
	    "\t make serial \t builds serial implementation (no assertions)\n"\
	    "\t make serial_assert \t builds serial implementation\n"\
	    "\t make mpi \t builds parallel implementation based on MPI\n"\
	    "\t make mpi_assert \t builds MPI version with assertions\n"\
            "\t Check the Makefile for further details\n"	

generic: $(OBJS) Makefile
	$(CC) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBS)

serial:
	$(MAKE) generic "CC=$(CC) -DNDEBUG" "INC=$(INC) -I../mpi_s"\
        "LIBS=$(LIBS) -L../mpi_s -lmpi"
serial_assert:
	$(MAKE) generic "CC=$(CC)" "INC=$(INC) -I../mpi_s" "LIBS=$(LIBS) \
	-L../mpi_s -lmpi"

mpi:
	$(MAKE) generic "CC=$(MPICC) -DNDEBUG" "LIBS= $(MPILIB) $(LIBS)" \
        "CFLAGS=$(CFLAGS) -D_MPI_"
mpi_assert:
	$(MAKE) generic "CC=$(MPICC) " "LIBS= $(MPILIB) $(LIBS)" \
        "CFLAGS=$(CFLAGS) -D_MPI_"

###############################################################################
#
#  Compilation rules
#
###############################################################################


#
#  Default rule for compilation in the current directory.
#
.c.o :
	$(CC) $(CFLAGS) $(INC) -c $?


.PHONY : clean
clean:
	-rm -f $(OBJS) $(EXE)
#
# gcc -Wall can leave a mess
#
gcc-clean:
	$(RM) ${OBJS:.o=*rtl}
	$(RM) ${OBJS:.o=*jump}
	$(RM) ${OBJS:.o=*addressof}
	$(RM) ${OBJS:.o=*cfg}
	$(RM) ${OBJS:.o=*life}
	$(RM) ${OBJS:.o=*lreg}
	$(RM) ${OBJS:.o=*greg}
	$(RM) ${OBJS:.o=*postreload}
	$(RM) ${OBJS:.o=*flow2}
