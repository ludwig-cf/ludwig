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
# COPTS     C compile time flags e.g., optimisation
# OPTS      Preprocessor options (as described above)
# LDFLAGS   Link time flags [-lm included by default]
# LIBS      External library flags
#
###############################################################################


OBJS=d2q9.o d3q15.o d3q19.o model.o propagation.o collision.o \
     physics.o \
     advection.o advection_bcs.o advection_rt.o \
     active.o \
     blue_phase.o blue_phase_rt.o \
     blue_phase_beris_edwards.o \
     brazovskii.o brazovskii_rt.o \
     brownian.o \
     cio.o cmd.o ccomms.o \
     colloids.o bbl.o build.o communicate.o \
     coords.o coords_rt.o \
     control.o \
     eigen3.o ewald.o \
     free_energy.o free_energy_rt.o free_energy_vector.o \
     gradient.o \
     gradient_2d_5pt_fluid.o gradient_3d_7pt_fluid.o \
     gradient_3d_27pt_fluid.o gradient_3d_27pt_solid.o \
     interaction.o io_harness.o\
     leslie_ericksen.o \
     model_le.o potential.o \
     phi.o phi_cahn_hilliard.o \
     phi_force.o phi_force_colloid.o phi_gradients.o \
     phi_lb_coupler.o phi_stats.o \
     phi_update.o phi_update_rt.o \
     polar_active.o polar_active_rt.o \
     lattice.o leesedwards.o pe.o ran.o runtime.o \
     surfactant.o surfactant_rt.o \
     symmetric.o symmetric_rt.o \
     site_map.o subgrid.o \
     stats_distribution.o stats_free_energy.o stats_rheology.o \
     stats_surfactant.o stats_turbulent.o \
     test.o timer.o util.o utilities.o wall.o main.o colloids_Q_tensor.o

EXE= Ludwig.exe

###############################################################################
#
#  Linux
#
###############################################################################

CC=gcc
MPICC=mpicc
OPTS = -D_D3Q19_ -O3
CFLAGS=$(OPTS) -Wall 

###############################################################################
#
#  Generic section
#
###############################################################################


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
	    "\t make serial \t builds serial implementation\n"\
	    "\t make mpi \t builds parallel implementation based on MPI\n"\
            "\t Check the Makefile for further details\n"	

generic: $(OBJS) Makefile
	$(CC) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBS)

serial:
	$(MAKE) generic "INC=$(INC) -I../mpi_s" "LIBS=$(LIBS) \
	-L../mpi_s -lmpi"
mpi:
	$(MAKE) generic "CC=$(MPICC)" "LIBS=$(LIBS)" "CFLAGS=$(CFLAGS)"

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
