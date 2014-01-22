###############################################################################
#
# Makefile for LUDWIG
#
# Preprocessor flags.
#
#  One of:
#
# -D_D2Q9_               Use D2Q9  or
# -D_D3Q15_              Use D3Q15 or
# -D_D3Q19_              Use D3Q19
#
# Please use the following for any production run
#
# -DNDEBUG               Switch off standard C assert()
#                        The assertions can make the code much slower.
#
# -DSIMDVL=N             SIMD vector length for collision code; this
#                        defaults to 1 if not set. Optimum value will
#                        depend on hardware: try 2, 4, 8, and 16.
#                        For 2d systems, use SIMDVL=1.
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
# $Id$
#
# Edinburgh Soft Matter and Statistical Physics Group and
# Edinburgh Parallel Computing Centre
#
# Kevin Stratford (kevin@epcc.ed.ac.uk)
# (c) 2010 The University of Edinburgh
#
###############################################################################

MAIN = main
EXECUTABLE = Ludwig.exe
LIBRARY = libludwig.a

CC = gcc -g
MPICC = mpicc
OPTS = -D_D3Q19_
CFLAGS = -Wall -O3

INC = -I.
LIBS = -lm
AR = ar
ARFLAGS = -cru

###############################################################################
#
# Compilation with PETSc is enabled by setting 'PETSC = some_non_empty_value'.
#
# For explicit libraries and includes the correct paths are obtained with
# 'make getincludedirs' and 'make getlinklibs' in $PETSC_DIR.
#
###############################################################################

PETSC =
ifdef PETSC
OPTS += -DPETSC
INC  +=
LIBS +=
endif 

###############################################################################
#
#  Files
#
###############################################################################

OBJS = d2q9.o d3q15.o d3q19.o model.o propagation.o propagation_ode.o \
     collision.o physics.o physics_rt.o \
     advection.o advection_bcs.o advection_rt.o \
     blue_phase.o blue_phase_init.o blue_phase_rt.o \
     blue_phase_beris_edwards.o \
     brazovskii.o brazovskii_rt.o \
     colloid_io.o colloids_init.o \
     colloid.o colloid_link.o colloids.o colloids_halo.o colloid_io_rt.o \
     colloid_sums.o bbl.o build.o collision_rt.o \
     coords.o coords_field.o coords_rt.o \
     control.o distribution_rt.o \
     ewald.o field.o field_grad.o \
     free_energy.o free_energy_vector.o fe_electro.o fe_electro_symmetric.o \
     gradient_rt.o \
     gradient_2d_5pt_fluid.o gradient_2d_tomita_fluid.o \
     gradient_3d_7pt_fluid.o gradient_3d_7pt_solid.o \
     gradient_3d_27pt_fluid.o gradient_3d_27pt_solid.o \
     hydro.o hydro_rt.o interaction.o io_harness.o \
     leslie_ericksen.o \
     lc_droplet.o lc_droplet_rt.o model_le.o map.o \
     noise.o potential.o \
     phi_cahn_hilliard.o \
     phi_force.o phi_force_colloid.o phi_force_stress.o \
     phi_lb_coupler.o phi_stats.o \
     polar_active.o polar_active_rt.o \
     psi.o psi_rt.o psi_stats.o psi_sor.o psi_init.o \
     psi_force.o psi_colloid.o nernst_planck.o psi_petsc.o \
     leesedwards.o pe.o ran.o runtime.o \
     surfactant.o surfactant_rt.o \
     symmetric.o symmetric_rt.o subgrid.o \
     stats_calibration.o stats_colloid.o \
     stats_distribution.o stats_free_energy.o stats_rheology.o \
     stats_sigma.o stats_symmetric.o \
     stats_surfactant.o stats_turbulent.o stats_velocity.o svn.o \
     timer.o util.o wall.o colloids_Q_tensor.o ludwig.o

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
	    "\t make mpi \t builds parallel implementation based on MPI\n"\
	    "\t make lib \t builds serial library\n"\
            "\t make libmpi \t builds parallel library\n"\
            "\t Check the Makefile for further details\n"	

code:	svn $(OBJS)
	$(CC) $(LDFLAGS) -o $(EXECUTABLE) $(OBJS) $(LIBS)
serial:
	$(MAKE) code "INC=$(INC) -I../mpi_s" "LIBS=$(LIBS) \
	-L../mpi_s -lmpi" "OBJS=$(OBJS) $(MAIN).o"
mpi:
	$(MAKE) code "OBJS=$(OBJS) $(MAIN).o" "CC=$(MPICC)"

libar:	$(OBJS)
	$(AR) $(ARFLAGS) $(LIBRARY) $(OBJS)
lib:
	$(MAKE) libar "INC=$(INC) -I../mpi_s" "LIBS= -L../mpi_s -lmpi $(LIBS)"
libmpi:
	$(MAKE) libar "CC=$(MPICC)"

# To obtain the latest svn version information, we use svnversion.
# If svnversion is not available there may be an ugly error message,
# although compilation should procede. The result should be double
# quoted, e.g. "1191", hence the apostrophe armageddon.

svn:
	$(CC) -D SVN_REVISION='"'`svnversion`'"' -c svn.c

###############################################################################
#
#  Compilation rules
#
###############################################################################


#
#  Default rule for compilation in the current directory.
#
.c.o :
	$(CC) $(OPTS) $(CFLAGS) $(INC) -c $?


.PHONY : clean
clean:
	-rm -f $(OBJS) $(EXECUTABLE) $(LIBRARY) $(MAIN).o
