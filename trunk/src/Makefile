###############################################################################
#
# Makefile for LUDWIG
#
# Preprocessor flags.
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
# make serial            for serial code (default model is D3Q19)
# make mpi               for parallel code (default model is D3Q19)
#
# make serial-d2q9       etc for serial D2Q19 or D3Q15 or D3Q19
# make mpi-d2q9          etc for parallel ditto
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
# Edinburgh Soft Matter and Statistical Physics Group and
# Edinburgh Parallel Computing Centre
#
# Kevin Stratford (kevin@epcc.ed.ac.uk)
# (c) 2010-2014 The University of Edinburgh
#
###############################################################################

MAIN = main
EXECUTABLE = Ludwig.exe
LIBRARY = libludwig.a

CC = gcc 
MPICC = mpicc
NVCC = nvcc

OPTS = -DNP_D3Q6
CFLAGS = -O2
LIBS = -lm

# TargetDP, under development
USEGPU=
ifeq ($(TARGETDP),GPU)
USEGPU=1
endif

ifdef USEGPU
TARGETCC = $(NVCC) -arch=sm_21
#TARGETOPTS =  -x cu -D_D3Q19_ -DCUDA -g -G -dc
TARGETOPTS =  $(OPTS) -x cu -DCUDA -dc #-g -G
OPTS += -DCUDAHOST
LIBS = -lm -L../targetDP -ltargetDP_CUDA  
else
TARGETCC = $(CC)
TARGETOPTS = $(OPTS)
LIBS = -lm -L../targetDP -ltargetDP_X86
endif

INC = -I. -I ../targetDP
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

OBJS = \
     physics.o physics_rt.o \
     advection.o advection_bcs.o advection_rt.o \
     blue_phase_init.o blue_phase_rt.o \
     blue_phase_beris_edwards.o \
     brazovskii.o brazovskii_rt.o \
     colloid_io.o colloids_init.o \
     colloid.o colloid_link.o colloids.o colloids_halo.o colloid_io_rt.o \
     colloid_sums.o bbl.o build.o collision_rt.o \
     colloids_rt.o lubrication.o \
     coords.o coords_field.o coords_rt.o \
     control.o distribution_rt.o \
     driven_colloid.o driven_colloid_rt.o \
     ewald.o \
     free_energy.o free_energy_vector.o free_energy_tensor.o \
     fe_electro.o fe_electro_symmetric.o \
     gradient_rt.o \
     gradient_2d_5pt_fluid.o gradient_2d_tomita_fluid.o \
     gradient_3d_7pt_solid.o \
     gradient_3d_27pt_fluid.o gradient_3d_27pt_solid.o \
      hydro_rt.o interaction.o io_harness.o \
     leslie_ericksen.o \
     lc_droplet.o lc_droplet_rt.o model_le.o map.o \
     noise.o pair_lj_cut.o pair_ss_cut.o pair_yukawa.o \
     angle_cosine.o bond_fene.o \
     phi_cahn_hilliard.o \
     phi_force_colloid.o  \
     phi_stats.o \
     polar_active.o polar_active_rt.o \
     psi.o psi_rt.o psi_stats.o psi_sor.o psi_init.o \
     psi_force.o psi_colloid.o \
     nernst_planck.o \
     psi_petsc.o psi_gradients.o \
     leesedwards.o pe.o ran.o runtime.o \
     surfactant.o surfactant_rt.o \
     symmetric_rt.o subgrid.o \
     stats_calibration.o stats_colloid.o \
     stats_distribution.o stats_free_energy.o stats_rheology.o \
     stats_sigma.o stats_symmetric.o \
     stats_surfactant.o stats_turbulent.o stats_velocity.o svn.o \
     timer.o wall.o colloids_Q_tensor.o ludwig.o

TARGETDP_OBJS = collision.o symmetric.o util.o field.o field_grad.o \
	propagation.o hydro.o model.o gradient_3d_7pt_fluid.o phi_lb_coupler.o \
	blue_phase.o phi_force_stress.o phi_force.o

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
	    "\t make serial \t\t builds serial implementation (D3Q19)\n"\
	    "\t make serial-d2q9 etc \t builds specific serial version\n" \
	    "\t make serial-d2q9r etc \t SOA data order\n" \
	    "\t make mpi \t\t builds MPI implementation (D3Q19)\n" \
	    "\t make mpi-d2q9 etc \t builds specific parallel version\n" \
            "\t Check the Makefile for further details\n"	

code:	$(MAIN).o
	$(TARGETCC) $(LDFLAGS) -o $(EXECUTABLE) $(MAIN).o $(LIBRARY) $(LIBS)

# Serial versions

serial:
	$(MAKE) serial-d3q19

serial-d2q9:
	$(MAKE) serial-model "LB=-D_D2Q9_" "LBOBJ=d2q9.o"

serial-d3q15:
	$(MAKE) serial-model "LB=-D_D3Q15_" "LBOBJ=d3q15.o"

serial-d3q19:
	$(MAKE) serial-model "LB=-D_D3Q19_" "LBOBJ=d3q19.o"

serial-d2q9r:
	$(MAKE) serial-d2q9 "OPTS= -DLB_DATA_SOA $(OPTS)"

serial-d3q15r:
	$(MAKE) serial-d3q15 "OPTS= -DLB_DATA_SOA $(OPTS)"

serial-d3q19r:
	$(MAKE) serial-d3q19 "OPTS= -DLB_DATA_SOA $(OPTS)"


serial-model:
	$(MAKE) lib
	$(MAKE) code "INC=$(INC) -I../mpi_s" "LIBS=$(LIBS) -L../mpi_s -lmpi"


# MPI versions

mpi:
	$(MAKE) mpi-d3q19

mpi-d2q9:
	$(MAKE) mpi-model "LB=-D_D2Q9_" "LBOBJ=d2q9.o"

mpi-d3q15:
	$(MAKE) mpi-model "LB=-D_D3Q15_" "LBOBJ=d3q15.o"

mpi-d3q19:
	$(MAKE) mpi-model "LB=-D_D3Q19_" "LBOBJ=d3q19.o"

mpi-d2q9r:
	$(MAKE) mpi-d2q9 "OPTS= -DLB_DATA_SOA $(OPTS)"

mpi-d3q15r:
	$(MAKE) mpi-d3q15 "OPTS= -DLB_DATA_SOA $(OPTS)"

mpi-d3q19r:
	$(MAKE) mpi-d3q19 "OPTS= -DLB_DATA_SOA $(OPTS)"


mpi-model:
	$(MAKE) libmpi
	$(MAKE) code "CC=$(MPICC)"

# Libraries

objs1:	$(OBJS) $(LBOBJ)
objs2:	$(TARGETDP_OBJS)
libar: 
	$(MAKE) objs1
	$(MAKE) objs2 "CC=$(TARGETCC)" "OPTS=$(TARGETOPTS)"
	$(AR) $(ARFLAGS) $(LIBRARY) $(OBJS) $(TARGETDP_OBJS) $(LBOBJ)

lib:
	$(MAKE) libar "INC=$(INC) -I../mpi_s" "LIBS= -L../mpi_s -lmpi $(LIBS)"
libmpi:
	$(MAKE) libar "CC=$(MPICC)"

# To obtain the latest svn version information, we use svnversion.
# If svnversion is not available there may be an ugly error message,
# although compilation should procede. The result should be double
# quoted, e.g. "1191", hence the apostrophe armageddon.

svn.o:
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
	$(CC) $(LB) $(OPTS) $(CFLAGS) $(INC) -c $?



.PHONY : clean
clean:
	rm -f d2q9.o d3q15.o d3q19.o
	rm -f $(OBJS) $(TARGETDP_OBJS) $(EXECUTABLE) $(LIBRARY) $(MAIN).o
