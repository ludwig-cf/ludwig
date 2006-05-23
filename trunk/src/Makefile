###############################################################################
#
# Makefile for LUDWIG
#
# Preprocessor flags.
#
# -D_D3Q19_              Use D3Q19 instead of D3Q15 (the default)
# -D_NOISE_              Switch on fluctuations
# -D_SINGLE_FLUID_       Use single fluid only
# -D_COLLOIDS_           Introduce particles
#
#
# make serial            for serial code
# make mpi               for parallel code
#
###############################################################################


OBJS= interaction.o main.o model.o \
      cells.o cio.o cmem.o cmd.o ccomms.o \
      colloids.o communicate.o coords.o cartesian.o control.o \
      d3q15.o d3q19.o eigen3.o free_energy.o \
      lattice.o leesedwards.o pe.o ran.o regsteer.o runtime.o \
      test.o timer.o utilities.o wall.o

EXE= Ludwig.exe

###############################################################################
#
#  SUN HPC section
#
###############################################################################

CC= cc
MPICC= tmcc

LDFLAGS= -Xc -v -errwarn -fast
OPTS = -D_SINGLE_FLUID_ -D_NOISE_ -D_D3Q19_

# For Sun perflib use -xlic_lib=sunperf

CFLAGS=  $(OPTS) $(LDFLAGS) -dalign

HOSTTYPE= sparc	# Value returned by uname (or $HOSTTYPE)

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

generic: $(OBJS)
	$(CC) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBS)

serial:
	$(MAKE) generic
mpi:
	$(MAKE) generic "CC=$(MPICC)" "LIBS=-lmpi $(LIBS)" "CFLAGS=$(CFLAGS) \
		-D_MPI_"

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
