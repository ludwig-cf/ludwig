###############################################################################
#
# Makefile for LUDWIG
#
# Preprocessor flags.
#
# -D_D3Q19_              Use D3Q19 instead of D3Q15 (the default)
# -D_NOISE_              Switch on fluctuations
# -D_SINGLE_FLUID_       Use single fluid only
# -DNDEBUG               Switch off standard C assert()
#
# make serial            for serial code
# make mpi               for parallel code
#
###############################################################################


OBJS= d3q15.o d3q19.o model.o propagation.o collision.o \
      active.o cio.o cmd.o ccomms.o \
      colloids.o bbl.o build.o communicate.o coords.o control.o \
      eigen3.o free_energy.o interaction.o \
      lattice.o leesedwards.o pe.o ran.o regsteer.o runtime.o \
      test.o timer.o utilities.o wall.o main.o

EXE= Ludwig.exe

###############################################################################
#
#  SUN HPC section
#
###############################################################################

CC= cc
MPICC= tmcc

LDFLAGS= -Xc -v -errwarn -fast
OPTS = -D_D3Q15_

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
