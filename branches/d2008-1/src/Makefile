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
###############################################################################


OBJS= d3q15.o d3q19.o model.o propagation.o collision.o \
      physics.o active.o brownian.o cio.o cmd.o ccomms.o \
      colloids.o bbl.o build.o communicate.o coords.o control.o \
      eigen3.o ewald.o free_energy.o interaction.o io_harness.o\
      potential.o phi.o phi_force.o phi_gradients.o \
      lattice.o leesedwards.o pe.o ran.o regsteer.o runtime.o \
      site_map.o subgrid.o test.o timer.o utilities.o wall.o main.o

EXE= Ludwig.exe

###############################################################################
#
#  SUN HPC section
#
###############################################################################


CC=cc
MPICC=tmcc
OPTS = -D_D3Q19_ -D_SINGLE_FLUID_ -D_COLLOIDS_ -D_COLLOIDS_TEST_ -D_COLLOIDS_TEST_AUTOCORRELATION_ 
CFLAGS=$(OPTS) -g -Xc -v -errwarn -xO3
LDFLAGS=


###############################################################################
#
# Bluegene
#
###############################################################################

#KOJAK=#/bgl/local/packages/kojak/bin/kinst-pomp
#HPCT = /bgl/local/packages/hpct_all/hpct_bgl_20070116

#CC= xlc_r
#MPICC=  mpixlc  #-I$(HPCT)/include
#COPTS= -qhalt=w -O3 -qhot #-qarch=440d -qtune=440
#LDFLAGS= -qipa #-L$(HPCT)/lib -lhpm.rts -lm -lbgl_perfctr.rts
#OPTS =  -D_D3Q19_  #-D_SINGLE_FLUID_ -D_NOISE_
#CFLAGS=  $(OPTS) $(COPTS) $(LDFLAGS)

#LIBS= -L/bgl/local/packages/mass-4.3/lib -lmass
#-L/bgl/local/lib/mpitrace -lmpitrace_c


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
	$(MAKE) generic "INC=$(INC) -I../mpi_s" "LIBS=$(LIBS) \
	-L../mpi_s -lmpi"
mpi:
	$(MAKE) generic "CC=$(MPICC)" "LIBS= $(LIBS) -lmpi" "CFLAGS=$(CFLAGS) \
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
