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
# COPTS     C compile time flags e.g., optimisation
# OPTS      Preprocessor options (as described above)
# LDFLAGS   Link time flags [-lm included by default]
# LIBS      External library flags
#
###############################################################################


OBJS= ../d3q15.o ../d3q19.o ../model.o ../propagation.o ../collision.o \
      ../physics.o ../active.o ../cio.o ../cmd.o ../ccomms.o \
      ../colloids.o ../bbl.o ../build.o ../communicate.o ../coords.o ../control.o \
      ../eigen3.o ../ewald.o ../free_energy.o ../interaction.o ../potential.o \
      ../lattice.o ../leesedwards.o ../pe.o ../ran.o ../regsteer.o ../runtime.o \
      ../subgrid.o ../test.o ../timer.o ../utilities.o ../wall.o mainHaloOnly.o

EXE= Ludwig.exe

###############################################################################
#
#  SUN HPC section
#
###############################################################################

#LDFLAGS= -Xc -v -errwarn -xO3
OPTS =  -D_D3Q15_ -D_SINGLE_FLUID_
COPTS = -O3
MPICC= mpicc

###############################################################################
#
# Bluegene
#
###############################################################################

KOJAK=#/bgl/local/packages/kojak/bin/kinst-pomp
HPCT = /bgl/local/packages/hpct_all/hpct_bgl_20070116

#CC= gcc
#MPICC=  mpixlc  #-I$(HPCT)/include
#COPTS= -qhalt=w -O3 -qhot #-qarch=440d -qtune=440
#LDFLAGS= -qipa #-L$(HPCT)/lib -lhpm.rts -lm -lbgl_perfctr.rts
#OPTS =  -D_D3Q19_  #-D_SINGLE_FLUID_ -D_NOISE_

#LIBS= -L/bgl/local/packages/mass-4.3/lib -lmass
##-L/bgl/local/lib/mpitrace -lmpitrace_c

###############################################################################
#
#  Home Computer
#
###############################################################################

#MPICC= mpicc

###############################################################################
#
#  Generic section
#
###############################################################################

CFLAGS=  $(OPTS) $(COPTS) $(LDFLAGS)
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
	$(MAKE) generic "CC=$(MPICC)" "LIBS= $(LIBS)" "CFLAGS=$(CFLAGS) \
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
