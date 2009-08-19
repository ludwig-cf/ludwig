###############################################################################
#
# Makefile for LUDWIG
#
###############################################################################
# retireve machine and run specific compile configuration 
include config.mk

###############################################################################


OBJS= d3q15.o d3q19.o model_ascii.o propagation.o phi_gradients.o collision.o \
      physics.o active.o brownian.o cio.o cmd.o ccomms.o \
      colloids.o bbl.o build.o communicate.o coords.o control.o \
      eigen3.o ewald.o free_energy.o interaction.o io_harness.o\
      model_le.o potential.o \
      phi.o phi_cahn_hilliard.o phi_force.o phi_stats.o \
      lattice.o leesedwards.o pe.o ran.o regsteer.o runtime.o \
      site_map.o subgrid.o stats_turbulent.o \
      test.o timer.o util.o utilities.o wall.o main.o

EXE= Ludwig.exe


###############################################################################
#
#  Generic section
#
###############################################################################

INC= -I.

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
	$(MAKE) generic "CC=$(MPICC)" "LIBS= $(MPILIB) $(LIBS)" "CFLAGS=$(CFLAGS) \
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
	$(CC) $(CFLAGS) $(INC) $(LIBS) -c $?


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
