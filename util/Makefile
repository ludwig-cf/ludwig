###############################################################################
#
#  Makefile
#
#  Various targets for utility programs to perform pre-processing
#  and post-processing tasks.
#
#  The Makefile does not have the dependencies encoded completely,
#  so you will need to ensure the correct serial targetDP and ludwig
#  libraries are available manually before action in this directory.
#  Intended to be run in serial.
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2016 The University of Edinburgh
#  Contributing authors:
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
###############################################################################

include ../Makefile.mk

LIBS = ../mpi_s/libmpi.a ../src/libludwig.a -L../target -ltarget -lm
INC = -I. -I../src -I../target -I../mpi_s

default:
	$(MAKE) extract

colloid_init:	colloid_init.c
	$(CC) $(CFLAGS) $(INC) colloid_init.c $(LIBS) -o colloid_init

extract_colloids:	extract_colloids.c
	$(CC) $(CFLAGS) $(INC) extract_colloids.c $(LIBS) -o extract_colloids

capillary:	capillary.c
	$(CC) $(CFLAGS) capillary.c -o capillary

extract:	extract.c
	$(CC) $(CFLAGS) $(INC) extract.c $(LIBS) -o extract

vtk_extract:	vtk_extract.c
	$(CC) $(CFLAGS) $(INC) vtk_extract.c $(LIBS) -o vtk_extract

.PHONY : clean
clean:
	rm -f colloid_init extract_colloids capillary extract vtk_extract


