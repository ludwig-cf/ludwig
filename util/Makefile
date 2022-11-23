###############################################################################
#
#  Makefile
#
#  Various targets for utility programs to perform pre-processing
#  and post-processing tasks.
#
#  Note all the builds are parallel, but the utilities are
#  intended to be run in serial.
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2016-2021 The University of Edinburgh
#
#  Contributing authors:
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
###############################################################################

include ../Makefile.mk

# Compilation etc

SRC  = $(ROOT_DIR)./src
INCL = -I$(SRC) $(TARGET_INC_PATH) $(MPI_INC_PATH)

BLIBS = $(MPI_LIB_PATH) $(MPI_LIB) $(TARGET_LIB_PATH) $(TARGET_LIB)
LIBS  = $(SRC)/libludwig.a ${BLIBS} -lm


default:
	$(MAKE) build

build:
	$(MAKE) extract
	$(MAKE) extract_colloids
	$(MAKE) multi_poly_init
	$(MAKE) trisphere_init
	$(MAKE) hexasphere_init
	$(MAKE) fullerene_init

colloid_init: colloid_init.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

extract_colloids: extract_colloids.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

extract_squirmer_polymer: extract_squirmer_polymer.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

capillary: capillary.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS) -lm

extract: extract.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

coll_squ_subgrid_init: coll_squ_subgrid_init.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

multi_poly_init: multi_poly_init.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

trisphere_init: trisphere_init.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
	
hexasphere_init: hexasphere_init.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

fullerene_init: fullerene_init.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)


# Default rules

.PHONY : clean
clean:
	rm -f *.o colloid_init extract_colloids capillary extract \
	coll_squ_subgrid_init multi_poly_init trisphere_init hexasphere_init fullerene_init

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CC) $(CFLAGS) $(INCL) -c $*.c

