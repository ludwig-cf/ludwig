###############################################################################
#
# Makefile for ./src
#
# You should not need to edit this file.
#
# Appropriate flags for your system should be set in the parent
# directory in the file config.mk. See ../Makefile.mk
#
# Edinburgh Soft Matter and Statistical Physics Group and
# Edinburgh Parallel Computing Centre
#
# Kevin Stratford (kevin@epcc.ed.ac.uk)
# (c) 2010-2023 The University of Edinburgh
#
###############################################################################

include ../Makefile.mk

MAIN = main
EXECUTABLE = Ludwig.exe
LIBRARY = libludwig.a

LIBS = $(MPI_LIB_PATH) $(MPI_LIB) $(TARGET_LIB_PATH) $(TARGET_LIB) -lm
INCL = $(MPI_INC_PATH) $(TARGET_INC_PATH)

###############################################################################
#
#  Files
#
###############################################################################

SOURCES = $(wildcard *.c)
OBJECTS = ${SOURCES:.c=.o}

LIBOBJECTS = $(filter-out main.o, $(OBJECTS))

###############################################################################
#
# Compilation with PETSc is enabled by setting HAVE_PETSC.
#
# For explicit libraries and includes the correct paths are obtained with
# 'make getincludedirs' and 'make getlinklibs' in $PETSC_DIR.
#
###############################################################################

ifdef HAVE_PETSC
OPTS += -DPETSC
INCL += $(PETSC_INC)
LIBS += $(PETSC_LIB)
endif 

###############################################################################
#
#  Targets
#
###############################################################################

default:
	$(MAKE) build

verbose:
	echo -e "LUDWIG make options ...\n"\
            "\t Check the Makefile for further details\n"	

build:	lib main.o
	$(CC) $(LDFLAGS) -o $(EXECUTABLE) main.o $(LIBRARY) $(LIBS)

# Library

lib:	$(LIBOBJECTS)
	$(AR) $(ARFLAGS) $(LIBRARY) $(LIBOBJECTS)

###############################################################################
#
#  Compilation rules
#
###############################################################################

#
#  Default rule for compilation in the current directory.
#

.c.o :
	$(CC) $(MODEL) $(OPTS) $(CFLAGS) $(INCL) -c $?

# Special case for compile-time information

GIT_REVISION = $(shell git rev-parse HEAD)

compiler.o:	compiler.c
	$(CC) $(CFLAGS) -DGIT_HASH=$(GIT_REVISION) -DCFLAGS="$(CFLAGS)" -c $<

# Special case for no optimaisation

util_sum.o:	util_sum.c
	$(CC) $(CFLAGS) $(INCL) -O0 -c util_sum.c

.PHONY : clean
clean:
	rm -f *.o $(EXECUTABLE) $(LIBRARY)
	rm -f *.gcno *.gcda
