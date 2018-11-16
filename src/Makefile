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
# Compiler flags.
#
# Appropriate flags for your system should be set in the parent
# directory in the file config.mk. See ../Makefile.mk
#
# Edinburgh Soft Matter and Statistical Physics Group and
# Edinburgh Parallel Computing Centre
#
# Kevin Stratford (kevin@epcc.ed.ac.uk)
# (c) 2010-2018 The University of Edinburgh
#
###############################################################################

include ../Makefile.mk

MAIN = main
EXECUTABLE = Ludwig.exe
LIBRARY = libludwig.a

OPTS =
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

HAVE_PETSC =

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
	$(MPICC) $(LDFLAGS) -o $(EXECUTABLE) main.o $(LIBRARY) $(LIBS)

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
	$(MPICC) $(MODEL) $(OPTS) $(CFLAGS) $(INCL) -c $?

.PHONY : clean
clean:
	rm -f *.o $(EXECUTABLE) $(LIBRARY)
