##############################################################################
#
#  config.mk
#
#  Cray Titan (host) AMD Opteron 6274 (Interlagos) 2.2 GHz 16 core per node
#  Each node contains 2 numa regions each with 8 cores sharing an L3.
#
#  module swap PrgEnv-cray PrgEnv-intel
#  modele swap intel intel/16.0.3.206
#
##############################################################################

ifeq ($(PE_ENV), CRAY)
  # Cray
  # msglevel_2 is cautions (rather tediously verbose)
  # msglevel_3 is warnings (ok)
  # Default C is c99

  CFLAGS_EXTRA=-h msglevel_3 -h stdc -h noomp
endif

ifeq ($(PE_ENV), INTEL)
  # Intel
  # -w2  gives errors and warnings
  # -w3  adds remarks (very verbose)

  CFLAGS_EXTRA= -w2 -strict-ansi -std=c99
endif

ifeq ($(PE_ENV), GNU)
  # GNU
  CFLAGS_EXTRA= -Wall -pedantic -std=c99
endif

CC=cc -qopenmp
MPICC=cc -qopenmp
CFLAGS=-fast -DVVL=1 -DNDEBUG
