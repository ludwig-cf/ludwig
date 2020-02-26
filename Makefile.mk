##############################################################################
#
#  Makefile.mk
#
#  This Makefile gets included in all Makefiles in this directory,
#  and relevant subdirectories.
#
#  Please copy one configuration file from the ./config
#  directory to this directory (top level Ludwig directory)
#  and make any appropriate changes for your platform.
#
#  No changes should be required in this file itself.
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2015-2018 The University of Edinburgh
#
#  Contributing authors:
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
##############################################################################

ROOT_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

include $(ROOT_DIR)/config.mk

# TargetDP library

TARGET_INC_PATH = -I$(ROOT_DIR)./target
TARGET_LIB_PATH = -L$(ROOT_DIR)./target
TARGET_LIB      = -ltarget

# Serial stubs required in serial

ifneq ("$(BUILD)","serial")

MY_BUILD = "parallel version"

else

MY_BUILD = "serial version"

MPI_INC_PATH  = -I$(ROOT_DIR)./mpi_s
MPI_LIB_PATH  = -L$(ROOT_DIR)./mpi_s
MPI_LIB       = -lmpi

endif
