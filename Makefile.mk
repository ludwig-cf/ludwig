##############################################################################
#
#  Makefile.mk
#
#  Please copy one configuration file from the ./config
#  directory to this directory (top level Ludwig directory)
#  and make any appropriate changes for your platform.
#
#  No changes should be required in this file itself.
#
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2015 The University of Edinburgh
#  Contributing authors:
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
##############################################################################

ROOT_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

include $(ROOT_DIR)/config.mk

