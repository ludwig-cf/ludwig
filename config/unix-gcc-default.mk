##############################################################################
#
#  unix-gcc-default.mk
#
#  A most simple serial build using gcc only.
#
##############################################################################

BUILD   = serial
MODEL   = -D_D3Q19_
TARGET  =

CC      = gcc
CFLAGS  = -O -g -Wall

AR      = ar
ARFLAGS = -cru
LDFLAGS =

LAUNCH_MPIRUN_CMD =
