##############################################################################
#
#  Makefile.mk
#
#  Define here platform-dependent information.
#  There are no targets.
#
##############################################################################

CC=cc
MPICC=cc
CFLAGS=-O0 -g

OPTS=-DNP_D3Q6

MPILAUNCHCMD=aprun
MPILAUNCH-NP=-n
