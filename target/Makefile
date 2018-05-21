
ifneq ("","$(wildcard ../Makefile.mk)") #use global settings

include ../Makefile.mk

else

#CPU:
CC=gcc
CFLAGS=-O2 -Wall -fopenmp

#GPU
#CC=nvcc
#CFLAGS=-O2 -arch=sm_35 -x cu -dc

endif

AR = ar -cru

.SUFFIXES:
.SUFFIXES: .c  .o


TARGETSRC = target_x86.c

ifeq ($(CC),nvcc)
TARGETSRC = target_cuda.c
endif

TARGETOBJ = ${TARGETSRC:.c=.o}

default:
	$(CC) $(CFLAGS) -c $(TARGETSRC)
	$(AR) libtarget.a $(TARGETOBJ)

test:
	$(MAKE)
	$(CC) $(CFLAGS) -c test.c
	$(CC) test.o libtarget.a
clean:
	rm -f *.o *.a a.out
