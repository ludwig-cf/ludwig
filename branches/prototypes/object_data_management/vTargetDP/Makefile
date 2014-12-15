# 
#Makefile
#


TARGETDPDIR=../../../../trunk/targetDP

MF= Makefile

EXE= prog

SRC= module.c main.c kernel.c


.SUFFIXES:
.SUFFIXES: .c  .o

OBJ=	$(SRC:.c=.o)

USEGPU=
ifeq ($(TARGETDP),GPU)
USEGPU=1
endif

ifdef USEGPU
CC = nvcc -arch=sm_21
CFLAGS =  -I$(TARGETDPDIR) -x cu -DCUDA -dc #-g -G
LIBS = -lm -L$(TARGETDPDIR) -ltargetDP_CUDA
else
CC = gcc
CFLAGS = -I$(TARGETDPDIR) 
LIBS = -lm -L$(TARGETDPDIR) -ltargetDP_X86
endif

.c.o: 
	$(CC) $(CFLAGS) -c $<

all:	$(EXE) 

$(EXE):	$(OBJ) *.h 
	$(CC) $(LFLAGS) -o $@ $(OBJ) $(LIBS)

$(OBJ):	$(MF) *.h 

clean:
	rm -f $(OBJ) $(EXE) core
