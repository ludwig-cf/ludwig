# 
#Makefile
#

MF= Makefile

SRC_X86= targetDP_X86.c 
SRC_CUDA= targetDP_CUDA.c 




AR = ar -cru

.SUFFIXES:
.SUFFIXES: .c  .o

# CC=	gcc-4.9

CC=	gcc
NVCC= nvcc 


ifeq ($(MYHOSTNAME),mbpag3)
CC= gcc-4.6
NVCC= nvcc -ccbin=gcc-4.6
endif



 CFLAGS = -O3 -fopenmp  #-save-temps
# CFLAGS = -g -fopenmp  #-save-temps

#mic
# CC= icc
# CFLAGS_BASE= -O3 -openmp -mmic -I $(TARGETDPDIR) #-vec-report=2
# LFLAGS=$(CFLAGS_BASE)
# CFLAGS=$(CFLAGS_BASE)
#endif

 CC_CUDA=	$(NVCC)
 CFLAGS_CUDA= -DCUDA -O3 -arch=sm_21 -I $(TARGETDPDIR) -dc -x cu #-save-temps
# CFLAGS_CUDA= -DCUDA -g -G -arch=sm_21 -I $(TARGETDPDIR) -dc -x cu #-save-temps


default: targetDP_X86

all: targetDP_CUDA targetDP_X86

targetDP_CUDA:
	$(CC_CUDA) $(CFLAGS_CUDA) -c targetDP_CUDA.c
	$(AR) libtargetDP_CUDA.a targetDP_CUDA.o

targetDP_X86:
	$(CC) $(CFLAGS) -c targetDP_X86.c
	$(AR) libtargetDP_X86.a targetDP_X86.o

$(OBJ):	$(MF) *.h 

clean:
	rm -f *.o *.a
