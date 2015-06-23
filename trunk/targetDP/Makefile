# 
#Makefile
#

MF= Makefile

SRC_C= targetDP_C.c 
SRC_CUDA= targetDP_CUDA.c 




AR = ar -cru

.SUFFIXES:
.SUFFIXES: .c  .o

# CC=	gcc-4.9

CC=	gcc
NVCC= nvcc 
CUARCH= sm_35

ifeq ($(MYHOSTNAME),mbpag3)
CC= gcc-4.9
NVCC= nvcc #-ccbin=gcc-4.9
CUARCH=sm_30
endif



 CFLAGS = -O3 -fopenmp  #-save-temps
# CFLAGS = -g -G #-save-temps
# CFLAGS = -g -fopenmp  #-save-temps

#mic
# CC= icc
# CFLAGS_BASE= -O3 -openmp -mmic -I $(TARGETDPDIR) #-vec-report=2
# LFLAGS=$(CFLAGS_BASE)
# CFLAGS=$(CFLAGS_BASE)
#endif

 CC_CUDA=	$(NVCC)
 CFLAGS_CUDA= -DCUDA -O3 -arch=$(CUARCH) -I $(TARGETDPDIR) -dc -x cu #-save-temps
# CFLAGS_CUDA= -DCUDA -g -G -arch=sm_30 -I $(TARGETDPDIR) -dc -x cu #-save-temps


default: targetDP_C

all: targetDP_CUDA targetDP_C

targetDP_CUDA:
	$(CC_CUDA) $(CFLAGS_CUDA) -c targetDP_CUDA.c
	$(AR) libtargetDP_CUDA.a targetDP_CUDA.o

targetDP_C:
	$(CC) $(CFLAGS) -c targetDP_C.c
	$(AR) libtargetDP_C.a targetDP_C.o

$(OBJ):	$(MF) *.h 

clean:
	rm -f *.o *.a
