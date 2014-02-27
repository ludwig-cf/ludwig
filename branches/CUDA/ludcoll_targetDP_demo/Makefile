# 
#Makefile
#


#PROGMODEL=X86
#PROGMODEL=X86_MIC
PROGMODEL=CUDA


MF= Makefile

EXE= ludcoll

SRC= ludcoll.c ludcoll_site.c 

UTILSRC_X86= targetDP_X86.c 
UTILSRC_CUDA= targetDP_CUDA.c 






.SUFFIXES:
.SUFFIXES: .c  .o

OBJ=	$(SRC:.c=.o)

ifeq ($(PROGMODEL),X86)

 SRC += $(UTILSRC_X86)	

 CC=	gcc
CFLAGS_BASE= -O3 -fopenmp #-save-temps
#CFLAGS_BASE= -g -fopenmp
 #CC= icc
 #CFLAGS_BASE= -O3 -openmp -save-temps #-mmic -vec-report=2
 LFLAGS=$(CFLAGS_BASE)
 CFLAGS=$(CFLAGS_BASE)
endif

ifeq ($(PROGMODEL),X86_MIC)

 SRC += $(UTILSRC_X86)	

 CC= icc
 CFLAGS_BASE= -O3 -openmp -mmic #-vec-report=2
 LFLAGS=$(CFLAGS_BASE)
 CFLAGS=$(CFLAGS_BASE)
endif


ifeq ($(PROGMODEL),CUDA)

 SRC += $(UTILSRC_CUDA)	

 CC=	nvcc
 CFLAGS_BASE= -DCUDA -O3 -arch=sm_35  #-save-temps
# CFLAGS_BASE= -DCUDA -g -G -arch=sm_35 #-save-temps
 LFLAGS=$(CFLAGS_BASE)
 CFLAGS= $(CFLAGS_BASE) -dc -x cu
 LIBS= -L/usr/lib/gcc/x86_64-redhat-linux/4.4.4 -lgomp

endif



#NVIDIA compiler can only deal with .cu suffixes
ifeq ($(PROGMODEL),CUDA)		
.c.o: 
	$(CC) $(CFLAGS) -c $<

else
.cu.o:
	$(CC) $(CFLAGS) -c $<
endif


all:	$(EXE) 

$(EXE):	$(OBJ) *.h 
	$(CC) $(LFLAGS) -o $@ $(OBJ) $(LIBS)

$(OBJ):	$(MF) *.h 

clean:
	rm -f $(OBJ) $(NVOBJ) $(OCLOBJ) $(EXE) core
