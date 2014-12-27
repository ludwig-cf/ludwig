# ATTEMPT 1

TARGETDP=../../../kevin/targetDP

MF= Makefile
SRC1= object1.c object1_kernel.c main.c
SRC2= $(SRC1) object2.c object2_kernel.c
OBJ1= $(SRC1:.c=.o)
OBJ2= $(SRC2:.c=.o) 


CC = mpicc -Wall
CFLAGS = -I$(TARGETDP) -g
LIBS = -lm -L$(TARGETDP) -ltargetDP_X86

NVCC= nvcc -arch=sm_35
NVCCFLAGS= -I. -I../../../kevin/mpi_s -I$(TARGETDP) -x cu -DCUDA -dc
NVCCLIBS= -L$(TARGETDP) -ltargetDP_CUDA

host1:	$(OBJ1)
	$(CC) $(LDFALGS) -o $@ $(OBJ1) $(LIBS)

host2:	$(OBJ2)
	$(CC) $(LDFLAGS) -o $@ $(OBJ2) $(LIBS);

gpu2:
	make gpu-2 "CC=$(NVCC)" "CFLAGS=$(NVCCFLAGS)" "LIBS=$(NVCCLIBS)"

gpu-2:	$(OBJ2)
	$(CC) $(LDFLAGS) -o gpu2 $(OBJ2) $(LIBS)

.SUFFIXES:
.SUFFIXES: .c .o

.c.o: 
	$(CC) $(CFLAGS) -c $<
clean:
	rm -f $(OBJ2) gpu2


