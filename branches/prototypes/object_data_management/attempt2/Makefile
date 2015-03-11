# ATTEMPT 2

TARGETDP=../../../kevin/targetDP

MF= Makefile
SRC1= main.c
OBJ1= $(SRC1:.c=.o)

CC = mpicc -Wall
CFLAGS = -I$(TARGETDP) -g -std=c99
LIBS = -lm -L$(TARGETDP) -ltargetDP_X86

NVCC= nvcc -arch=sm_35
NVCCFLAGS= -I. -I../../../kevin/mpi_s -I$(TARGETDP) -x cu -DCUDA -dc
NVCCLIBS= -L$(TARGETDP) -ltargetDP_CUDA

host1:	$(OBJ1)
	$(CC) $(LDFALGS) -o $@ $(OBJ1) $(LIBS)

gpu1:
	make gpu-x "CC=$(NVCC)" "CFLAGS=$(NVCCFLAGS)" "LIBS=$(NVCCLIBS)"

gpu-x:	$(OBJ1)
	$(CC) $(LDFLAGS) -o gpu-x $(OBJ1) $(LIBS)

.SUFFIXES:
.SUFFIXES: .c .o

.c.o: 
	$(CC) $(CFLAGS) -c $<
clean:
	rm -f $(OBJ2) gpu-x


