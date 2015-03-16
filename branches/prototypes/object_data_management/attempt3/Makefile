# ATTEMPT 2

TARGETDP=../../../kevin/targetDP

MF= Makefile
SRC1= main.c
OBJ1= $(SRC1:.c=.o)

CC = cc -h noomp
CFLAGS = -I$(TARGETDP) 
OMPFLAG =
LIBS = -lm -L$(TARGETDP) -ltargetDP_X86

NVCC= nvcc -arch=sm_35 #--verbose
NVCCFLAGS= -I. -I../../../kevin/mpi_s -I$(TARGETDP) -x cu -DCUDA -dc
NVCCLIBS= -L$(TARGETDP) -ltargetDP_CUDA

host:	$(OBJ1)
	$(CC) $(LDFLAGS) -o host-x $(OBJ1) $(LIBS)

host-omp:
	make host "CFLAGS=${CFLAGS} ${OMPFLAG}" "LDFLAGS=${OMPFLAG}"

gpu:
	make gpu-x "CC=$(NVCC)" "CFLAGS=$(NVCCFLAGS)" "LIBS=$(NVCCLIBS)"

gpu-x:	$(OBJ1)
	$(CC) $(LDFLAGS) -o gpu-x $(OBJ1) $(NVCCLIBS)

.SUFFIXES:
.SUFFIXES: .c .o

.c.o: 
	$(CC) $(CFLAGS) -c $<
clean:
	rm -f $(OBJ1) host-x gpu-x


