# ATTEMPT 4

TARGETDP=../../../kevin-1397/targetDP

MF= Makefile
SRC1= functions.c
SRC2= constants.c
OBJ1= $(SRC1:.c=.o)

CC = gcc -Wall
CFLAGS = -I$(TARGETDP) 
OMPFLAG = -fopenmp
LIBS = -lm -L$(TARGETDP) -ltarget

NVCC= nvcc -arch=sm_35 #--verbose
NVCCFLAGS= -I. -I../../../kevin/mpi_s -I$(TARGETDP) -x cu -dc
NVCCLIBS= -L$(TARGETDP) -ltarget

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


