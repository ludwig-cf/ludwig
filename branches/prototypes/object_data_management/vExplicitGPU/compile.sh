nvcc -arch=sm_21 -g -G  -dc -c main.cu
nvcc -arch=sm_21 -g -G  -dc -c module.cu
nvcc -arch=sm_21 -g -G -dc -c kernel.cu
nvcc -arch=sm_21 -g -G main.o module.o kernel.o -o prog
