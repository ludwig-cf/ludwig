#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "cutil_inline.h"

const int nbThreadsPerBloc=256;

__global__ 
void addition(int size, int *d_C, int *d_A, int *d_B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid<size) {
		d_C[tid]=d_A[tid]+d_B[tid];
	}
}


int main( int argc, char** argv) 
{
	if(argc!=2) { 
		printf("usage: ex1 nb_components\n");
		exit(0);
	}

	int size=atoi(argv[1]);
	int i;
	int *h_arrayA=(int*)malloc(size*sizeof(int));
	int *h_arrayB=(int*)malloc(size*sizeof(int));
	int *h_arrayC=(int*)malloc(size*sizeof(int));
	int *h_arrayCgpu=(int*)malloc(size*sizeof(int));
	int *d_arrayA, *d_arrayB, *d_arrayC;

	cudaMalloc((void**)&d_arrayA,size*sizeof(int));
	cudaMalloc((void**)&d_arrayB,size*sizeof(int));
	cudaMalloc((void**)&d_arrayC,size*sizeof(int));

	for(i=0;i<size;i++) {
		h_arrayA[i]=i;
		h_arrayB[i]=2*i;
	}

	unsigned int timer_cpu = 0;
	cutilCheckError(cutCreateTimer(&timer_cpu));
  cutilCheckError(cutStartTimer(timer_cpu));
	for(i=0;i<size;i++) {
		h_arrayC[i]=h_arrayA[i]+h_arrayB[i];
	}
	cutilCheckError(cutStopTimer(timer_cpu));
	printf("CPU processing time : %f (ms) \n", cutGetTimerValue(timer_cpu));
	cutDeleteTimer(timer_cpu);

	unsigned int timer_gpu = 0;
	cutilCheckError(cutCreateTimer(&timer_gpu));
  cutilCheckError(cutStartTimer(timer_gpu));
	cudaMemcpy(d_arrayA,h_arrayA, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arrayB,h_arrayB, size * sizeof(int), cudaMemcpyHostToDevice);
	
	int nbBlocs=(size+nbThreadsPerBloc-1)/nbThreadsPerBloc;
	addition<<<nbBlocs,nbThreadsPerBloc>>>(size,d_arrayC,d_arrayA,d_arrayB);
	cudaMemcpy(h_arrayCgpu,d_arrayC, size * sizeof(int), cudaMemcpyDeviceToHost);

	cutilCheckError(cutStopTimer(timer_gpu));
	printf("GPU processing time : %f (ms) \n", cutGetTimerValue(timer_gpu));
	cutDeleteTimer(timer_gpu);

	for(i=0;i<size;i++)
		assert(h_arrayC[i]==h_arrayCgpu[i]);

	cudaFree(d_arrayA);
	cudaFree(d_arrayB);
	cudaFree(d_arrayC);
	free(h_arrayA);
	free(h_arrayB);
	free(h_arrayC);
	return 0;
}
