#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "cutil_inline.h"
#include <cublas_v2.h>

const int nbThreadsPerBloc=256;

__global__ 
void addition(int size, double *d_C, double *d_A, double *d_B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid<size) {
		d_C[tid]=d_A[tid]+d_B[tid];
	}
}

__global__ 
void inverse(int size, double *d_x) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid<size) {
		d_x[tid]=1./d_x[tid];
	}
}


int main( int argc, char** argv) 
{
	if(argc!=2) { 
		printf("usage: ex2 nb_components\n");
		exit(0);
	}

	int size=atoi(argv[1]);
	cublasStatus_t stat;
	cublasHandle_t handle; 
	stat=cublasCreate(&handle);
	int i;
	double *h_arrayA=(double*)malloc(size*sizeof(double));
	double *h_arrayB=(double*)malloc(size*sizeof(double));
	double *h_arrayC=(double*)malloc(size*sizeof(double));
	double *h_arrayCgpu=(double*)malloc(size*sizeof(double));
	double *d_arrayA, *d_arrayB, *d_arrayC;

	cudaMalloc((void**)&d_arrayA,size*sizeof(double));
	cudaMalloc((void**)&d_arrayB,size*sizeof(double));
	cudaMalloc((void**)&d_arrayC,size*sizeof(double));

	for(i=0;i<size;i++) {
		h_arrayA[i]=i+1;
		h_arrayB[i]=2*(i+1);
	}

	unsigned int timer_cpu = 0;
	cutilCheckError(cutCreateTimer(&timer_cpu));
  cutilCheckError(cutStartTimer(timer_cpu));
	double dot=0;
	for(i=0;i<size;i++) {
		h_arrayC[i]=h_arrayA[i]+h_arrayB[i];
		dot+=(1./h_arrayC[i])*(1./h_arrayA[i]);
	}
	cutilCheckError(cutStopTimer(timer_cpu));
	printf("CPU processing time : %f (ms) \n", cutGetTimerValue(timer_cpu));
	cutDeleteTimer(timer_cpu);

	unsigned int timer_gpu = 0;
	cutilCheckError(cutCreateTimer(&timer_gpu));
  cutilCheckError(cutStartTimer(timer_gpu));
	stat = cublasSetVector(size,sizeof(double),h_arrayA,1,d_arrayA,1);
	stat = cublasSetVector(size,sizeof(double),h_arrayB,1,d_arrayB,1);
	int nbBlocs=(size+nbThreadsPerBloc-1)/nbThreadsPerBloc;

	addition<<<nbBlocs,nbThreadsPerBloc>>>(size,d_arrayC,d_arrayA,d_arrayB);
	inverse<<<nbBlocs,nbThreadsPerBloc>>>(size,d_arrayC);
	inverse<<<nbBlocs,nbThreadsPerBloc>>>(size,d_arrayA);
	double dot_gpu=0;
	stat = cublasDdot(handle,size,d_arrayC,1,d_arrayA,1,&dot_gpu);

	cutilCheckError(cutStopTimer(timer_gpu));
	printf("GPU processing time : %f (ms) \n", cutGetTimerValue(timer_gpu));
	cutDeleteTimer(timer_gpu);
	printf("cpu dot %e --- gpu dot %e\n",dot,dot_gpu);

	cudaFree(d_arrayA);
	cudaFree(d_arrayB);
	cudaFree(d_arrayC);
	free(h_arrayA);
	free(h_arrayB);
	free(h_arrayC);
	free(h_arrayCgpu);
	cublasDestroy(handle);
	return 0;
}
