#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "cutil_inline.h"
#include <cublas_v2.h>

const int width=16;
const int nbTh=width*width;

const int size=1024;
const 	int sizeMat=size*size;

__global__ 
void matmul(float *d_A, float *d_B, float *d_C) {
	int i= blockIdx.y*blockDim.y+ threadIdx.y;
	int j= blockIdx.x*blockDim.x+ threadIdx.x;

	float sum=0;
	for(int k=0;k<size;k++) {
		sum+=d_A[i*size+k]*d_B[k*size+j];
	}	
	d_C[i*size+j]=sum;
}

int main( int argc, char** argv) 
{
	float *h_arrayA=(float*)malloc(sizeMat*sizeof(float));
	float *h_arrayB=(float*)malloc(sizeMat*sizeof(float));
	float *h_arrayC=(float*)malloc(sizeMat*sizeof(float));
	float *h_arrayCgpu=(float*)malloc(sizeMat*sizeof(float));

	float *d_arrayA, *d_arrayB, *d_arrayC;

	cudaMalloc((void**)&d_arrayA,sizeMat*sizeof(float));
	cudaMalloc((void**)&d_arrayB,sizeMat*sizeof(float));
	cudaMalloc((void**)&d_arrayC,sizeMat*sizeof(float));

	srand48(32);
	for(int i=0;i<sizeMat;i++) {
		h_arrayA[i]=drand48();
		h_arrayB[i]=drand48();
		h_arrayC[i]=0;
		h_arrayCgpu[i]=0;

	}

	cudaMemcpy(d_arrayA,h_arrayA, sizeMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arrayB,h_arrayB, sizeMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arrayC,h_arrayC, sizeMat * sizeof(float), cudaMemcpyHostToDevice);

	unsigned int timer_cpu = 0;
	cutilCheckError(cutCreateTimer(&timer_cpu));
  cutilCheckError(cutStartTimer(timer_cpu));
	int sum=0;
	for(int i=0;i<size;i++) {
		for(int j=0;j<size;j++) {
			for(int k=0;k<size;k++) {
				h_arrayC[size*i+j]+=h_arrayA[size*i+k]*h_arrayB[size*k+j];
			}	
		}	
	}
	cutilCheckError(cutStopTimer(timer_cpu));
	printf("CPU processing time : %f (ms) \n", cutGetTimerValue(timer_cpu));
	cutDeleteTimer(timer_cpu);

	unsigned int timer_gpu = 0;
	cutilCheckError(cutCreateTimer(&timer_gpu));
  cutilCheckError(cutStartTimer(timer_gpu));

	dim3 dimGrid(size/width,size/width);
	dim3 dimBlock(width,width);

	matmul<<<dimGrid,dimBlock>>>(d_arrayA,d_arrayB,d_arrayC);
	cudaThreadSynchronize();
	
	cutilCheckError(cutStopTimer(timer_gpu));
	printf("GPU processing time : %f (ms) \n", cutGetTimerValue(timer_gpu));
	cutDeleteTimer(timer_gpu);
	
	cudaMemcpy(h_arrayCgpu,d_arrayC, sizeMat * sizeof(float), cudaMemcpyDeviceToHost);
	
	for(int i=0;i<sizeMat;i++)
		if (fabs(h_arrayC[i]-h_arrayCgpu[i])>1e-4)
			printf("%f %f\n",h_arrayC[i],h_arrayCgpu[i]);
	
	cudaFree(d_arrayA);
	cudaFree(d_arrayB);
	cudaFree(d_arrayC);
	free(h_arrayA);
	free(h_arrayB);
	free(h_arrayC);
	free(h_arrayCgpu);
	return 0;
}
