/*
 * targetDP_CUDA.c: API Implementation for targetDP: CUDA version
 * Alan Gray, November 2013
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

void checkTargetError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
				cudaGetErrorString( err) );
		fflush(stdout);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}                         
}

void targetMalloc(void **address_of_ptr,size_t size){

 
  cudaMalloc(address_of_ptr,size);
  checkTargetError("targetMalloc");

  return;
}

void targetCalloc(void **address_of_ptr,size_t size){

 
  cudaMalloc(address_of_ptr,size);
  double ZERO=0.;
  cudaMemset(*address_of_ptr, ZERO, size);
  checkTargetError("targetCalloc");

  return;
}

void copyToTarget(void *targetData,const void* data,size_t size){

  cudaMemcpy(targetData,data,size,cudaMemcpyHostToDevice);
  checkTargetError("copyToTarget");
  return;
}

void copyFromTarget(void *data,const void* targetData,size_t size){

  cudaMemcpy(data,targetData,size,cudaMemcpyDeviceToHost);
  checkTargetError("copyFromTarget");
  return;

}

void syncTarget(){
  cudaThreadSynchronize();
  checkTargetError("syncTarget");
  return;
}

void targetFree(void *ptr){
  
  cudaFree(ptr);
  checkTargetError("targetFree");
  return;
  
}
void copyConstantIntToTarget(int *data_d, int *data, int size){
  cudaMemcpyToSymbol(*data_d, data, size, 0,cudaMemcpyHostToDevice);
  checkTargetError("copyConstantIntToTarget");
  return;
} 
void copyConstantInt1DArrayToTarget(int *data_d, int *data, int size){
  cudaMemcpyToSymbol(*data_d, data, size, 0,cudaMemcpyHostToDevice);
  checkTargetError("copyConstantInt1DArrayToTarget");
  return;
} 
void copyConstantInt2DArrayToTarget(int **data_d, int *data, int size){
  cudaMemcpyToSymbol(*data_d, data, size, 0,cudaMemcpyHostToDevice);
  checkTargetError("copyConstantInt2DArrayToTarget");
  return;
} 
void copyConstantDoubleToTarget(double *data_d, double *data, int size){
  cudaMemcpyToSymbol(*data_d, data, size, 0,cudaMemcpyHostToDevice);
  checkTargetError("copyConstantDoubleToTarget");
  return;
} 
void copyConstantDouble1DArrayToTarget(double *data_d, double *data, int size){
  cudaMemcpyToSymbol(*data_d, data, size, 0,cudaMemcpyHostToDevice);
  checkTargetError("copyConstantDouble1DArrayToTarget");
  return;
} 

void copyConstantDouble2DArrayToTarget(double **data_d, double *data, int size){
  cudaMemcpyToSymbol(*data_d, data, size, 0,cudaMemcpyHostToDevice);
  checkTargetError("copyConstantDouble2DArrayToTarget");
  return;
} 
void copyConstantDouble3DArrayToTarget(double ***data_d, double *data, int size){
  cudaMemcpyToSymbol(*data_d, data, size, 0,cudaMemcpyHostToDevice);
  checkTargetError("copyConstantDouble3DArrayToTarget");
  return;
}

