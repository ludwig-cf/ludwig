/*
 * targetDP_X86.c: API Implementation for targetDP: X86 version
 * Alan Gray, November 2013
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "targetDP.h"

/* Globally reserved names */

dim3 gridDim;
dim3 blockDim;

/* Utilities */


uint3 __x86_builtin_threadIdx_init(void) {
  uint3 threads = {1, 1, 1};
  threads.x = __x86_get_thread_num();
  return threads;
}

uint3 __x86_builtin_blockIdx_init(void) {
  uint3 blocks = {1, 1, 1};
  return blocks;
}

void __x86_prelaunch(dim3 nblocks, dim3 nthreads) {

  gridDim = nblocks;
  blockDim = nthreads;

  /* sanity checks on user settings here... */

  /* In case we request fewer threads than are available: */

  __x86_set_num_threads(blockDim.x*blockDim.y*blockDim.z);

  return;
}

void __x86_postlaunch(void) {

  int nthreads;

  /* Reset the default number of threads. */
  nthreads = __x86_get_max_threads();
  __x86_set_num_threads(nthreads);

  return;
}


void checkTargetError(const char *msg){

  return;

}

void targetInit(size_t nsites,size_t nfieldsmax){
  return;
}

void targetFinalize(){
  return;
}



void targetMalloc(void **address_of_ptr,size_t size){

  void* ptr;
  ptr = malloc(size);

  if(!ptr){
    printf("malloc failed\n");
    exit(1);
  }
    

  *address_of_ptr=ptr;



  return;
  
}

void targetCalloc(void **address_of_ptr,size_t size){

  void* ptr;
  ptr = calloc(1,size);

  if(!ptr){
    printf("calloc failed\n");
    exit(1);
  }
    

  *address_of_ptr=ptr;



  return;
  
}



void targetFree(void *ptr){
  
  free(ptr);
  return;
  
}


void copyToTarget(void *targetData,const void* data,const size_t size){

  memcpy(targetData,data,size);
  return;

}

void copyFromTarget(void *data,const void* targetData,const size_t size){

  memcpy(data,targetData,size);
  return;

}

void copyToTargetMasked(double *targetData,const double* data,const size_t nsites,
			const size_t nfields,char* siteMask){

  int i,j;
  for (i=0;i<nfields;i++){
    for (j=0;j<nsites;j++){
      if(siteMask[j]) targetData[i*nsites+j]=data[i*nsites+j];
    }
  }
  return;
  
}


void copyToTargetMaskedAoS(double *targetData,const double* data,const size_t nsites,
			const size_t nfields,char* siteMask){

  int i,j;
  for (j=0;j<nsites;j++){
    if(siteMask[j]){
      for (i=0;i<nfields;i++){
	targetData[j*nfields+i]=data[j*nfields+i];
      }
    }
  }

  return;
  
}

void copyFromTargetMasked(double *data,const double* targetData,const size_t nsites,
			const size_t nfields,char* siteMask){

  int i, j;
  for (i=0;i<nfields;i++){
    for (j=0;j<nsites;j++){
       if(siteMask[j]) data[i*nsites+j]=targetData[i*nsites+j];
    }
  }
  return;

}


void copyFromTargetMaskedAoS(double *data,const double* targetData,const size_t nsites,
			const size_t nfields,char* siteMask){

  int i, j;
  for (j=0;j<nsites;j++){
    if(siteMask[j]){
      for (i=0;i<nfields;i++){
	data[j*nfields+i]=targetData[j*nfields+i];
      }
    }
  }
  return;
    
}


void syncTarget(){
  return;
}

void copyConstantIntToTarget(int *data_d, const int *data, const int size){
  memcpy(data_d, data, size); 
  return;
} 
void copyConstantInt1DArrayToTarget(int *data_d, const int *data, const int size){
  memcpy(data_d, data, size); 
  return;
} 
void copyConstantInt2DArrayToTarget(int **data_d, const int *data, const int size){
  memcpy(data_d, data, size); 
  return;
} 
void copyConstantDoubleToTarget(double *data_d, const double *data, const int size){
  memcpy(data_d, data, size); 
  return;
} 
void copyConstantDouble1DArrayToTarget(double *data_d, const double *data, const int size){
  memcpy(data_d, data, size); 
  return;
} 
void copyConstantDouble2DArrayToTarget(double **data_d, const double *data, const int size){
  memcpy(data_d, data, size); 
  return;
} 
void copyConstantDouble3DArrayToTarget(double ***data_d, const double *data, const int size){
  memcpy(data_d, data, size); 
  return;
}
void  copyConstantMufnFromTarget(mu_fntype* data, mu_fntype* data_d, const int size ){
  memcpy(data, data_d, size);
  return;
}
void  copyConstantPthfnFromTarget(pth_fntype* data, pth_fntype* data_d, const int size ){
  memcpy(data, data_d, size);
  return;
}
void copyConstantDoubleFromTarget(double *data, const double *data_d, const int size){
  memcpy(data, data_d, size); 
  return;
} 

