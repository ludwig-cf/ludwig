/*
 * targetDP_C.c: API Implementation for targetDP: C version
 * Alan Gray, November 2013
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "targetDP.h"


void checkTargetError(const char *msg){

  return;

}

void targetInit(int extents[3], size_t nfieldsmax, int nhalo){
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


void targetMallocUnified(void **address_of_ptr,size_t size){

  void* ptr;
  ptr = malloc(size);

  if(!ptr){
    printf("malloc failed\n");
    exit(1);
  }
    

  *address_of_ptr=ptr;



  return;
  
}


void targetCallocUnified(void **address_of_ptr,size_t size){

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


void targetSynchronize(){
  return;
}


void targetZero(double* array,size_t size){

  int i;

  for(i=0;i<size;i++){
    
    array[i]=0.;
    
  }

}
