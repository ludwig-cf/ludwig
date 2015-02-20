/*
 * targetDP_CUDA.c: API Implementation for targetDP: CUDA version
 * Alan Gray, November 2013
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#include "targetDP.h"

//pointers to internal work space
static double* dwork;
static double* dwork_d;
static int* iwork;
static int* iwork_d;


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

void targetInit(size_t nsites, size_t nfieldsmax){


  // allocate internal work space

  dwork = (double*) malloc (nsites*nfieldsmax*sizeof(double));
  
  cudaMalloc(&dwork_d,nsites*nfieldsmax*sizeof(double));
  checkTargetError("malloc dwork_d");


  iwork = (int*) malloc (nsites*sizeof(int));
  
  cudaMalloc(&iwork_d,nsites*sizeof(int));
  checkTargetError("malloc iwork_d");



  return;
}

void targetFinalize(){

  free(iwork);
  free(dwork);
  cudaFree(iwork_d);
  cudaFree(dwork_d);
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


__global__ static void copy_field_partial_gpu_d(double* f_out, const double* f_in, int nsites, int nfields, int *fullindex_d, int packedsize, int inpack) {

  int threadIndex;
  int i;


    threadIndex = blockIdx.x*blockDim.x+threadIdx.x;


  if ((threadIndex < packedsize))
    {


      for (i=0;i<nfields;i++)
	{
	    
	  if (inpack)
	    f_out[i*nsites+fullindex_d[threadIndex]]
	    =f_in[i*packedsize+threadIndex];
	  else
	   f_out[i*packedsize+threadIndex]
	      =f_in[i*nsites+fullindex_d[threadIndex]];
	  
	}
    }


  return;
}

__global__ static void copy_field_partial_gpu_AoS_d(double* f_out, const double* f_in, int nsites, int nfields, int *fullindex_d, int packedsize, int inpack) {

  int threadIndex;
  int i;


    threadIndex = blockIdx.x*blockDim.x+threadIdx.x;


  if ((threadIndex < packedsize))
    {

      for (i=0;i<nfields;i++)
	{
	    
	  /* if (inpack) */
	  /*   f_out[i*nsites+fullindex_d[threadIndex]] */
	  /*   =f_in[i*packedsize+threadIndex]; */
	  /* else */
	  /*  f_out[i*packedsize+threadIndex] */
	  /*     =f_in[i*nsites+fullindex_d[threadIndex]]; */


	  if (inpack)
	    f_out[fullindex_d[threadIndex]*nfields+i]
	    =f_in[threadIndex*nfields+i];
	  else
	   f_out[threadIndex*nfields+i]
	     =f_in[fullindex_d[threadIndex]*nfields+i];
	  
	}
    }


  return;
}



void copyToTargetMasked(double *targetData,const double* data,size_t nsites,
			size_t nfields,char* siteMask){


  int i;
  int index;

  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;

  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMask[i]){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;


  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);


    
  //compress grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(siteMask[index]){
	  tmpGrid[i*packedsize+j]=data[i*nsites+index];
	  j++;
	}
      }
      
    }


  //put compressed grid on GPU
  cudaMemcpy(tmpGrid_d, tmpGrid, packedsize*nfields*sizeof(double), cudaMemcpyHostToDevice); 



  //uncompress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(targetData,tmpGrid_d,nsites,
						    nfields,
						    fullindex_d, packedsize, 1);
  cudaThreadSynchronize();

  
  return;
  
}


void copyFromTargetMasked(double *data,const double* targetData,size_t nsites,
			size_t nfields,char* siteMask){




  int i;
  int index;


  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;


  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMask[i]){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;

  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);

  
  //compress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(tmpGrid_d,targetData,nsites,
						    nfields,
						    fullindex_d, packedsize, 0);
  cudaThreadSynchronize();

  //get compressed grid from GPU
  cudaMemcpy(tmpGrid, tmpGrid_d, packedsize*nfields*sizeof(double), cudaMemcpyDeviceToHost); 

    
  //expand into final grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(siteMask[index]){
	  data[i*nsites+index]=tmpGrid[i*packedsize+j];	
	  j++;
	}
      }
      
    }

  return;

}



int haloEdge(int index, int extents[3],int offset, int depth){

  int coords[3];


  targetCoords3D(coords,extents,index);

  int returncode=0;

  int i;



    for (i=0;i<3;i++){
      if ( 
	  (coords[i]>=(offset)) && 
	  (coords[i]<(offset+depth))  
	  
	   ) returncode=1;
      
      if ( 
	  (coords[i] >= (extents[i]-offset-depth) ) && 
	  (coords[i] < (extents[i]-offset) )   
	   ) returncode=1;


  }

  
  
    return returncode;


}



void copyFromTargetBoundary3D(double *data,const double* targetData,int extents[3], size_t nfields, int offset,int depth){


  size_t nsites=extents[0]*extents[1]*extents[2];


  int i;
  int index;


  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;


  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(haloEdge(i,extents,offset,depth)){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;

  
  

  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);

  
  //compress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(tmpGrid_d,targetData,nsites,
						    nfields,
						    fullindex_d, packedsize, 0);
  cudaThreadSynchronize();

  //get compressed grid from GPU
  cudaMemcpy(tmpGrid, tmpGrid_d, packedsize*nfields*sizeof(double), cudaMemcpyDeviceToHost); 

    
  //expand into final grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(haloEdge(index,extents,offset,depth)){
	  data[i*nsites+index]=tmpGrid[i*packedsize+j];	
	  j++;
	}
      }
      
    }

  checkTargetError("copyFromTargetHaloEdge");
  return;

}

void copyToTargetBoundary3D(double *targetData,const double* data, int extents[3], size_t nfields, int offset,int depth){

  size_t nsites=extents[0]*extents[1]*extents[2];

  int i;
  int index;

  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;


  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(haloEdge(i,extents,offset,depth)){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;


  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);


    
  //compress grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(haloEdge(index,extents,offset,depth)){
	  tmpGrid[i*packedsize+j]=data[i*nsites+index];
	  j++;
	}
      }
      
    }


  //put compressed grid on GPU
  cudaMemcpy(tmpGrid_d, tmpGrid, packedsize*nfields*sizeof(double), cudaMemcpyHostToDevice); 



  //uncompress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(targetData,tmpGrid_d,nsites,
						    nfields,
						    fullindex_d, packedsize, 1);
  cudaThreadSynchronize();

  checkTargetError("copyToTargetHaloEdge");
  
  return;
  
}





void copyToTargetMaskedAoS(double *targetData,const double* data,size_t nsites,
			size_t nfields,char* siteMask){


  int i;
  int index;

  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;

  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMask[i]){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;


  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);


    
  //compress grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(siteMask[index]){
	  //tmpGrid[i*packedsize+j]=data[i*nsites+index];
	  tmpGrid[j*nfields+i]=data[index*nfields+i];
	  j++;
	}
      }
      
    }


  //put compressed grid on GPU
  cudaMemcpy(tmpGrid_d, tmpGrid, packedsize*nfields*sizeof(double), cudaMemcpyHostToDevice); 



  //uncompress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_AoS_d<<<nblocks,DEFAULT_TPB>>>(targetData,tmpGrid_d,nsites,
						    nfields,
						    fullindex_d, packedsize, 1);
  cudaThreadSynchronize();

  
  return;
  
}

void copyFromTargetMaskedAoS(double *data,const double* targetData,size_t nsites,
			size_t nfields,char* siteMask){




  int i;
  int index;


  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;


  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMask[i]){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;

  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);

  
  //compress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_AoS_d<<<nblocks,DEFAULT_TPB>>>(tmpGrid_d,targetData,nsites,
						    nfields,
						    fullindex_d, packedsize, 0);
  cudaThreadSynchronize();

  //get compressed grid from GPU
  cudaMemcpy(tmpGrid, tmpGrid_d, packedsize*nfields*sizeof(double), cudaMemcpyDeviceToHost); 

    
  //expand into final grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(siteMask[index]){
	  //data[i*nsites+index]=tmpGrid[i*packedsize+j];	
	  data[index*nfields+i]=tmpGrid[j*nfields+i];	
	  j++;
	}
      }
      
    }


  return;

}



void targetSynchronize(){
  cudaThreadSynchronize();
  checkTargetError("syncTarget");
  return;
}

void targetFree(void *ptr){
  
  cudaFree(ptr);
  checkTargetError("targetFree");
  return;
  
}

