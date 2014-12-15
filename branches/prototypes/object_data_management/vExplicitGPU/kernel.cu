#include <stdio.h>
#include "module.h"

void checkCUDAError(const char *msg);

__global__ void kernel (obj_kernel_data kernel_data, obj_const_data* const_ptr){


  printf("Setting first 2 components of d_array to constants:\n");
  kernel_data.field[0]=const_ptr->c1;
  kernel_data.field[1]=const_ptr->c2;

  printf("d_array[0]= %f\n",kernel_data.field[0]);
  printf("d_array[1]= %f\n",kernel_data.field[1]);

}


void kernel_wrapper(obj_kernel_data kernel_data, obj_const_data* const_ptr){

  printf("before kernel\n");

  kernel <<<1,1>>> (kernel_data, const_ptr);
  
  cudaThreadSynchronize();
  
  checkCUDAError("kernel");
  printf("after kernel\n");
}


/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
