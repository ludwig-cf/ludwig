/*****************************************************************************
 *
 *  cuda_stub_api.h
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_stub_api.h"

static cudaError_t lastError;

#define error_return_if(expr, error) \
  do { if ((expr)) { 		     \
      lastError = error;	     \
      return error;		     \
    }				     \
  } while(0)


cudaError_t cudaDeviceSynchronize(void) {

  /* do nothing */

  return cudaSuccess;
}

cudaError_t cudaFree(void ** devPtr) {

  error_return_if(devPtr == NULL, cudaErrorInvalidDevicePointer);

  free(*devPtr);

  return cudaSuccess;
}


const char *  cudaGetErrorString(cudaError_t error) {

  /* Need some strings */

  return "Oh dear";
}

cudaError_t cudaGetLastError(void) {

  return lastError;
}

cudaError_t cudaMalloc(void ** devPtr, size_t size) {

  assert(devPtr);

  *devPtr = malloc(size);

  error_return_if(*devPtr == NULL, cudaErrorMemoryAllocation);

  return cudaSuccess;
}


cudaError_t cudaMemcpy(void * dst, const void * src, size_t count,
		       cudaMemcpyKind kind) {

  assert(dst);
  assert(src);

  error_return_if(count < 1, cudaErrorInvalidValue);

  switch (kind) {
  case cudaMemcpyHostToDevice:
    error_return_if(dst == NULL, cudaErrorInvalidDevicePointer);
    memcpy(dst, src, count);
    break;
  case cudaMemcpyDeviceToHost:
    error_return_if(src == NULL, cudaErrorInvalidDevicePointer);
    memcpy(dst, src, count);
    break;
  case cudaMemcpyHostToHost:
    memcpy(dst, src, count);
    break;
  case cudaMemcpyDeviceToDevice:
    memcpy(dst, src, count);
    break;
  case cudaMemcpyDefault:
  default:
    error_return_if(1, cudaErrorInvalidMemcpyDirection);
  }

  return cudaSuccess;
}


cudaError_t cudaMemset(void * devPtr, int value, size_t count) {

  error_return_if(devPtr == NULL, cudaErrorInvalidDevicePointer);
  error_return_if(value < 0, cudaErrorInvalidValue);
  error_return_if(value > 255, cudaErrorInvalidValue);

  memset(devPtr, value, count);

  return cudaSuccess;
}
