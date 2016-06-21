/*****************************************************************************
 *
 *  cuda_stub_api.h
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_stub_api.h"

/* Globally reserved names. */

dim3 gridDim = {1, 1, 1};
dim3 blockDim = {1, 1, 1};

static cudaError_t lastError;
static int staticStream;

#define error_return_if(expr, error) \
  do { if ((expr)) { 		     \
      lastError = error;	     \
      return error;		     \
    }				     \
  } while(0)

#define error_return(error) \
  error_return_if(1, error)


cudaError_t cudaDeviceSynchronize(void) {

  /* do nothing */

  return cudaSuccess;
}

cudaError_t cudaFree(void ** devPtr) {

  error_return_if(devPtr == NULL, cudaErrorInvalidDevicePointer);

  free(*devPtr);

  return cudaSuccess;
}

cudaError_t cudaFreeHost(void * ptr) {

  free(ptr);

  return cudaSuccess;
}

/*****************************************************************************
 *
 *  Return id of device currently being used.
 *
 *****************************************************************************/

cudaError_t cudaGetDevice(int * device) {

  assert(device);
  assert(0);       /* Should not be here if no device */

  *device = -1;

  return cudaSuccess;
}

/*****************************************************************************
 *
 *  Return number of available devices
 *
 *****************************************************************************/

cudaError_t cudaGetDeviceCount(int * count) {

  assert(count);

  *count = 0;

  return cudaErrorInsufficientDriver;
}

const char *  cudaGetErrorString(cudaError_t error) {

  /* Need some strings */

  return "Oh dear";
}

cudaError_t cudaGetLastError(void) {

  return lastError;
}

cudaError_t cudaHostAlloc(void ** phost, size_t size, unsigned int flags) {

  void * ptr = NULL;

  error_return_if(phost == NULL, cudaErrorInvalidValue);

  switch (flags) {
  case cudaHostAllocDefault:
  case cudaHostAllocPortable:
  case cudaHostAllocMapped:
  case cudaHostAllocWriteCombined:

    ptr = malloc(size);
    error_return_if(ptr == NULL, cudaErrorMemoryAllocation);

    *phost = ptr;
    break;

  default:
    error_return(cudaErrorInvalidValue);
  }

  return cudaSuccess;
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
    error_return(cudaErrorInvalidMemcpyDirection);
  }

  return cudaSuccess;
}

cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol,
				 size_t count, size_t offset,
				 cudaMemcpyKind kind) {
  assert(dst);
  assert(symbol);

  error_return_if(count < 1, cudaErrorInvalidValue);
  error_return_if(offset != 0, cudaErrorInvalidValue);

  switch (kind) {
  case cudaMemcpyDefault:
  case cudaMemcpyDeviceToHost:
    error_return_if(symbol == NULL, cudaErrorInvalidSymbol);
    memcpy(dst, symbol, count);
    break;
  case cudaMemcpyDeviceToDevice:
    error_return_if(dst == NULL, cudaErrorInvalidDevicePointer);
    error_return_if(symbol == NULL, cudaErrorInvalidSymbol);
    memcpy(dst, symbol, count);
    break;
  case cudaMemcpyHostToDevice:
  case cudaMemcpyHostToHost:
  default:
    error_return(cudaErrorInvalidMemcpyDirection);
  }

  return cudaSuccess;
}

/* Cuda wants "const void * symbol", but this is avoided as we need
 * a memset(void * dst, const void * src, ...) . */

cudaError_t cudaMemcpyToSymbol(void * symbol, const void * src,
			       size_t count, size_t offset,
			       cudaMemcpyKind kind) {
  assert(symbol);
  assert(src);

  error_return_if(count < 1, cudaErrorInvalidValue);
  error_return_if(offset != 0, cudaErrorInvalidValue);

  switch (kind) {
  case cudaMemcpyDefault:
  case cudaMemcpyHostToDevice:
    error_return_if(symbol == NULL, cudaErrorInvalidSymbol);
    memcpy(symbol, src, count);
    break;
  case cudaMemcpyDeviceToDevice:
    error_return_if(src == NULL, cudaErrorInvalidDevicePointer);
    memcpy(symbol, src, count);
    break;
  case cudaMemcpyDeviceToHost:
  case cudaMemcpyHostToHost:
  default:
    error_return(cudaErrorInvalidMemcpyDirection);
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


cudaError_t cudaStreamCreate(cudaStream_t * stream) {

  error_return_if(stream == NULL, cudaErrorInvalidValue);

  *stream = &staticStream;

  return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {

  error_return_if(stream != &staticStream, cudaErrorInvalidResourceHandle);

  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {

  error_return_if(stream != &staticStream, cudaErrorInvalidResourceHandle);

  /* Success */

  return cudaSuccess;
}

/* No optional arguments */

cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count,
			    cudaMemcpyKind kind, cudaStream_t stream) {

  /* Just ignore the stream argument and copy immediately */

  return cudaMemcpy(dst, src, count, kind);
}
