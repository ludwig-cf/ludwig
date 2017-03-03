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

dim3 threadIdx;
dim3 blockIdx;
dim3 gridDim = {1, 1, 1};
dim3 blockDim = {1, 1, 1};

static tdpError_t lastError;
static int staticStream;

#define error_return_if(expr, error) \
  do { if ((expr)) { 		     \
      lastError = error;	     \
      return error;		     \
    }				     \
  } while(0)

#define error_return(error) \
  error_return_if(1, error)



tdpError_t tdpDeviceSynchronize(void) {

  /* do nothing */

  return tdpSuccess;
}

tdpError_t tdpFree(void * devPtr) {

  error_return_if(devPtr == NULL, tdpErrorInvalidDevicePointer);

  free(devPtr);

  return tdpSuccess;
}

tdpError_t tdpFreeHost(void * ptr) {

  free(ptr);

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  Return id of device currently being used.
 *
 *****************************************************************************/

tdpError_t tdpGetDevice(int * device) {

  assert(device);
  assert(0);       /* Should not be here if no device */

  *device = -1;

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  Return number of available devices
 *
 *****************************************************************************/

tdpError_t tdpGetDeviceCount(int * device) {

  *device = 0;

#ifdef FAKE_DEVICE /* "Fake" device */
  *device = 1;
#endif

  /* Strictly, we should return tdpErrorInsufficientDriver or ... */

  return tdpErrorNoDevice;
}

const char *  tdpGetErrorString(tdpError_t error) {

  /* Need some strings */

  return "Oh dear";
}

tdpError_t tdpGetLastError(void) {

  return lastError;
}

tdpError_t tdpGetSymbolAddress(void ** devPtr, const void * symbol) {

  /* No device symbols available... */
  error_return(tdpErrorInvalidSymbol);

  return tdpSuccess;
}

tdpError_t tdpHostAlloc(void ** phost, size_t size, unsigned int flags) {

  void * ptr = NULL;

  error_return_if(phost == NULL, tdpErrorInvalidValue);

  switch (flags) {
  case tdpHostAllocDefault:
  case tdpHostAllocPortable:
  case tdpHostAllocMapped:
  case tdpHostAllocWriteCombined:

    ptr = malloc(size);
    error_return_if(ptr == NULL, tdpErrorMemoryAllocation);

    *phost = ptr;
    break;

  default:
    error_return(tdpErrorInvalidValue);
  }

  return tdpSuccess;
}

tdpError_t tdpMalloc(void ** devPtr, size_t size) {

  assert(devPtr);

  *devPtr = malloc(size);

  error_return_if(*devPtr == NULL, tdpErrorMemoryAllocation);

  return tdpSuccess;
}


tdpError_t tdpMemcpy(void * dst, const void * src, size_t count,
		       tdpMemcpyKind kind) {

  assert(dst);
  assert(src);

  error_return_if(count < 1, tdpErrorInvalidValue);

  switch (kind) {
  case tdpMemcpyHostToDevice:
    error_return_if(dst == NULL, tdpErrorInvalidDevicePointer);
    memcpy(dst, src, count);
    break;
  case tdpMemcpyDeviceToHost:
    error_return_if(src == NULL, tdpErrorInvalidDevicePointer);
    memcpy(dst, src, count);
    break;
  case tdpMemcpyHostToHost:
    memcpy(dst, src, count);
    break;
  case tdpMemcpyDeviceToDevice:
    memcpy(dst, src, count);
    break;
  case tdpMemcpyDefault:
  default:
    error_return(tdpErrorInvalidMemcpyDirection);
  }

  return tdpSuccess;
}

tdpError_t tdpMemcpyFromSymbol(void * dst, const void * symbol,
				 size_t count, size_t offset,
				 tdpMemcpyKind kind) {
  assert(dst);
  assert(symbol);

  error_return_if(count < 1, tdpErrorInvalidValue);
  error_return_if(offset != 0, tdpErrorInvalidValue);

  switch (kind) {
  case tdpMemcpyDefault:
  case tdpMemcpyDeviceToHost:
    error_return_if(symbol == NULL, tdpErrorInvalidSymbol);
    memcpy(dst, symbol, count);
    break;
  case tdpMemcpyDeviceToDevice:
    error_return_if(dst == NULL, tdpErrorInvalidDevicePointer);
    error_return_if(symbol == NULL, tdpErrorInvalidSymbol);
    memcpy(dst, symbol, count);
    break;
  case tdpMemcpyHostToDevice:
  case tdpMemcpyHostToHost:
  default:
    error_return(tdpErrorInvalidMemcpyDirection);
  }

  return tdpSuccess;
}

/* Tdp wants "const void * symbol", but this is avoided as we need
 * a memset(void * dst, const void * src, ...) . */

tdpError_t tdpMemcpyToSymbol(void * symbol, const void * src,
			       size_t count, size_t offset,
			       tdpMemcpyKind kind) {
  assert(symbol);
  assert(src);

  error_return_if(count < 1, tdpErrorInvalidValue);
  error_return_if(offset != 0, tdpErrorInvalidValue);

  switch (kind) {
  case tdpMemcpyDefault:
  case tdpMemcpyHostToDevice:
    error_return_if(symbol == NULL, tdpErrorInvalidSymbol);
    memcpy(symbol, src, count);
    break;
  case tdpMemcpyDeviceToDevice:
    error_return_if(src == NULL, tdpErrorInvalidDevicePointer);
    memcpy(symbol, src, count);
    break;
  case tdpMemcpyDeviceToHost:
  case tdpMemcpyHostToHost:
  default:
    error_return(tdpErrorInvalidMemcpyDirection);
  }

  return tdpSuccess;
}

tdpError_t tdpMemset(void * devPtr, int value, size_t count) {

  error_return_if(devPtr == NULL, tdpErrorInvalidDevicePointer);
  error_return_if(value < 0, tdpErrorInvalidValue);
  error_return_if(value > 255, tdpErrorInvalidValue);

  memset(devPtr, value, count);

  return tdpSuccess;
}


tdpError_t tdpStreamCreate(tdpStream_t * stream) {

  error_return_if(stream == NULL, tdpErrorInvalidValue);

  *stream = &staticStream;

  return tdpSuccess;
}

tdpError_t tdpStreamDestroy(tdpStream_t stream) {

  error_return_if(stream != &staticStream, tdpErrorInvalidResourceHandle);

  return tdpSuccess;
}

tdpError_t tdpStreamSynchronize(tdpStream_t stream) {

  error_return_if(stream != &staticStream, tdpErrorInvalidResourceHandle);

  /* Success */

  return tdpSuccess;
}

/* No optional arguments */

tdpError_t tdpMemcpyAsync(void * dst, const void * src, size_t count,
			  tdpMemcpyKind kind, tdpStream_t stream) {

  /* Just ignore the stream argument and copy immediately */

  return tdpMemcpy(dst, src, count, kind);
}
