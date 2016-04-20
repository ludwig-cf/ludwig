/* Cuda stub interface. API functions are added as required. */

#ifndef CUDA_STUB_API_H
#define CUDA_STUB_API_H

#include <stdlib.h>

typedef enum cudaMemcpyKind_enum {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4}
  cudaMemcpyKind;


/* cudaGetLastError() can return... */

enum cudaError {
  cudaSuccess = 0,
  cudaErrorMissingConfiguration = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorLaunchFailure = 4,
  cudaErrorLaunchTimeout = 6,
  cudaErrorLaunchOutOfResources = 7,
  cudaErrorInvalidDeviceFunction = 8,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidValue = 11,
  cudaErrorInvalidPitchValue = 12,
  cudaErrorInvalidSymbol = 13,
  cudaErrorUnmapBufferObjectFailed = 15,
  cudaErrorInvalidHostPointer = 16,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidTexture = 18,
  cudaErrorInvalidTextureBinding = 19,
  cudaErrorInvalidChannelDescriptor = 20,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorInvalidFilterSetting = 26,
  cudaErrorUnknown = 30,
  cudaErrorInvalidResourceHandle = 33,
  cudaErrorInsufficientDriver = 35,
  cudaErrorSetOnActiveProcess = 36,
  cudaErrorStartupFailure = 0x7f
};

#define cudaHostAllocDefault       0x00
#define cudaHostAllocMapped        0x02
#define cudaHostAllocPortable      0x01
#define cudaHostAllocWriteCombined 0x04

/* Device memory qualifiers / executation space qualifiers */

#define __host__
#define __global__
#define __shared__
#define __device__
#define __constant__

#if (__STDC__VERSION__ >= 19901)
  #define __forceinline__ inline
  #define __noinline__
#else
  #define __forceinline__
  #define __noinline__
#endif

/* Built-in variable implementation. */

typedef struct __uint3_s uint3;
typedef struct __dim3_s dim3;

struct __uint3_s {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct __dim3_s {
  int x;
  int y;
  int z;
};

/* Smuggle in gridDim and blockDim here; names must be reserved. */

extern dim3 gridDim;
extern dim3 blockDim;

typedef enum cudaError cudaError_t;     /* an enum type */
typedef int * cudaStream_t;             /* an opaque handle */

/* API */

__host__ __device__ cudaError_t cudaDeviceSynchronize(void);
__host__ __device__ cudaError_t cudaFree(void ** devPtr);
__host__            cudaError_t cudaFreeHost(void * phost);
__host__ __device__ cudaError_t cudaGetDevice(int * device);
__host__ __device__ cudaError_t cudaGetDeviceCount(int * count);
__host__ __device__ const char* cudaGetErrorString(cudaError_t error);
__host__ __device__ cudaError_t cudaGetLastError(void);

__host__ __device__ cudaError_t cudaHostAlloc(void ** phost, size_t size,
					      unsigned int flags);
__host__ __device__ cudaError_t cudaMalloc(void ** devRtr, size_t size);
__host__            cudaError_t cudaMemcpy(void * dst, const void * src,
					   size_t count,
					   cudaMemcpyKind kind);
__host__            cudaError_t cudaMemcpyFromSymbol(void * dst,
						     const void * symbol,
						     size_t count,
						     size_t offset,
						     cudaMemcpyKind kind);
__host__            cudaError_t cudaMemcpyToSymbol(void * symbol,
						   const void * src,
						   size_t count, size_t offset,
						   cudaMemcpyKind kind);
__host__            cudaError_t cudaMemset(void * devPtr, int value,
					   size_t count);


#endif
