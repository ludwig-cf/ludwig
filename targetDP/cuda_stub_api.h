/* Cuda stub interface. API functions are added as required. */

#ifndef CUDA_STUB_API_H
#define CUDA_STUB_API_H

#include <stdlib.h>

typedef enum tdpFuncCache_enum {
  tdpFuncCachePreferNone = 0,
  tdpFuncCachePreferShared = 1,
  tdpFuncCachePreferL1 = 2,
  tdpFuncCachePreferEqual = 3}
  tdpFuncCache;

typedef enum tdpMemcpyKind_enum {
  tdpMemcpyHostToHost = 0,
  tdpMemcpyHostToDevice = 1,
  tdpMemcpyDeviceToHost = 2,
  tdpMemcpyDeviceToDevice = 3,
  tdpMemcpyDefault = 4}
  tdpMemcpyKind;

/* Device attributes (potentially a lot of them) */

typedef enum tdpDeviceAttr_enum {
  tdpDevAttrMaxThreadsPerBlock = 1,
  tdpDevAttrMaxBlockDimX = 2,
  tdpDevAttrMaxBlockDimY = 3,
  tdpDevAttrMaxBlockDimZ = 4,
  tdpDevAttrMaxGridDimX = 5,
  tdpDevAttrMaxGridDimY = 6,
  tdpDevAttrMaxGridDimZ = 7
} tdpDeviceAttr;

/* tdpGetLastError() can return... */

enum tdpError {
  tdpSuccess = 0,
  tdpErrorMissingConfiguration = 1,
  tdpErrorMemoryAllocation = 2,
  tdpErrorInitializationError = 3,
  tdpErrorLaunchFailure = 4,
  tdpErrorLaunchTimeout = 6,
  tdpErrorLaunchOutOfResources = 7,
  tdpErrorInvalidDeviceFunction = 8,
  tdpErrorInvalidConfiguration = 9,
  tdpErrorInvalidDevice = 10,
  tdpErrorInvalidValue = 11,
  tdpErrorInvalidPitchValue = 12,
  tdpErrorInvalidSymbol = 13,
  tdpErrorUnmapBufferObjectFailed = 15,
  tdpErrorInvalidHostPointer = 16,
  tdpErrorInvalidDevicePointer = 17,
  tdpErrorInvalidTexture = 18,
  tdpErrorInvalidTextureBinding = 19,
  tdpErrorInvalidChannelDescriptor = 20,
  tdpErrorInvalidMemcpyDirection = 21,
  tdpErrorInvalidFilterSetting = 26,
  tdpErrorUnknown = 30,
  tdpErrorInvalidResourceHandle = 33,
  tdpErrorInsufficientDriver = 35,
  tdpErrorSetOnActiveProcess = 36,
  tdpErrorInvalidSurface = 37,
  tdpErrorNoDevice = 38,
  tdpErrorStartupFailure = 0x7f
};

#define tdpHostAllocDefault       0x00
#define tdpHostAllocMapped        0x02
#define tdpHostAllocPortable      0x01
#define tdpHostAllocWriteCombined 0x04

#define tdpMemAttachGlobal        0x01
#define tdpMemAttachHost          0x02
#define tdpMemAttachSingle        0x04

/* Device memory qualifiers / executation space qualifiers */

#define __host__
#define __global__
#define __shared__ static
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
/* Storage actually in cuda_stub_api.c */

extern dim3 gridDim;
extern dim3 blockDim;
extern dim3 threadIdx;
extern dim3 blockIdx;

#ifdef _OPENMP
  /* Globals must be ... */ 
  #pragma omp threadprivate(gridDim, blockDim, threadIdx, blockIdx)
#endif

typedef enum tdpError tdpError_t;     /* an enum type */
typedef int * tdpStream_t;            /* an opaque handle */

/* API */

/* Device management */

__host__ tdpError_t tdpDeviceSetCacheConfig(tdpFuncCache cacheConfig);

__host__ __device__ tdpError_t tdpDeviceGetAttribute(int * value,
						     tdpDeviceAttr attr,
						     int device);
__host__ __device__ tdpError_t tdpDeviceSynchronize(void);
__host__ __device__ tdpError_t tdpGetDevice(int * device);
__host__ __device__ tdpError_t tdpGetDeviceCount(int * count);

/* Error handling */

__host__ __device__ const char * tdpGetErrorName(tdpError_t error);
__host__ __device__ tdpError_t tdpGetLastError(void);

/* Stream management */

__host__ tdpError_t tdpStreamCreate(tdpStream_t * stream);
__host__ tdpError_t tdpStreamDestroy(tdpStream_t stream);
__host__ tdpError_t tdpStreamSynchronize(tdpStream_t stream);

/* Execution control */

/* See target_api.h for tdpLaunchKernel() */

/* Memory management */

__host__ tdpError_t tdpFreeHost(void * phost);
__host__ tdpError_t tdpGetSymbolAddress(void ** devPtr, const void * symbol);
__host__ tdpError_t tdpMallocManaged(void ** devptr, size_t size,
				     unsigned int flag);
__host__ tdpError_t tdpMemcpy(void * dst, const void * src, size_t count,
			      tdpMemcpyKind kind);
__host__ tdpError_t tdpMemcpyAsync(void * dst, const void * src, size_t count,
				   tdpMemcpyKind kind, tdpStream_t stream);
__host__ tdpError_t tdpMemcpyFromSymbol(void * dst, const void * symbol,
					size_t count, size_t offset,
					tdpMemcpyKind kind);
__host__ tdpError_t tdpMemcpyToSymbol(void * symbol, const void * src,
				      size_t count, size_t offset,
				      tdpMemcpyKind kind);
__host__ tdpError_t tdpMemset(void * devPtr, int value, size_t count);


__host__ __device__ tdpError_t tdpFree(void * devPtr);
__host__ __device__ tdpError_t tdpHostAlloc(void ** phost, size_t size,
					    unsigned int flags);
__host__ __device__ tdpError_t tdpMalloc(void ** devRtr, size_t size);

/* Type-specific atomic operations */

__device__ int atomicAddInt(int * sum, int val);
__device__ int atomicMaxInt(int * maxval, int val);
__device__ int atomicMinInt(int * minval, int val);
__device__ double atomicAddDouble(double * sum, double val);
__device__ double atomicMaxDouble(double * maxval, double val);
__device__ double atomicMinDouble(double * minval, double val);

/* Type-specific intra-block reductions. */

__device__ int atomicBlockAddInt(int * partsum);
__device__ double atomicBlockAddDouble(double * partsum);

#endif
