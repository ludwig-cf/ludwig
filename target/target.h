/*****************************************************************************
 *
 *  target.h
 *
 *  Target API.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stddef.h>
#include <stdio.h>

#ifndef LUDWIG_TARGET_H
#define LUDWIG_TARGET_H

/* Implementation details */

#ifdef __NVCC__
#include "target_cuda.h"
#else
#include "target_x86.h"
#endif

/* API */

/* Device management */

__host__ tdpError_t tdpDeviceSetCacheConfig(tdpFuncCache cacheConfig);
__host__ tdpError_t tdpGetDeviceProperties(struct tdpDeviceProp * prop, int);
__host__ tdpError_t tdpSetDevice(int device);

__host__ __device__ tdpError_t tdpDeviceGetAttribute(int * value,
						     tdpDeviceAttr attr,
						     int device);
__host__ __device__ tdpError_t tdpDeviceGetCacheConfig(tdpFuncCache * cache);
__host__ __device__ tdpError_t tdpDeviceSynchronize(void);
__host__ __device__ tdpError_t tdpGetDevice(int * device);
__host__ __device__ tdpError_t tdpGetDeviceCount(int * count);

/* Error handling */

__host__ __device__ const char * tdpGetErrorName(tdpError_t error);
__host__ __device__ const char * tdpGetErrorString(tdpError_t error);
__host__ __device__ tdpError_t tdpGetLastError(void);
__host__ __device__ tdpError_t tdpPeekAtLastError(void);

/* Stream management */

__host__ tdpError_t tdpStreamCreate(tdpStream_t * stream);
__host__ tdpError_t tdpStreamDestroy(tdpStream_t stream);
__host__ tdpError_t tdpStreamSynchronize(tdpStream_t stream);

/* Execution control */

/* tdpLaunchKernel() is implementation-dependant */

/* Memory management */

__host__ tdpError_t tdpFreeHost(void * phost);
__host__ tdpError_t tdpHostAlloc(void ** phost, size_t size,
				 unsigned int flags);
__host__ tdpError_t tdpMallocManaged(void ** devptr, size_t size,
				     unsigned int flag);
__host__ tdpError_t tdpMemcpy(void * dst, const void * src, size_t count,
			      tdpMemcpyKind kind);
__host__ tdpError_t tdpMemcpyAsync(void * dst, const void * src, size_t count,
				   tdpMemcpyKind kind, tdpStream_t stream);
__host__ tdpError_t tdpMemset(void * devPtr, int value, size_t count);


__host__ __device__ tdpError_t tdpFree(void * devPtr);
__host__ __device__ tdpError_t tdpMalloc(void ** devRtr, size_t size);

/* Memory management involving symbols
 * These are slightly awkward as there is never a host pointer to
 * device symbols. */

#ifdef __NVCC__
#else
__host__ tdpError_t tdpGetSymbolAddress(void ** devPtr, const void * symbol);
__host__ tdpError_t tdpMemcpyFromSymbol(void * dst, const void * symbol,
					size_t count, size_t offset,
					tdpMemcpyKind kind);
__host__ tdpError_t tdpMemcpyToSymbol(void * symbol, const void * src,
				      size_t count, size_t offset,
				      tdpMemcpyKind kind);
#endif

/* Additional API */

__host__ tdpError_t tdpThreadModelInfo(FILE * fp);

/* Type-specific atomic operations */

__device__ int tdpAtomicAddInt(int * sum, int val);
__device__ int tdpAtomicMaxInt(int * maxval, int val);
__device__ int tdpAtomicMinInt(int * minval, int val);
__device__ double tdpAtomicAddDouble(double * sum, double val);
__device__ double tdpAtomicMaxDouble(double * maxval, double val);
__device__ double tdpAtomicMinDouble(double * minval, double val);

/* Type-specific intra-block reductions. */

__device__ int tdpAtomicBlockAddInt(int * partsum);
__device__ double tdpAtomicBlockAddDouble(double * partsum);

/* Help for error checking */

__host__ __device__ void tdpErrorHandler(tdpError_t ifail, const char * file,
					 int line, int fatal);
#define tdpAssert(call) { tdpErrorHandler((call), __FILE__, __LINE__, 1); }

#endif
