/*****************************************************************************
 *
 *  target_cuda.c
 *
 *  CUDA implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

#include "target.h"

/*****************************************************************************
 *
 *  tdpThreadModelInfo
 *
 *  Provide some information on the model, usually to stdout.
 *
 ****************************************************************************/

__host__ tdpError_t tdpThreadModelInfo(FILE * fp) {

  assert(fp);

  fprintf(fp, "Target thread model: CUDA.\n");
  fprintf(fp, "Default threads per block: %d; max. threads per block: %d.\n",
	  tdp_get_max_threads(), 1024);

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpAtomicAddInt
 *
 *****************************************************************************/

__device__ int tdpAtomicAddInt(int * sum, int val) {

  return atomicAdd(sum, val);
}

/*****************************************************************************
 *
 *  tdpAtomicAddDouble
 *
 *  See CUDA C programming guide section on atomics.
 *
 *  The original (I think) from:
 *  https://devtalk.nvidia.com/default/topic/529341/?comment=3739638
 *
 *****************************************************************************/

__device__ double tdpAtomicAddDouble(double * sum, double val) {

#if __CUDA_ARCH__ >= 600

  return atomicAdd(sum, val);

#else

  unsigned long long int * address_as_ull = (unsigned long long int *) sum;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);

#endif
}

/*****************************************************************************
 *
 *  tdpAtomicMinDouble
 *
 *****************************************************************************/

__device__ double tdpAtomicMinDouble(double * minval, double val) {

  unsigned long long int * address_as_ull = (unsigned long long int *) minval;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong
		    (fminf(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

/*****************************************************************************
 *
 *  tdpAtomicBlockAddInt
 *
 *  partsum is per-thread contribution on input
 *  Returns on thread 0 the sum for block (other elements destroyed).
 *
 *****************************************************************************/

__device__ int tdpAtomicBlockAddInt(int * partsum) {

  int istr;
  int nblock;
  int nthread = TARGET_MAX_THREADS_PER_BLOCK;
  int idx = threadIdx.x;

  nblock = pow(2.0, ceil(log(1.0*nthread)/log(2.0)));

  for (istr = nblock/2; istr > 0; istr /= 2) {
    __syncthreads();
    if (idx < istr && idx + istr < nthread) {
      partsum[idx] += partsum[idx + istr];
    }
  }

  return partsum[0];
}

/*****************************************************************************
 *
 *  tdpAtomicBlockAddDouble
 *
 *  Type-specfic version for double
 *
 *****************************************************************************/

__device__ double tdpAtomicBlockAddDouble(double * partsum) {

  int istr;
  int nblock;
  int nthread = TARGET_MAX_THREADS_PER_BLOCK;
  int idx = threadIdx.x;

  nblock = pow(2.0, ceil(log(1.0*nthread)/log(2.0)));

  for (istr = nblock/2; istr > 0; istr /= 2) {
    __syncthreads();
    if (idx < istr && idx + istr < nthread) {
      partsum[idx] += partsum[idx + istr];
    }
  }

  return partsum[0];
}

/*****************************************************************************
 *
 *  tdpErrorHandler
 *
 *****************************************************************************/

__host__ __device__ void tdpErrorHandler(tdpError_t ifail, const char * file,
					 int line, int fatal) {
#ifdef __CUDA_ARCH__

  if (ifail != tdpSuccess) {
    printf("Line %d (%s): %s %s\n", line, file, cudaGetErrorName(ifail),
	   cudaGetErrorString(ifail));
    if (fatal) assert(0);
  }
#else
  if (ifail != tdpSuccess) {
    fprintf(stderr, "Line %d (%s): %s: %s\n", line, file,
	    cudaGetErrorName(ifail), cudaGetErrorString(ifail));
    if (fatal) exit(ifail);
  }
#endif

  return;
}

/*****************************************************************************
 *
 *  Stubs cf cuda_runtime_api.h
 *
 *****************************************************************************/

__host__ __device__ tdpError_t tdpDeviceGetAttribute(int * value,
						     tdpDeviceAttr attr,
						     int device) {

  return cudaDeviceGetAttribute(value, attr, device);
}

__host__ __device__ tdpError_t tdpDeviceGetCacheConfig(tdpFuncCache * cache) {
  return cudaDeviceGetCacheConfig(cache);
}

__host__ tdpError_t tdpDeviceSetCacheConfig(tdpFuncCache cacheConfig) {

  return cudaDeviceSetCacheConfig(cacheConfig);
}

__host__ __device__ tdpError_t tdpDeviceSynchronize(void) {

  return cudaDeviceSynchronize();
}

__host__ tdpError_t tdpGetDeviceProperties(struct tdpDeviceProp * prop,
					   int device) {

  return cudaGetDeviceProperties(prop, device);
}

__host__ tdpError_t tdpSetDevice(int device) {

  return cudaSetDevice(device);
}

__host__ __device__ tdpError_t tdpGetDevice(int * device) {

  return cudaGetDevice(device);
}

__host__ __device__ tdpError_t tdpGetDeviceCount(int * count) {

  return cudaGetDeviceCount(count);
}

/* Error handling */

__host__ __device__ const char * tdpGetErrorName(tdpError_t error) {

  return cudaGetErrorName(error);
}


__host__ __device__ const char * tdpGetErrorString(tdpError_t error) {

  return cudaGetErrorString(error);
}

__host__ __device__ tdpError_t tdpGetLastError(void) {

  return cudaGetLastError();
}

__host__ __device__ tdpError_t tdpPeekAtLastError(void) {

  return cudaPeekAtLastError();
}


/* Stream management */

__host__ tdpError_t tdpStreamCreate(tdpStream_t * stream) {

  return cudaStreamCreate(stream);
}

__host__ tdpError_t tdpStreamDestroy(tdpStream_t stream) {

  return cudaStreamDestroy(stream);
}

__host__ tdpError_t tdpStreamSynchronize(tdpStream_t stream) {

  return cudaStreamSynchronize(stream);
}

/* Memory management */

__host__ tdpError_t tdpFreeHost(void * phost) {

  return cudaFreeHost(phost);
}

__host__ tdpError_t tdpMallocManaged(void ** devptr, size_t size,
				     unsigned int flag) {

  return cudaMallocManaged(devptr, size, flag);
}

__host__ tdpError_t tdpMemcpy(void * dst, const void * src, size_t count,
			      tdpMemcpyKind kind) {

  return cudaMemcpy(dst, src, count, kind);
}

__host__ tdpError_t tdpMemcpyAsync(void * dst, const void * src, size_t count,
				   tdpMemcpyKind kind, tdpStream_t stream) {

  return cudaMemcpyAsync(dst, src, count, kind, stream);
}

__host__ __device__ tdpError_t tdpMalloc(void ** devptr, size_t size) {

  return cudaMalloc(devptr, size);
}

__host__ tdpError_t tdpMemset(void * devptr, int value, size_t count) {

  return cudaMemset(devptr, value, count);
}

__host__ __device__ tdpError_t tdpFree(void * devptr) {

  return cudaFree(devptr);
}

__host__ tdpError_t tdpHostAlloc(void ** phost, size_t size,
				 unsigned int flags) {

  return cudaHostAlloc(phost, size, flags);
}
