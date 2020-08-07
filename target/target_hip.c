/*****************************************************************************
 *
 *  target_hip.c
 *
 *  CUDA implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <hip/hip_runtime.h>
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
//#ifdef __HIPCC__

  if (ifail != tdpSuccess) {
    printf("Line %d (%s): %s %s\n", line, file, hipGetErrorName(ifail),
	   hipGetErrorString(ifail));
    if (fatal) assert(0);
  }
//#else
//  if (ifail != tdpSuccess) {
//    fprintf(stderr, "Line %d (%s): %s: %s\n", line, file,
//	    hipGetErrorName(ifail), hipGetErrorString(ifail));
//    if (fatal) exit(ifail);
//  }
//#endif

  return;
}

/*****************************************************************************
 *
 *  Stubs cf hip/hip_runtime_api.h
 *
 *****************************************************************************/

__host__ __device__ tdpError_t tdpDeviceGetAttribute(int * value,
						     tdpDeviceAttr attr,
						     int device) {

  return hipDeviceGetAttribute(value, attr, device);
}

__host__ __device__ tdpError_t tdpDeviceGetCacheConfig(tdpFuncCache * cache) {
  return hipDeviceGetCacheConfig(cache);
}

__host__ tdpError_t tdpDeviceSetCacheConfig(tdpFuncCache cacheConfig) {

  return hipDeviceSetCacheConfig(cacheConfig);
}

__host__ __device__ tdpError_t tdpDeviceSynchronize(void) {

  return hipDeviceSynchronize();
}

__host__ tdpError_t tdpGetDeviceProperties(struct tdpDeviceProp * prop,
					   int device) {

  return hipGetDeviceProperties(prop, device);
}

__host__ tdpError_t tdpSetDevice(int device) {

  return hipSetDevice(device);
}

__host__ __device__ tdpError_t tdpGetDevice(int * device) {

  return hipGetDevice(device);
}

__host__ __device__ tdpError_t tdpGetDeviceCount(int * count) {

  return hipGetDeviceCount(count);
}

/* Error handling */

__host__ __device__ const char * tdpGetErrorName(tdpError_t error) {

  return hipGetErrorName(error);
}


__host__ __device__ const char * tdpGetErrorString(tdpError_t error) {

  return hipGetErrorString(error);
}

__host__ __device__ tdpError_t tdpGetLastError(void) {

  return hipGetLastError();
}

__host__ __device__ tdpError_t tdpPeekAtLastError(void) {

  return hipPeekAtLastError();
}


/* Stream management */

__host__ tdpError_t tdpStreamCreate(tdpStream_t * stream) {

  return hipStreamCreate(stream);
}

__host__ tdpError_t tdpStreamDestroy(tdpStream_t stream) {

  return hipStreamDestroy(stream);
}

__host__ tdpError_t tdpStreamSynchronize(tdpStream_t stream) {

  return hipStreamSynchronize(stream);
}

/* Memory management */

__host__ tdpError_t tdpFreeHost(void * phost) {

  return hipHostFree(phost);
}

__host__ tdpError_t tdpMallocManaged(void ** devptr, size_t size,
				     unsigned int flag) {

  return hipMallocManaged(devptr, size, flag);
}

__host__ tdpError_t tdpMemcpy(void * dst, const void * src, size_t count,
			      tdpMemcpyKind kind) {

  return hipMemcpy(dst, src, count, kind);
}

__host__ tdpError_t tdpMemcpyAsync(void * dst, const void * src, size_t count,
				   tdpMemcpyKind kind, tdpStream_t stream) {

  return hipMemcpyAsync(dst, src, count, kind, stream);
}

__host__ __device__ tdpError_t tdpMalloc(void ** devptr, size_t size) {

  return hipMalloc(devptr, size);
}

__host__ tdpError_t tdpMemset(void * devptr, int value, size_t count) {

  return hipMemset(devptr, value, count);
}

__host__ __device__ tdpError_t tdpFree(void * devptr) {

  return hipFree(devptr);
}

__host__ tdpError_t tdpHostAlloc(void ** phost, size_t size,
				 unsigned int flags) {

  return hipHostMalloc(phost, size, flags);
}
