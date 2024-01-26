/*****************************************************************************
 *
 *  target_hip.c
 *
 *  HIP implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *    Nikola Vasilev: did the original implementation in 2020.
 *    Some contributions: Kevin Stratford (kevin@epcc.ed.ac.uk)
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

  fprintf(fp, "Target thread model: HIP.\n");
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

#if __HIPCC__

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
 *  tdpAtomicMaxDouble
 *
 *****************************************************************************/

__device__ double tdpAtomicMaxDouble(double * address, double val) {

  if (*address >= val) return *address;

  {
    unsigned long long * const address_as_ull = (unsigned long long *) address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
      assumed = old;
      if (__longlong_as_double(assumed) >= val) break;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
  }
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

#ifdef __HIP_DEVICE_COMPILE__
  if (ifail != tdpSuccess) {
    /*    printf("Line %d (%s): %s %s\n", line, file, hipGetErrorName(ifail),
	  hipGetErrorString(ifail));*/
    if (fatal) assert(0);
  }
#else
  if (ifail != tdpSuccess) {
    fprintf(stderr, "Line %d (%s): %s: %s\n", line, file,
	    hipGetErrorName(ifail), hipGetErrorString(ifail));
    if (fatal) exit(ifail);
  }
#endif

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
#ifdef __HIP_DEVICE_COMPILE__
  return hipErrorInvalidDevice;
#else
  if (attr == tdpDevAttrManagedMemory) {
    *value = 0;
    return hipSuccess;
  }
  else {
    return hipDeviceGetAttribute(value, attr, device);
  }
#endif
}

__host__ __device__ tdpError_t tdpDeviceGetCacheConfig(tdpFuncCache * cache) {
#ifdef __HIP_DEVICE_COMPILE__
  return hipErrorInitializationError;
#else
  return hipDeviceGetCacheConfig(cache);
#endif
}

__host__ tdpError_t tdpDeviceGetP2PAttribute(int * value,
					     tdpDeviceP2PAttr attr,
					     int srcDevice, int dstDevice) {
  return hipDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
}

__host__ tdpError_t tdpDeviceSetCacheConfig(tdpFuncCache cacheConfig) {

  return hipDeviceSetCacheConfig(cacheConfig);
}

__host__ tdpError_t tdpDeviceSynchronize(void) {

#ifdef __HIP_DEVICE_COMPILE__
  /* Do nothing */
  return hipSuccess;
#else
  return hipDeviceSynchronize();
#endif
}

__host__ tdpError_t tdpGetDeviceProperties(struct tdpDeviceProp * prop,
					   int device) {

  return hipGetDeviceProperties(prop, device);
}

__host__ tdpError_t tdpSetDevice(int device) {

  return hipSetDevice(device);
}

__host__ __device__ tdpError_t tdpGetDevice(int * device) {

#ifdef __HIP_DEVICE_COMPILE__
  /* Not available as device function in HIP so ... */
  return hipErrorNoDevice;
#else
  return hipGetDevice(device);
#endif
}

__host__ __device__ tdpError_t tdpGetDeviceCount(int * count) {

#ifdef __HIP_DEVICE_COMPILE__
  /* Not available as device function in HIP so ... */
  return hipErrorNoDevice;
#else
  return hipGetDeviceCount(count);
#endif
}

/* Error handling */

__host__ __device__ const char * tdpGetErrorName(tdpError_t error) {

#ifdef __HIP_DEVICE_COMPILE__
  /* Not available as device function, so improvise... */
  const char * err = "tdpGetErrorName: unrecognised (device) error code";
  return err;
#else
  return hipGetErrorName(error);
#endif
}


__host__ __device__ const char * tdpGetErrorString(tdpError_t error) {

#ifdef __HIP_DEVICE_COMPILE__
  /* Not available as device function, so improvise... */
  const char * err = "tdpGetErrorString: unrecognised (device) error code";
  return err;
#else
  return hipGetErrorString(error);
#endif
}

__host__ __device__ tdpError_t tdpGetLastError(void) {

#ifdef __HIP_DEVICE_COMPILE__
  /* Not available as device function, so improvise ... */
  return hipErrorInvalidDeviceFunction;
#else
  return hipGetLastError();
#endif
}

__host__ __device__ tdpError_t tdpPeekAtLastError(void) {

#ifdef __HIP_DEVICE_COMPILE__
  /* Not available as device function, so ... */
  return hipErrorInvalidDeviceFunction;
#else
  return hipPeekAtLastError();
#endif
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

__host__ tdpError_t tdpMemcpyPeer(void * dst, int dstDevice, const void * src,
				  int srcDevice, size_t count) {

  return hipMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

__host__ tdpError_t tdpMemcpyPeerAsync(void * dst, int dstDevice,
				       const void * src, int srcDevice,
				       size_t count, tdpStream_t stream) {

  return hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
}

__host__ __device__ tdpError_t tdpMalloc(void ** devptr, size_t size) {

#ifdef __HIP_DEVICE_COMPILE__
  assert(0); /* Should not be here. */
  return hipErrorMemoryAllocation;
#else
  return hipMalloc(devptr, size);
#endif
}

__host__ tdpError_t tdpMemset(void * devptr, int value, size_t count) {

  return hipMemset(devptr, value, count);
}

__host__ __device__ tdpError_t tdpFree(void * devptr) {

#ifdef __HIP_DEVICE_COMPILE__
  assert(0); /* cf. hipMalloc() */
  return hipErrorInvalidValue;
#else
  return hipFree(devptr);
#endif
}

__host__ tdpError_t tdpHostAlloc(void ** phost, size_t size,
				 unsigned int flags) {

  return hipHostMalloc(phost, size, flags);
}

/* Peer device memory access */

__host__ tdpError_t tdpDeviceCanAccessPeer(int * canAccessPeer, int device,
					   int peerDevice) {

  return hipDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
}

__host__ tdpError_t tdpDeviceDisablePeerAccess(int peerDevice) {

  return hipDeviceDisablePeerAccess(peerDevice);
}

__host__ tdpError_t tdpDeviceEnablePeerAccess(int peerDevice,
					      unsigned int flags) {

  return hipDeviceEnablePeerAccess(peerDevice, flags);
}

/*****************************************************************************
 *
 *  tdpGraphAddKernelNode
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphAddKernelNode(tdpGraphNode_t * pGraphNode,
                                          tdpGraph_t graph,
                                          const tdpGraphNode_t * pDependencies,
                                          size_t numDependencies,
                                          const tdpKernelNodeParams * nParams) {
  return hipGraphAddKernelNode(pGraphNode, graph, pDependencies,
			       numDependencies, nParams);
}

/*****************************************************************************
 *
 *  tdpGraphAddMemcpyNode
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphAddMemcpyNode(tdpGraphNode_t * pGraphNode,
                                          tdpGraph_t graph,
                                          const tdpGraphNode_t * pDependencies,
                                          size_t numDependencies,
                                          const tdpMemcpy3DParms * copyParams) {
  return hipGraphAddMemcpyNode(pGraphNode, graph, pDependencies,
			       numDependencies, copyParams);
}


/*****************************************************************************
 *
 *  tdpGraphCreate
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphCreate(tdpGraph_t * pGraph, unsigned int flags) {

  return hipGraphCreate(pGraph, flags);
}


/*****************************************************************************
 *
 *  tdpGraphDestroy
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphDestroy(tdpGraph_t graph) {

  return hipGraphDestroy(graph);
}

/*****************************************************************************
 *
 *  tdpGraphInstantiate
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphInstantiate(tdpGraphExec_t * pGraphExec,
                                        tdpGraph_t graph,
                                        unsigned long long flags) {

  /* Note API has changed between CUDA 11 and CUDA 12 */
  return hipGraphInstantiate(pGraphExec, graph, NULL, NULL, flags);
}

/*****************************************************************************
 *
 *  tdpGraphLaunch
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphLaunch(tdpGraphExec_t exec, tdpStream_t stream) {

  return hipGraphLaunch(exec, stream);
}

/*****************************************************************************
 *
 *  make_tdpExtent
 *
 *****************************************************************************/

__host__ struct tdpExtent make_tdpExtent(size_t w, size_t h, size_t d) {

  struct hipExtent extent = make_hipExtent(w, h, d);

  return extent;
}

/*****************************************************************************
 *
 *  make_tdpPos
 *
 *****************************************************************************/

__host__ struct tdpPos make_tdpPos(size_t x, size_t y, size_t z) {

  struct hipPos pos = make_hipPos(x, y, z);

  return pos;
}

/*****************************************************************************
 *
 *  make_tdpPitchedPtr
 *
 *****************************************************************************/

__host__ struct tdpPitchedPtr make_tdpPitchedPtr(void * d, size_t p,
                                                 size_t xsz, size_t ysz) {

  struct hipPitchedPtr ptr = make_hipPitchedPtr(d, p, xsz, ysz);

  return ptr;
}
