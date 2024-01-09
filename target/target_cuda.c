/*****************************************************************************
 *
 *  target_cuda.c
 *
 *  CUDA implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
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
 *  tdpAtomicMaxDouble
 *
 *****************************************************************************/

__device__ double tdpAtomicMaxDouble(double * address, double val) {

  assert(address);

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
		    (fmin(val, __longlong_as_double(assumed))));
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

__host__ tdpError_t tdpDeviceGetP2PAttribute(int * value,
					     tdpDeviceP2PAttr attr,
					     int srcDevice, int dstDevice) {
  return cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
}

__host__ tdpError_t tdpDeviceSetCacheConfig(tdpFuncCache cacheConfig) {

  return cudaDeviceSetCacheConfig(cacheConfig);
}

/* nb. CUDA 11.6 has deprecated the __device__ version of
 * cudaDeviceSynchronize(). It should be used only on the host. */
__host__ tdpError_t tdpDeviceSynchronize(void) {

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

__host__ tdpError_t tdpMemcpyPeer(void * dst, int dstDevice, const void * src,
				  int srcDevice, size_t count) {

  return cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

__host__ tdpError_t tdpMemcpyPeerAsync(void * dst, int dstDevice,
				       const void * src, int srcDevice,
				       size_t count, tdpStream_t stream) {

  return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
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

__host__ tdpError_t tdpDeviceCanAccessPeer(int * canAccessPeer, int device,
					   int peerDevice) {

  return cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
}

__host__ tdpError_t tdpDeviceDisablePeerAccess(int peerDevice) {

  return cudaDeviceDisablePeerAccess(peerDevice);
}

__host__ tdpError_t tdpDeviceEnablePeerAccess(int peerDevice,
					      unsigned int flags) {

  return cudaDeviceEnablePeerAccess(peerDevice, flags);
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
  return cudaGraphAddKernelNode(pGraphNode, graph, pDependencies,
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
  return cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies,
				numDependencies, copyParams);
}

/*****************************************************************************
 *
 *  tdpGraphCreate
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphCreate(tdpGraph_t * pGraph, unsigned int flags) {

  return cudaGraphCreate(pGraph, flags);
}

/*****************************************************************************
 *
 *  tdpGraphDestroy
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphDestroy(tdpGraph_t graph) {

  return cudaGraphDestroy(graph);
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
  return cudaGraphInstantiate(pGraphExec, graph, NULL, NULL, flags);
}

/*****************************************************************************
 *
 *  tdpGraphLaunch
 *
 *****************************************************************************/

__host__ tdpError_t tdpGraphLaunch(tdpGraphExec_t exec, tdpStream_t stream) {

  return cudaGraphLaunch(exec, stream);
}

/*****************************************************************************
 *
 *  make_tdpExtent
 *
 *****************************************************************************/

__host__ struct tdpExtent make_tdpExtent(size_t w, size_t h, size_t d) {

  struct cudaExtent extent = make_cudaExtent(w, h, d);

  return extent;
}

/*****************************************************************************
 *
 *  make_tdpPos
 *
 *****************************************************************************/

__host__ struct tdpPos make_tdpPos(size_t x, size_t y, size_t z) {

  struct cudaPos pos = make_cudaPos(x, y, z);

  return pos;
}

/*****************************************************************************
 *
 *  make_tdpPitchedPtr
 *
 *****************************************************************************/

__host__ struct tdpPitchedPtr make_tdpPitchedPtr(void * d, size_t p,
                                                 size_t xsz, size_t ysz) {

  struct cudaPitchedPtr ptr = make_cudaPitchedPtr(d, p, xsz, ysz);

  return ptr;
}
