/****************************************************************************
 *
 *  target_cuda.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 * (c) 2018-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_TARGET_CUDA_H
#define LUDWIG_TARGET_CUDA_H

#include "cuda_runtime_api.h"

typedef cudaFuncCache tdpFuncCache;

#define tdpFuncCachePreferNone   cudaFuncCachePreferNone
#define tdpFuncCachePreferShared cudaFuncCachePreferShared
#define tdpFuncCachePreferL1     cudaFuncCachePreferL1
#define tdpFuncCahcePreferEqual  cudaFuncCachePreferEqual

/* enums */

typedef cudaMemcpyKind    tdpMemcpyKind;
typedef cudaDeviceAttr    tdpDeviceAttr;
typedef cudaDeviceP2PAttr tdpDeviceP2PAttr;

/* defines */

#define tdpDeviceProp           cudaDeviceProp
#define tdpDevAttrManagedMemory cudaDevAttrManagedMemory
#define tdpSuccess              cudaSuccess

/* cudaMemcpyKind */

#define tdpMemcpyHostToDevice cudaMemcpyHostToDevice
#define tdpMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define tdpMemcpyHostToHost cudaMemcpyHostToHost
#define tdpMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

/* cudaDeviceP2PAttr */
/* Note "tdpDevP@PAttrArray..." */

#define tdpDevP2PAttrPerformanceRank       cudaDevP2PAttrPerformanceRank
#define tdpDevP2PAttrAccessSupported       cudaDevP2PAttrAccessSupported
#define tdpDevP2PAttrNativeAtomicSupported cudaDevP2PAttrNativeAtomicSupported
#define tdpDevP2PAttrArrayAccessSupported  cudaDevP2PAttrCudaArrayAccessSupported

#define tdpMemAttachHost   cudaMemAttachHost
#define tdpMemAttachGlobal cudaMemAttachGlobal

#define tdpHostAllocDefault cudaHostAllocDefault

typedef cudaStream_t tdpStream_t;
typedef cudaError_t tdpError_t;

/* Graph API and related */

typedef cudaArray_t     tdpArray_t;

typedef cudaGraph_t     tdpGraph_t;
typedef cudaGraphExec_t tdpGraphExec_t;
typedef cudaGraphNode_t tdpGraphNode_t;

typedef cudaKernelNodeParams tdpKernelNodeParams;
typedef cudaMemcpy3DParms    tdpMemcpy3DParms;

#define tdpExtent       cudaExtent
#define tdpPos          cudaPos
#define tdpPitchedPtr   cudaPitchedPtr

__host__ tdpError_t tdpGraphAddKernelNode(tdpGraphNode_t * pGraphNode,
                                          tdpGraph_t graph,
                                          const tdpGraphNode_t * pDependencies,
                                          size_t numDependencies,
                                          const tdpKernelNodeParams * nParams);
__host__ tdpError_t tdpGraphAddMemcpyNode(tdpGraphNode_t * pGraphNode,
                                          tdpGraph_t graph,
                                          const tdpGraphNode_t * pDependencies,
                                          size_t numDependencies,
                                          const tdpMemcpy3DParms * copyParams);
__host__ tdpError_t tdpGraphCreate(tdpGraph_t * pGraph, unsigned int flags);
__host__ tdpError_t tdpGraphDestroy(tdpGraph_t graph);
__host__ tdpError_t tdpGraphInstantiate(tdpGraphExec_t * pGraphExec,
                                        tdpGraph_t graph,
                                        unsigned long long flags);
__host__ tdpError_t tdpGraphLaunch(tdpGraphExec_t exec, tdpStream_t stream);

__host__ struct tdpExtent make_tdpExtent(size_t w, size_t h, size_t d);
__host__ struct tdpPos    make_tdpPos(size_t x, size_t y, size_t z);
__host__ struct tdpPitchedPtr make_tdpPitchedPtr(void * d, size_t p,
                                                 size_t xsz, size_t ysz);


#define TARGET_MAX_THREADS_PER_BLOCK 128
#define TARGET_PAD                     1

/* Macros for calls involing device symbols */

#define tdpSymbol(x) x
#define tdpGetSymbolAddress(dst, symbol)		\
        tdpAssert(cudaGetSymbolAddress(dst, symbol))
#define tdpMemcpyToSymbol(symbol, src, count, offset, kind)	\
        tdpAssert(cudaMemcpyToSymbol(symbol, src, count, offset, kind))
#define tdpMemcpyFromSymbol(dst, symbol, count, offset, kind) \
        tdpAssert(cudaMemcpyFromSymbol(dst, symbol, count, offset, kind))

#define	tdpLaunchKernel(kernel, nblocks, nthreads, shmem, stream, ...) \
  kernel<<<nblocks, nthreads, shmem, stream>>>(__VA_ARGS__);

#define for_simt_parallel(index, ndata, stride) \
  index = (stride)*(blockIdx.x*blockDim.x + threadIdx.x); \
  if (index < (ndata))

#define for_simd_v(iv, nsimdvl) \
  for (iv = 0; iv < (nsimdvl); iv++)

#define for_simd_v_reduction(iv, nsimdvl, clause) \
  for (iv = 0; iv < (nsimdvl); iv++)

#define tdp_get_max_threads() TARGET_MAX_THREADS_PER_BLOCK

#endif
