/****************************************************************************
 *
 *  target_hip.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 * (c) 2020-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *    Nikola Vasilev did the original implementation in 2020.
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_TARGET_HIP_H
#define LUDWIG_TARGET_HIP_H

#include <hip/hip_runtime.h>

typedef hipFuncCache_t tdpFuncCache;

#define tdpFuncCachePreferNone   hipFuncCachePreferNone
#define tdpFuncCachePreferShared hipFuncCachePreferShared
#define tdpFuncCachePreferL1     hipFuncCachePreferL1
#define tdpFuncCahcePreferEqual  hipFuncCachePreferEqual

/* enums */

typedef hipMemcpyKind            tdpMemcpyKind;
typedef hipDeviceAttribute_t     tdpDeviceAttr;
typedef hipDeviceP2PAttr         tdpDeviceP2PAttr;

/* defines */

#define tdpDeviceProp hipDeviceProp_t

/* There is no e.g., hipDevAttrManagedMemory */
/* We therefore provide an explicit value at trap it in tdpGetDeviceAttr() */
#define tdpDevAttrManagedMemory (tdpDeviceAttr) 83

#define tdpSuccess hipSuccess
#define tdpMemcpyHostToDevice hipMemcpyHostToDevice
#define tdpMemcpyDeviceToHost hipMemcpyDeviceToHost
#define tdpMemcpyHostToHost hipMemcpyHostToHost
#define tdpMemcpyDeviceToDevice hipMemcpyDeviceToDevice

/* P2P */

#define tdpDevP2PAttrPerformaceRank        hipDevP2PAttrPerformanceRank
#define tdpDevP2PAttrAccessSupported       hipDevP2PAttrAccessSupported
#define tdpDevP2PAttrNativeAtomicSupported hipDevP2PAttrNativeAtomicSupported
#define tdpDevP2PAttrArrayAccessSupported  hipDevP2PAttrHipArrayAccessSupported


#define tdpMemAttachHost   hipMemAttachHost
#define tdpMemAttachGlobal hipMemAttachGlobal

#define tdpHostAllocDefault hipHostMallocDefault

typedef hipStream_t tdpStream_t;
typedef hipError_t tdpError_t;


#define TARGET_MAX_THREADS_PER_BLOCK 128
#define TARGET_PAD                     1

/* Macros for calls involing device symbols */

#define tdpSymbol(x) HIP_SYMBOL(x)
#define tdpGetSymbolAddress(dst, symbol) \
        tdpAssert(hipGetSymbolAddress(dst, symbol))
#define tdpMemcpyToSymbol(symbol, src, count, offset, kind)	\
        tdpAssert(hipMemcpyToSymbol(symbol, src, count, offset, kind))
#define tdpMemcpyFromSymbol(dst, symbol, count, offset, kind) \
        tdpAssert(hipMemcpyFromSymbol(dst, symbol, count, offset, kind))

#define	tdpLaunchKernel(kernel, nblocks, nthreads, shmem, stream, ...) \
  hipLaunchKernelGGL(kernel, nblocks, nthreads, shmem, stream, __VA_ARGS__);

#define for_simt_parallel(index, ndata, stride) \
  index = (stride)*(blockIdx.x*blockDim.x + threadIdx.x); \
  if (index < (ndata))

#define for_simd_v(iv, nsimdvl) \
  for (iv = 0; iv < (nsimdvl); iv++)

#define for_simd_v_reduction(iv, nsimdvl, clause) \
  for (iv = 0; iv < (nsimdvl); iv++)

#define tdp_get_max_threads() TARGET_MAX_THREADS_PER_BLOCK

#endif
