/*****************************************************************************
 *
 *  target_api.h
 *
 *****************************************************************************/

#ifndef TARGET_API_H
#define TARGET_API_H

/* Target-independent host-side API (x86 at the moment) */

#include "target_x86.h"

/* Target-dependent API */

#ifdef __NVCC__

  /* CUDA */

  #include "target_cuda.h"
  #include "cuda_runtime_api.h"
  #define __inline__ __forceinline__

  /* Macros for calls involing device symbols */

  #define tdpSymbol(x) x
  #define tdpGetSymbolAddress(dst, symbol) \
          tdpAssert(cudaGetSymbolAddress(dst, symbol))
  #define tdpMemcpyToSymbol(symbol, src, count, offset, kind)	\
          tdpAssert(cudaMemcpyToSymbol(symbol, src, count, offset, kind))
  #define tdpMemcpyFromSymbol(dst, symbol, count, offset, kind) \
          tdpAssert(cudaMemcpyFromSymbol(dst, symbol, count, offset, kind))

  #define TARGET_MAX_THREADS_PER_BLOCK CUDA_MAX_THREADS_PER_BLOCK
  #define tdp_simd_for(iv, nsimdvl) tdp_cuda_simd_for(iv, nsimdvl)
  #define tdp_simt_for(index, ndata, stride) tdp_cuda_simt_for(index, ndata, stride)

  /* Additional host-side API */

  #define tdp_host_threads_per_block() DEFAULT_TPB
  #define \
    tdpLaunchKernel(kernel_function, nblocks, nthreads, shmem, stream, ...) \
    kernel_function<<<nblocks, nthreads, shmem, stream>>>(__VA_ARGS__);

#else

  /* x86. CUDA stub material plus host/target API */ 

  #include "target_x86.h"
  #include "cuda_stub_api.h"
  #define __inline__ __forceinline__

  /* Symbols must be addressed as a pointer */

  #define tdpSymbol(x) &(x)

  /* Private interface wanted for these helper functions? */

  void  tdp_x86_prelaunch(dim3 nblocks, dim3 nthreads);
  void  tdp_x86_postlaunch(void);

  /* ... execution configuration should  set the global
   * gridDim and blockDim so they are available in kernel, and
   * sets the number of threads which could be < omp_get_max_threads()
   */

  #define \
    tdpLaunchKernel(kernel_function, nblocks, nthreads, shmem, stream, ...)\
    tdp_host_parallel_region()\
    { \
    tdp_x86_prelaunch(nblocks, nthreads);\
    kernel_function(__VA_ARGS__);\
    tdp_x86_postlaunch();\
    }

  #define TARGET_MAX_THREADS_PER_BLOCK X86_MAX_THREADS_PER_BLOCK

  #define targetdp_simd_for(iv, nsimdvl) tdp_host_simd_for(iv, nsimdvl)
  #define targetdp_simt_for(index, ndata, stride) tdp_host_simt_for(index, ndata, stride)
  #define __syncthreads()                 tdp_host_barrier()
  #define tdp_host_threads_per_block()    tdp_host_get_max_threads()



#endif /* __NVCC__ */

#endif
