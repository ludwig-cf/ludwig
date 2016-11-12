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

  #define TARGET_MAX_THREADS_PER_BLOCK CUDA_MAX_THREADS_PER_BLOCK
  #define __target_simt_parallel_region()
  #define __target_simt_for(index, ndata, stride) __cuda_simt_for(index, ndata, stride)
  #define __target_simt_parallel_for(index, ndata, stride) __cuda_simt_parallel_for(index, ndata, stride)

  #define __target_simt_threadIdx_init()
  #define __target_syncthreads() __syncthreads()

  /* Additional host-side API */

  #define __host_threads_per_block() DEFAULT_TPB
  #define __host_launch(...) __cuda_launch(__VA_ARGS__)
  #define __host_launch4s(...) __cuda_launch4s(__VA_ARGS__)

#else

  /* x86. CUDA stub material plus host/target API */ 

  #include "target_x86.h"
  #include "cuda_stub_api.h"
  #define __inline__ __forceinline__

  /* Private interface wanted for these helper functions? */

  void  __x86_prelaunch(dim3 nblocks, dim3 nthreads);
  void  __x86_postlaunch(void);
  uint3 __x86_builtin_threadIdx_init(void);
  uint3 __x86_builtin_blockIdx_init(void);

  /* ... execution configuration should  set the global
   * gridDim and blockDim so they are available in kernel, and
   * sets the number of threads which could be < omp_get_max_threads()
   */

  #define __host_launch(kernel_function, nblocks, nthreads, ...)	\
    __x86_prelaunch(nblocks, nthreads);					\
    kernel_function(__VA_ARGS__);					\
    __x86_postlaunch();

  #define \
  __host_launch4s(kernel_function, nblocks, nthreads, shmem, stream, ...) \
  __host_launch(kernel_function, nblocks, nthreads, __VA_ARGS__)

  /* Within simt_parallel_region(), provide access/initialisation. */
  /* Must be a macro expansiosn. */

  #define __host_simt_threadIdx_init()			\
    uint3 threadIdx;					\
    threadIdx = __x86_builtin_threadIdx_init();

  /* May want another for blockIdx */

  #define TARGET_MAX_THREADS_PER_BLOCK X86_MAX_THREADS_PER_BLOCK

  #define __target_simt_parallel_region() __host_simt_parallel_region()
  #define __target_simt_for(index, ndata, stride) __host_simt_for(index, ndata, stride)

  #define __target_simt_parallel_for(index, ndata, stride) __host_simt_parallel_for(index, ndata, stride)
  #define __target_simt_threadIdx_init()  __host_simt_threadIdx_init()
  #define __target_syncthreads()          __host_barrier()
  #define __target_atomic()               __host_atomic()

  #define __host_threads_per_block()      __host_get_max_threads()
  #define __host_launch_kernel(...)       __host_launch(__VA_ARGS__)

#endif /* __NVCC__ */

#endif
