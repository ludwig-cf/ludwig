/*****************************************************************************
 *
 *  target_api.h
 *
 *****************************************************************************/

#ifndef TARGET_API_H
#define TARGET_API_H

/* Interface */

#ifdef __NVCC__

  /* CUDA */

  #include "target_cuda.h"
  #include "cuda_runtime_api.h"

  #define TARGET_MAX_THREADS_PER_BLOCK CUDA_MAX_THREADS_PER_BLOCK
  #define __target_simt_parallel_region()
  #define __target_simt_for(index, ndata, stride) __cuda_simt_for(index, ndata, stride)
  #define __target_simt_parallel_for(index, ndata, stride) __cuda_simt_parallel_for(index, ndata, stride)

  #define __target_simt_threadIdx_init()

  /* Additional host-side API */

  #define __host_threads_per_block() DEFAULT_TPB
  #define __host_launch_kernel(...) __cuda_launch(__VA_ARGS__)

#else

  /* x86 */ 

  #include "target_x86.h"
  #include "cuda_stub_api.h"

  #define TARGET_MAX_THREADS_PER_BLOCK X86_MAX_THREADS_PER_BLOCK
  #define __target_simt_parallel_region() __x86_simt_parallel_region()
  #define __target_simt_for(index, ndata, stride) __x86_simt_for(index, ndata, stride)
  #define __target_simt_parallel_for(index, ndata, stride) __x86_simt_parallel_for(indx, ndata, stride)

  #define __target_simt_threadIdx_init()  __x86_simt_threadIdx_init()

  #define __syncthreads()                 __x86_barrier()

  #define __host_threads_per_block()      __x86_get_max_threads()
  #define __host_launch_kernel(...)       __x86_launch(__VA_ARGS__)

#endif /* __NVCC__ */


#endif
