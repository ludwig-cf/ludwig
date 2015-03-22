/* OpenMP trial */

/* The application is allowed to write a general CUDA-like kernel
 * which will translate into OpenMP
 */


/* PROGRAMMING INTERFACE */

/*
 *
 * C Extensions supported
 *
 *
 * Execution space qualifiers. The following specify whether a function
 * is callable from host or device or both.
 *
 * From CUDA          TargetDP alias
 *
 * __device__         __target__
 * __global__         __target_entry__
 * __host__           __targetHost__ (?)
 *
 * For x86 implementation all execution spaces collapse to host.
 *
 * The additional compiler directives are allowed. Whether they have
 * effect is compiler-dependent:
 *
 * __forceinline__    __forceline__
 * __noinline__       __noinline__
 *
 *
 *
 * Device memory qualifiers. The following may be used to specify
 * device memory space
 *
 * __device__         __target__
 * __constant__       __targetConst__
 * __shared__         TBC
 * __restrict__       __restrict__
 *
 * For x86 implementation these are respected in a threaded context.
 * Note __managed__ is not supported.
 *
 * Restriction (from CUDA): only __shared__ variables may be declared as
 * static within __device__ or __global__ functions.
 *
 * 
 *
 * Built-in variables. The following built-in variables (and corresponding
 * type definitions) are available:
 *
 * dim3  gridDim        gridDim
 * dim3  blockDim       blockDim
 * uint3 blockIdx       blockIdx
 * uint3 threadIdx      threadIdx
 * int   warpSize       warpSize
 *
 * Note for CUDA implementation all these are available in __device__
 * and __global__ functions. References from host execution space
 * are erroneous (nvcc error).
 * 
 * This x86 implementation has gridDim and
 * blockDim additionally in scope in __host__ functions. However,
 * references should be disallowed owing to "reserved status".
 *
 * Additional restriction:
 *
 * Initialisation of in-built variables is ensured via a call to:
 *
 *   __target_simt_threadIdx_init()
 *
 * within a target_simt_parallel_region() (see below). Failure to
 * do this will result in a compile-time error "variable not defined"
 * in the host implementation.
 * 
 *
 *
 * Synchronisation functions
 *
 * void __syncthreads(void)
 * 
 * CUDA supplies a number of others int __syncthreads_count(),
 * int __syncthreads_and() and int syncthreads_or() which would
 * require per-block reduction (awkward).
 *
 * CUDA also implements various __threadfence() and __atomic() functions
 * which I haven't looked at. It may be possible to ape these in the
 * OpenMP content.
 *
 *
 * Standard C functions.
 *
 * assert() is supported
 * printf() is supported with some limitations from CUDA.
 * malloc() etc are supported.
 *
 *
 * Execution configuration
 *
 * __host_launch_kernel(void * kernel_function, dim3 nblocks, dim3 nthreads,
 *                      ...)
 *
 * Note we enforce a strict C interface around the built-in
 * variables gridDim, blockDim, so no C++ polymorphism is allowed.
 *
 * extern __shared__ * variables with dynamic allocation could
 * be accommodated within the execution configuration by merely
 * allocating and deallocating the file-scope object, which is
 * has shared scope in the kernel.
 *
 *
 * In general CUDA has the extension to standard C
 *
 * function<<<nblocks, nthreads_per_block, nbytes_shared, stream>>>(...)
 *
 * which can be replaced by the more standard
 *
 * cudaLaunchKernel()  (API v7.0;  now "deprecated" is  cudaLaunch())
 *
 * and related calls. This host-side interface could also be supported.
 *
 *
 * Additional host extensions:
 *
 * int __host_simt_threads_per_block(void)
 *
 * Return default threads per block.
 *
 *
 * Additional kernel function extensions:
 *
 * __target_simt_parallel_region() {
 *   __target_simt_threadIdx_init();
 *   ...
 * }
 *
 * This is a structured block used to delimit the extent of threads
 * (specifically in the host implementation). Thread-private variables
 * must be declared within the scope of the parallel region. All
 * variables outside the block default to shared.
 *
 * Note __target_simt_builtin_init() ensures initialisation of the
 * built-in variables threadIdx and blockIdx as described above.
 *
 *
 * __target_simt_for(index, nadata, istride) {
 * }
 *
 * A structured block used to introduce worksharing. In the host
 * implementation this must be replaced by a loop:
 *
 * for (index = 0; index < ndata; index += istride) {
 * }
 *
 *
 * And ...
 *
 * __target_simt_parallel_for(index, ndata, istride) {
 * }
 *
 * Convenience structured block which combines __target_simt_parallel_region()
 * and __target_simt_for().
 *
 *
 *
 * Host-side API
 *
 * I suggest a direct 1-1 mapping for "low-level" cuda stub interface
 *
 * "DP" Memcpy() -> cudaMemcpy() etc from CUDA runtime
 *
 * plus "higher-level" API as required (for stuff with no direct
 * analogy in CUDA).
 *
 *
 *
 * Additional restrictions from CUDA:
 *
 * Pointers
 *
 * & obtains addresses valid only for the device for __device__
 * __shared__ and __constant__ variables. Addresses of __device__
 * and __constant__ variables obtained be dpeeGetSymbolAddress()
 * can only be used in the host execution space.
 *
 * Operators
 *
 * __constant__ variables may only be assigned in host code.
 * __shared__   variables cannot be initialised as part of a
 *              declaration.
 *
 * & may not be used with built-in variables.
 *
 *
 * Additional restrictions from OpenMP:
 *
 * Branching into/out of a parallel region is not allowed
 * (to include the use of return statements). 
 *
 */

#include <assert.h>
#include <stdio.h>

#include "targetDP.h"

/* SIMPLE EXAMPLE */

#define NDATA 8
#define SIMDVL 2

__global__ void kernel_trial(int ndata, double * data);

int main(int argc, char ** argv) {

  int ndata = NDATA;
  int n;
  dim3 nblocks;
  dim3 nthreads;
  double hdata[NDATA];
  double * data;

  for (n = 0; n < ndata; n++) {
    hdata[n] = 1.0*n;
  }

  cudaMalloc((void **) &data, NDATA*sizeof(double));
  /*__dpMemcpy(data, hdata, NDATA*sizeof(double), cudaMemcpyHostToDevice); */

  if (target_is_host()) {
    data = hdata;
  }
  else {
    cudaMemcpy(data, hdata, NDATA*sizeof(double), cudaMemcpyHostToDevice);
  }

  /* Block size.
   * For OpenMP, it seems reasonable to set the number of blocks to
   *   {1, 1, 1}
   * in (almost all) cases and rely on ndata to express the extent of
   * the parallel loop in the kernel. The number of threads is at most
   *   omp_get_max_threads()
   * and we will generally want to use as many threads as possible.
   *
   * For CUDA, the block size is dependent on problem size and threads
   * per block requested. Any given application may want a helper
   * function to work this out. */

  nblocks.x = 1; nblocks.y = 1; nblocks.z = 1;

  nthreads.x = __host_threads_per_block();
  nthreads.y = 1;
  nthreads.z = 1;

  __host_launch_kernel(kernel_trial, nblocks, nthreads, ndata, data);

  cudaDeviceSynchronize();

  /* __dpMemcpy();*/
  if (!target_is_host()) {
    copyFromTarget(hdata, data, NDATA*sizeof(double));
  }

  printf("ndara is %d\n", ndata);
  for (n = 0; n < ndata; n++) {
    printf("data[%2i] = %5.1f\n", n, hdata[n]);
  }

  return 0;
}

/*
 * Target-side interface
 *
 *  Additional restrictions:
 *
 * __shared__ declarations must precede target_simt_region()
 *   (CUDA allows them in any scoping unit in kernel) so that
 * OpenMP sees shared memory
 *
 * In host code thread private variables must come after
 * target_simt_region() (cf. whole function scope is private in CUDA).
 *
 * __simt_threadIdx_init() must occur inside target_simt_parallel_region()
 * and before any references ot threadIdx etc (if present).
 * If there are no references to in-built variables (unlikely),
 *  __simt_threadIdx_init() may be omitted.
 * The builtin variables come into scope with this call; if the current
 * scope is finished, and a new one begun, a new call to
 * __simt_threadIDx_init() is required.
 *
 * Comments:
 *
 * target_simt_parallel_region() does job of "omp parallel"
 * (One could enter the OpenMP parallel region at the point
 * of the execution configuration, but that means __shared__
 * declarations in kernel are impossible except via extern.)
 *
 * target_simt_loop() does job of "omp for"
 *
 */


__target_entry__ void kernel_trial(int ndata, double * data) {

  int index;                     /* Problem: shared in OpenMP implementation
				  * but private in CUDA; programmer error. */
  __shared__ int updates[32];    /* OK; shared in device */

  assert(ndata % SIMDVL == 0);

  __target_simt_parallel_region() {

    /* Threads are now gauranteed to have started. */
    /* Declare thread-private variables if required; here... */

    __target_simt_threadIdx_init();
    int nupdate = 0;
    int ia;

    __target_simt_for(index, ndata, SIMDVL) {

      /* Worksharing  */

      printf("Thread %d of %d index %d\n", threadIdx.x, blockDim.x, index);

      int iv;    /* index for simd loop private */

      for (iv = 0; iv < SIMDVL; iv++) {
	/*printf("Update simd %d\n", index + iv);*/
	data[index + iv] *= 2.0;
	nupdate += 1;
      }
    }

    /* Reduction (all threads) Care with number of threads here
     * as I have declared only updates[32] */

    updates[threadIdx.x] = nupdate;
    printf("Updates by thread %i = %i\n", threadIdx.x, updates[threadIdx.x]);

    for (ia = blockDim.x/2; ia > 0; ia /= 2) {
      __syncthreads();
      if (threadIdx.x < ia) {
        updates[threadIdx.x] += updates[threadIdx.x + ia];
      }
    }
  }

  printf("blockIdx.x = 0 reduction: %i\n", updates[0]);

  return;
}
