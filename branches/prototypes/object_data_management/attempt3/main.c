/* OpenMP trial */

#include <assert.h>
#include <stdio.h>

#include "targetDP.h"
#define NDATA 8
#define SIMDVL 2


#ifdef _OPENMP
  /* have OpenMP */
  #include <omp.h>
#else
  /* NULL OpenMP implmentation */
  #define omp_get_thread_num()  0
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
  #define omp_set_num_threads(nthread)
#endif


#ifdef __NVCC__
/* target defs, including real target_simd_loop() */
#define target_simt_parallel_region()
#define target_simt_loop(index, ndata, simdvl) \
  index = SIMDVL*(blockIdx.x*blockDim.x + threadIdx.x);

#define __simt_builtin_init()
#define __simt_threadIdx_init()
#define __simt_blockIdx_init()

#define execution_configuration(...) target_launch(__VA_ARGS__)

#else

/* Device memory qualifier */

#define __shared__

/* built-in variable implmentation. */

typedef struct uint3_s uint3;
typedef uint3 uint3_t;

struct uint3_s {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

typedef struct dim3_s dim3;
typedef dim3 dim3_t;

struct dim3_s {
  int x;
  int y;
  int z;
};

/* Smuggle in gridDim and blockDim through static file scope object;
 * probably ok as names must be resevered. */

static dim3_t gridDim;
static dim3_t blockDim;

/* ... executation configuration should  set the global
 * gridDim and blockDim so they are available in kernel, and
 * sets the number of threads which could be < omp_get_max_threads()
 * Additional sanity checks could be envisaged.
 */

void __simt_target_prelaunch(dim3_t nblocks, dim3_t nthreads) {
  gridDim = nblocks;
  blockDim = nthreads;
  /* sanity checks on user settings */
  omp_set_num_threads(blockDim.x*blockDim.y*blockDim.z);

  return;
}

void __simt_target_postlaunch(void) {
  omp_set_num_threads(omp_get_max_threads());
  return;
}

#define execution_configuration(kernel_trial, nblocks, nthreads, ...) \
  __simt_target_prelaunch(nblocks, nthreads); \
  target_launch(kernel_trial, nblocks, nthreads, __VA_ARGS__); \
  __simt_target_postlaunch();

/* Synchronisation */

#define __syncthreads() _Pragma("omp barrier")

/* Utilities */

uint3_t __simd_builtin_threadIdx_init(void) {
  uint3_t threads = {1, 1, 1};
  threads.x = omp_get_thread_num();
  return threads;
}

uint3_t __simd_builtin_blockIdx_init(void) {
  uint3_t blocks = {1, 1, 1};
  return blocks;
}

int __simt_default_threads_per_block() {

  int ntpb = 1;

#ifdef _OPENMP
  ntpb = omp_get_max_threads();
#elif __NVCC__
  ntpb = 32;  /* or whatever for CUDA */
#endif

  return ntpb;
}

/* Within target_simd_parallel_region(), provide access/initialisation */
/* If don't need both, use a single version to prevent unused variable
 * warnings */

#define __simt_threadIdx_init() \
  uint3_t threadIdx = __simd_builtin_threadIdx_init();

#define __simt_blockIdx_init() \
  uint3_t blockIdx  = __simd_builtin_blockIdx_init();

#define __simt_builtin_init() \
  __simt_threadIdx_init(); \
  __simt_blockIdx_init();

/* data parallel OpenMP */

#define target_simt_parallel_region() _Pragma("omp parallel")

#define target_simt_loop(index, ndata, simdvl) \
  _Pragma("omp for nowait") \
  for (index = 0; index < ndata; index += simdvl)


#endif

__target_entry__ void kernel_trial(int ndata, double * data);

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

  targetMalloc((void **) &data, NDATA*sizeof(double));
  if (target_is_host()) {
    data = hdata;
  }
  else {
    copyToTarget(data, hdata, NDATA*sizeof(double));
  }

  /* Block size.
   * For OpenMP, it seems reasonable to set the number of blocks to
   *   {1, 1, 1}
   * in (almost all) cases and rely on ndata to express the extent of
   * the parallel loop in the kernel. The number of threads is at most
   *   omp_get_max_threads()
   * and we need to use as many threads as possible.
   *
   * For CUDA, the block size is dependent on problem size and threads
   * per block requested. Any given application may want a helper
   * function to work this out. */

  nblocks.x = 1; nblocks.y = 1; nblocks.z = 1;

  nthreads.x = __simt_default_threads_per_block();
  nthreads.y = 1;
  nthreads.z = 1;

  execution_configuration(kernel_trial, nblocks, nthreads, ndata, data);

  syncTarget();

  if (target_is_host()) {
    ;
  }
  else {
    copyFromTarget(hdata, data, NDATA*sizeof(double));
  }

  printf("ndara is %d\n", ndata);
  for (n = 0; n < ndata; n++) {
    printf("data[%2i] = %5.1f\n", n, hdata[n]);
  }

  return 0;
}

/* Additional restrictions:
 *
 * __shared__ declarations must precede target_simt_region()
 *   (CUDA allows them in any scoping unit in kernel) so that
 * OpenMP sees shared memory
 *
 * extern __shared__ * types with dynamic allocation could
 * be accommodated with the exectuation configuration.
 *
 * In host code thread private variables must come after
 * target_simt_region() (cf. whole function scope is private in CUDA).
 *
 * __simt_builtin_init() must occur inside target_simt_parallel_region()
 * and before any references ot threadIdx etc (if present).
 * If there are no references to in-built variables (unlikely),
 *  __simt_builtin_init() may be omitted.
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
 * In the same vein as "omp parallel for" one could combine
 * two in a convenience version:
 *
 *   target_simt_parallel_loop(index, ndata, SIMDVL)
 *
 * If no SIMD loop is required I suggest having, e.g.,
 *
 *   target_simt_parallel_loop(index, ndata, IGNORE_SIMDVL)
 *
 * with IGNORE_SIMDVL = 1
 */


__target_entry__ void kernel_trial(int ndata, double * data) {

  int index;                     /* Problem: shared in host implementation */
  __shared__ int updates[32];    /* OK; shared in device */

  assert(ndata % SIMDVL == 0);

  target_simt_parallel_region() {

    /* Declare thread-private variables if required; here... */

    __simt_threadIdx_init();
    int nupdate = 0;
    int ia;

    target_simt_loop(index, ndata, SIMDVL) {

      /* ... or here  */

      printf("Thread %d of %d index %d\n", threadIdx.x, blockDim.x, index);

      if (index < ndata) {

	int iv;    /* index for simd loop private */

	for (iv = 0; iv < SIMDVL; iv++) {
	  /*printf("Update simd %d\n", index + iv);*/
	  data[index + iv] *= 2.0;
	  nupdate += 1;
	}
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
