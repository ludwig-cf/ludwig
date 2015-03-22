/*****************************************************************************
 *
 *  target_x86.h
 *
 *  Low level interface for targetDP to allow host executation either
 *  in serial or via OpenMP.
 *
 *****************************************************************************/

#ifndef TARGET_X86_H
#define TARGET_X86_H


#ifdef _OPENMP

  /* Have OpenMP */

  #include <omp.h>

  /* Some additional definitions provide a level of abstraction
   * and prevent  "unrecognised pragma" warnings at compile time
   * when there is no OpenMP .
   *
   * This also provides a concise reference statement about which
   * features of OpenMP are to be supported.
   */

  #define __x86_parallel_region()  _Pragma("omp parallel")
  #define __x86_for()              _Pragma("omp for nowait")
  #define __x86_parallel_for()     _Pragma("omp parallel for nowait")
  #define __x86_barrier()          _Pragma("omp barrier")

  #define __x86_get_num_threads()  omp_get_num_threads()
  #define __x86_get_thread_num()   omp_get_thread_num()
  #define __x86_get_max_threads()  omp_get_max_threads()
  #define __x86_set_num_threads(n) omp_set_num_threads(n)

#else

  /* NULL OpenMP implementation (macros for brevity here) */

  #define __x86_parallel_region()
  #define __x86_for()
  #define __x86_parallel_for()
  #define __x86_barrier()

  #define __x86_get_thread_num()   0
  #define __x86_get_num_threads()  1
  #define __x86_get_max_threads()  1
  #define __x86_set_num_threads(n)

#endif /* _OPENMP */


/* Device memory qualifiers / executation space qualifiers */

#define __global__
#define __shared__
#define __device__
#define __constant__ const

#if (__STDC__VERSION__ >= 19901)
  #define __forceinline__ inline
  #define __noinline__
#else
  #define __forceinline__
  #define __noinline__
#endif

/* Built-in variable implementation. */

typedef struct __uint3_s uint3;
typedef struct __dim3_s dim3;

struct __uint3_s {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct __dim3_s {
  int x;
  int y;
  int z;
};

/* Smuggle in gridDim and blockDim through static file scope object;
 * names must be reserved. */

static dim3 gridDim;
static dim3 blockDim;

/* ... execution configuration should  set the global
 * gridDim and blockDim so they are available in kernel, and
 * sets the number of threads which could be < omp_get_max_threads()
 */

static void __x86_prelaunch(dim3 nblocks, dim3 nthreads) {

  gridDim = nblocks;
  blockDim = nthreads;

  /* sanity checks on user settings here... */

  /* In case we request fewer threads than are available: */

  __x86_set_num_threads(blockDim.x*blockDim.y*blockDim.z);

  return;
}

static  void __x86_postlaunch(void) {

  int nthreads;

  /* Reset the default number of threads. */
  nthreads = __x86_get_max_threads();
  __x86_set_num_threads(nthreads);

  return;
}

#define __x86_launch(kernel_function, nblocks, nthreads, ...)		\
  __x86_prelaunch(nblocks, nthreads);					\
  kernel_function(__VA_ARGS__);						\
  __x86_postlaunch();

/* Utilities */

static uint3 __x86_builtin_threadIdx_init(void) {
  uint3 threads = {1, 1, 1};
  threads.x = __x86_get_thread_num();
  return threads;
}

static uint3 __x86_builtin_blockIdx_init(void) {
  uint3 blocks = {1, 1, 1};
  return blocks;
}

/* Within x86_simt_parallel_region(), provide access/initialisation */
/* "threadIdx_init" does both. Must be a macro expansiosn. */

#define __x86_simt_threadIdx_init()			\
  uint3 threadIdx;					\
  uint3 blockIdx;					\
  threadIdx = __x86_builtin_threadIdx_init();		\
  blockIdx  = __x86_builtin_blockIdx_init();

#define __x86_simt_parallel_region() __x86_parallel_region()

#define __x86_simt_for(index, ndata, simdvl)		\
  __x86_for()					\
  for (index = 0; index < (ndata); index += (simdvl))

#define __x86_simt_parallel_for(index, ndata, stride) \
  __x86_parallel_for() \
  for (index = 0; index < (ndata); index += (stride))

#endif
