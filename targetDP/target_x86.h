/*****************************************************************************
 *
 *  target_x86.h
 *
 *  Low level interface for targetDP to allow host executation either
 *  via OpenMP or serial execution.
 *
 *****************************************************************************/

#ifndef TARGET_X86_H
#define TARGET_X86_H

#ifdef _OPENMP

  /* Have OpenMP */

  #include <omp.h>
  #define X86_MAX_THREADS_PER_BLOCK 256

  /* Some additional definitions provide a level of abstraction
   * and prevent  "unrecognised pragma" warnings at compile time
   * when there is no OpenMP .
   *
   * This also provides a concise reference statement about which
   * features of OpenMP are to be supported.
   */

  #define __host_parallel_region()  _Pragma("omp parallel")
  #define __host_for()              _Pragma("omp for nowait")
  #define __host_parallel_for()     _Pragma("omp parallel for")
  #define __host_barrier()          _Pragma("omp barrier")

  #define __host_get_num_threads()  omp_get_num_threads()
  #define __host_get_thread_num()   omp_get_thread_num()
  #define __host_get_max_threads()  omp_get_max_threads()
  #define __host_set_num_threads(n) omp_set_num_threads(n)

  /* OpenMP work sharing */
  #define __host_simt_for(index, ndata, stride)		\
  __host_for()						\
  for (index = 0; index < (ndata); index += (stride))

  /* SIMD safe loops */
  #define __host_simd_for(iv, nsimdvl) \
  _Pragma("omp simd") \
  for (iv = 0; iv < (nsimdvl); ++iv)


#else

  /* NULL OpenMP implementation (macros for brevity here) */

  #define X86_MAX_THREADS_PER_BLOCK 1
  #define __host_parallel_region()
  #define __host_for()
  #define __host_parallel_for()
  #define __host_barrier()

  #define __host_get_thread_num()   0
  #define __host_get_num_threads()  1
  #define __host_get_max_threads()  1
  #define __host_set_num_threads(n)

  /* "Worksharing" is provided by a loop */
  #define __host_simt_for(index, ndata, stride)	\
  for (index = 0; index < (ndata); index += (stride))

  #define __host_simd_for(iv, nsimdvl) for (iv = 0; iv < (nsimdvl); ++iv)

#endif /* _OPENMP */

#endif
