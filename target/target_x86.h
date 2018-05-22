/*****************************************************************************
 *
 *  target_x86.h
 *
 *  Low level interface for targetDP to allow host executation either
 *  via OpenMP or serial execution.
 *
 *****************************************************************************/

#ifndef LUDWIG_TARGET_X86_H
#define LUDWIG_TARGET_X86_H

typedef enum tdpFuncCache_enum {
  tdpFuncCachePreferNone = 0,
  tdpFuncCachePreferShared = 1,
  tdpFuncCachePreferL1 = 2,
  tdpFuncCachePreferEqual = 3}
  tdpFuncCache;

typedef enum tdpMemcpyKind_enum {
  tdpMemcpyHostToHost = 0,
  tdpMemcpyHostToDevice = 1,
  tdpMemcpyDeviceToHost = 2,
  tdpMemcpyDeviceToDevice = 3,
  tdpMemcpyDefault = 4}
  tdpMemcpyKind;

/* Device attributes (potentially a lot of them) */

typedef enum tdpDeviceAttr_enum {
  tdpDevAttrMaxThreadsPerBlock = 1,
  tdpDevAttrMaxBlockDimX = 2,
  tdpDevAttrMaxBlockDimY = 3,
  tdpDevAttrMaxBlockDimZ = 4,
  tdpDevAttrMaxGridDimX = 5,
  tdpDevAttrMaxGridDimY = 6,
  tdpDevAttrMaxGridDimZ = 7,
  tdpDevAttrManagedMemory = 83
} tdpDeviceAttr;

/* tdpGetLastError() can return... */

enum tdpError {
  tdpSuccess = 0,
  tdpErrorMissingConfiguration = 1,
  tdpErrorMemoryAllocation = 2,
  tdpErrorInitializationError = 3,
  tdpErrorLaunchFailure = 4,
  tdpErrorLaunchTimeout = 6,
  tdpErrorLaunchOutOfResources = 7,
  tdpErrorInvalidDeviceFunction = 8,
  tdpErrorInvalidConfiguration = 9,
  tdpErrorInvalidDevice = 10,
  tdpErrorInvalidValue = 11,
  tdpErrorInvalidPitchValue = 12,
  tdpErrorInvalidSymbol = 13,
  tdpErrorUnmapBufferObjectFailed = 15,
  tdpErrorInvalidHostPointer = 16,
  tdpErrorInvalidDevicePointer = 17,
  tdpErrorInvalidTexture = 18,
  tdpErrorInvalidTextureBinding = 19,
  tdpErrorInvalidChannelDescriptor = 20,
  tdpErrorInvalidMemcpyDirection = 21,
  tdpErrorInvalidFilterSetting = 26,
  tdpErrorUnknown = 30,
  tdpErrorInvalidResourceHandle = 33,
  tdpErrorInsufficientDriver = 35,
  tdpErrorSetOnActiveProcess = 36,
  tdpErrorInvalidSurface = 37,
  tdpErrorNoDevice = 38,
  tdpErrorStartupFailure = 0x7f
};

#define tdpHostAllocDefault       0x00
#define tdpHostAllocMapped        0x02
#define tdpHostAllocPortable      0x01
#define tdpHostAllocWriteCombined 0x04

#define tdpMemAttachGlobal        0x01
#define tdpMemAttachHost          0x02
#define tdpMemAttachSingle        0x04

/* Device memory qualifiers / executation space qualifiers */

#define __host__
#define __global__
#define __shared__ static
#define __device__
#define __constant__

#if (__STDC__VERSION__ >= 19901)
  #define __forceinline__ inline
  #define __noinline__
#else
  #define __forceinline__
  #define __noinline__
#endif

/* Built-in variable implementation. */

typedef struct tdp_uint3_s uint3;
typedef struct tdp_dim3_s dim3;

struct tdp_uint3_s {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct tdp_dim3_s {
  int x;
  int y;
  int z;
};

extern dim3 gridDim;
extern dim3 blockDim;
extern dim3 threadIdx;
extern dim3 blockIdx;

#ifdef _OPENMP
  /* These names are reserved and must be ... */ 
  #pragma omp threadprivate(gridDim, blockDim, threadIdx, blockIdx)
#endif

typedef enum tdpError tdpError_t;     /* an enum type */
typedef int * tdpStream_t;            /* an opaque handle */

/* Incomplete. */
struct tdpDeviceProp {
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
};


#define tdpSymbol(x) &(x)
void  tdp_x86_prelaunch(dim3 nblocks, dim3 nthreads);
void  tdp_x86_postlaunch(void);

#ifdef _OPENMP

/* Help to expand OpenMP clauses which need to be retained as strings */
#define xstr(a) str(a)
#define str(a) #a

/* Have OpenMP */

#include <omp.h>
#define TARGET_MAX_THREADS_PER_BLOCK 256

#define __syncthreads() _Pragma("omp barrier")

/* Kernel launch is a __VA_ARGS__ macro, thus: */
#define tdpLaunchKernel(kernel, nblocks, nthreads, shmem, stream, ...) \
  _Pragma("omp parallel")					       \
  {								       \
    tdp_x86_prelaunch(nblocks, nthreads);			       \
    kernel(__VA_ARGS__);					       \
    tdp_x86_postlaunch();					       \
  }

  /* OpenMP work sharing */
  #define for_simt_parallel(index, ndata, stride)	\
  _Pragma("omp for nowait")				\
  for (index = 0; index < (ndata); index += (stride))

  /* SIMD safe loops */
  #define for_simd_v(iv, nsimdvl) \
  _Pragma("omp simd") \
  for (iv = 0; iv < (nsimdvl); ++iv)

  #define for_simd_v_reduction(iv, nsimdvl, clause) \
  _Pragma(xstr(omp simd reduction(clause)))	\
  for (iv = 0; iv < nsimdvl; ++iv)

#else /* Not OPENMP */

#define TARGET_MAX_THREADS_PER_BLOCK 1
#define omp_get_num_threads() 1
#define omp_get_thread_num()  0
#define omp_get_max_threads() 1
#define omp_set_num_threads(n)
#define __syncthreads()

/* NULL implementation */

/* Kernel launch is a __VA_ARGS__ macro, thus: */
#define tdpLaunchKernel(kernel, nblocks, nthreads, shmem, stream, ...) \
  tdp_x86_prelaunch(nblocks, nthreads);				       \
  kernel(__VA_ARGS__);						       \
  tdp_x86_postlaunch();

/* "Worksharing" is provided by a loop */
#define for_simt_parallel(index, ndata, stride)		\
  for (index = 0; index < (ndata); index += (stride))

/* Vectorised loops */
#define for_simd_v(iv, nsimdvl) for (iv = 0; iv < (nsimdvl); iv++)
#define for_simd_v_reduction(iv, nsimdvl, clause)	\
  for (iv = 0; iv < nsimdvl; iv++)

#endif /* _OPENMP */

#define tdp_get_max_threads() omp_get_max_threads()

#endif
