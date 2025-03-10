/*****************************************************************************
 *
 *  target_x86.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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

typedef enum tdpDeviceP2PAttr {
  tdpDevP2PAttrPerformanceRank = 1,
  tdpDevP2PAttrAccessSupported = 2,
  tdpDevP2pAttrNativeAtomicSupported = 3,
  tdpDevP2PAttrArrayAccessSupported = 4
} tdpDeviceP2PAttr;

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

/* Device memory qualifiers / execution space qualifiers */

#define __host__
#define __global__
#define __shared__ static
#define __device__
#define __constant__

#if (__STDC__VERSION__ >= 19901)
  #define __forceinline__
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

/* Other vector types (as required) */

typedef struct tdp_double3_s double3;

struct tdp_double3_s {
  double x;
  double y;
  double z;
};

#ifdef _OPENMP
  /* These names are reserved and must be ... */
  #pragma omp threadprivate(gridDim, blockDim, threadIdx, blockIdx)
#endif

typedef enum tdpError tdpError_t;     /* an enum type */
typedef int  tdpStream_t;             /* an opaque handle */

/* Incomplete. */
struct tdpDeviceProp {
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  char name[256];
};

/* Graph API and related ... */

typedef void * tdpArray_t;     /* An general type */

typedef int * tdpGraph_t;      /* an opaque type (actually CUgraph_st) */
typedef int * tdpGraphExec_t;  /* ditto (actually CUgraphExec_st) */
typedef int * tdpGraphNode_t;  /* ditto (actually CUgraphNode_st) */

typedef struct tdpKernelNodeParams_s {
  dim3 blockDim;
  void * extra;
  void * func;
  dim3 gridDim;
  void ** kernelParams;
  unsigned int sharedMemBytes;
} tdpKernelNodeParams;

struct tdpExtent {
  size_t depth;
  size_t height;
  size_t width;
};

struct tdpPos {
  size_t x;
  size_t y;
  size_t z;
};

struct tdpPitchedPtr {
  size_t pitch;
  void * ptr;
  size_t xsize;
  size_t ysize;
};

typedef struct tdpMemcpy3DParms_s {
  tdpArray_t           dstArray;
  struct tdpPos        dstPos;
  struct tdpPitchedPtr dstPtr;
  struct tdpExtent     extent;
  tdpMemcpyKind        kind;
  tdpArray_t           srcArray;
  struct tdpPos        srcPos;
  struct tdpPitchedPtr srcPtr;
} tdpMemcpy3DParms;

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



/* Macros */

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
#define TARGET_PAD                     8

#define __syncthreads() _Pragma("omp barrier")
#define __threadfence() /* only __syncthreads() is a barrier */

/* Kernel launch is a __VA_ARGS__ macro, thus: */

#define tdpLaunchKernel(kernel, nblocks, nthreads, shmem, stream, ...) \
  _Pragma("omp parallel")					       \
  {								       \
    tdp_x86_prelaunch(nblocks, nthreads);			       \
    for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) {       \
      kernel(__VA_ARGS__);					       \
    }								       \
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
#define TARGET_PAD                   1
#define omp_get_num_threads() 1
#define omp_get_thread_num()  0
#define omp_get_max_threads() 1
#define omp_set_num_threads(n)
#define __syncthreads()
#define __threadfence()

/* NULL implementation */

/* Kernel launch is a __VA_ARGS__ macro, thus: */
#define tdpLaunchKernel(kernel, nblocks, nthreads, shmem, stream, ...) \
  tdp_x86_prelaunch(nblocks, nthreads);				       \
  for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) {     \
    kernel(__VA_ARGS__);					       \
  }                                                                    \
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

/* For "critical section" it's handy to use atomicCAS() and atomicExch()
 * in place (together with __threadfence()); until some better mechanism
 * is available */

#define atomicCAS(address, old, new) (old)
#define atomicExch(address, val)

#endif
