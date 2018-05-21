/****************************************************************************
 *
 *  target_cuda.h
 *
 ****************************************************************************/

#ifndef LUDWIG_TARGET_CUDA_H
#define LUDWIG_TARGET_CUDA_H

#include "cuda_runtime_api.h"

#define MAX_THREADS_PER_BLOCK 128

/* Macros for calls involing device symbols */

#define tdpSymbol(x) x
#define tdpGetSymbolAddress(dst, symbol)		\
        tdpAssert(cudaGetSymbolAddress(dst, symbol))
#define tdpMemcpyToSymbol(symbol, src, count, offset, kind)	\
        tdpAssert(cudaMemcpyToSymbol(symbol, src, count, offset, kind))
#define tdpMemcpyFromSymbol(dst, symbol, count, offset, kind) \
        tdpAssert(cudaMemcpyFromSymbol(dst, symbol, count, offset, kind))

#define	tdpLaunchKernel(kernel, nblocks, nthreads, shmem, stream, ...) \
  kernel<<<nblocks, nthreads, shmem, stream>>>(__VA_ARGS__);

#define for_simt_parallel(index, ndata, stride) \
  index = (stride)*(blockIdx.x*blockDim.x + threadIdx.x); \
  if (index < (ndata))

#define for_simd_v(iv, nsimdvl) \
  for (iv = 0; iv < (nsimdvl); iv++)

#define for_simd_v_reduction(iv, nsimdvl, clause) \
  for (iv = 0; iv < (nsimdvl); iv++)

#endif
