/****************************************************************************
 *
 *  target_cuda.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 * (c) 2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_TARGET_CUDA_H
#define LUDWIG_TARGET_CUDA_H

#include "cuda_runtime_api.h"

typedef cudaFuncCache tdpFuncCache;
typedef cudaMemcpyKind tdpMemcpyKind;
typedef cudaDeviceAttr tdpDeviceAttr;

#define tdpDevAttrManagedMemory cudaDevAttrManagedMemory

#define tdpSuccess cudaSuccess
#define tdpMemcpyHostToDevice cudaMemcpyHostToDevice
#define tdpMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define tdpMemcpyHostToHost cudaMemcpyHostToHost
#define tdpMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define tdpMemAttachHost   cudaMemAttachHost
#define tdpMemAttachGlobal cudaMemAttachGlobal

#define tdpHostAllocDefault cudaHostAllocDefault

typedef cudaStream_t tdpStream_t;
typedef cudaError_t tdpError_t;


#define TARGET_MAX_THREADS_PER_BLOCK 128

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

#define tdp_get_max_threads() TARGET_MAX_THREADS_PER_BLOCK

#endif
