/****************************************************************************
 *
 *  target_cuda.h
 *
 ****************************************************************************/

#ifndef TARGET_CUDA_H
#define TARGET_CUDA_H

#define CUDA_MAX_THREADS_PER_BLOCK 128

#define __cuda_simt_for_all(index, ndata, stride) \
  index = (stride)*(blockIdx.x*blockDim.x + threadIdx.x);

#define __cuda_simt_for(index, ndata, stride) \
  __cuda_simt_for_all(index, ndata, stride) \
  if (index < (ndata))


#define __cuda_simt_parallel_for(index, ndata, stride) \
  __cuda_simt_for(index, ndata, stride)

#define __cuda_launch(kernel_function, nblocks, ntpb, ...) \
  kernel_function<<<nblocks, ntpb>>>(__VA_ARGS__)

#define __cuda_launch4s(kernel_function, nblocks, ntpb, shmem, stream, ...) \
  kernel_function<<<nblocks, ntpb, shmem, stream>>>(__VA_ARGS__)

#define __cuda_simd_for(iv, nsimdvl) \
  for (iv = 0; iv < (nsimdvl); iv++)

#endif
