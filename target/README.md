# TargetDP

TargetDP (Target Data Parallel) provides a simple abstraction of
thread-level parallelism which allows code to be run on either
CPUs (OpenMP) or GPUs (CUDA/HIP). TargetDP was first written by
ALan Gray in 2013/2014 and this version has been maintained as
part of the Ludwig code.


Note that the HIP implementation is under development.

## Kernels

Kernels are introduced with the `__global__` execution space qualifier
and are launched on the device via `tdpLaunchKernel()`. The kernel
is executed by a number of blocks and threads set be the kernel
launch parameters. A typical example of a kernel might be
```

#define NARRAY 132 /* Number of elements in array n[] */

__global__ void kerneltest1(int * n) {

  int p;

  for_simt_parallel(p, NARRAY, 1) {
    n[p] = 2*n[p];
  }
  return;
}
```
Here, TargetDP arranges that the code is executed for all values
`0 <= p < NARRAY`.

## CUDA-like host-side API


## Additional device-side TargetDP API
