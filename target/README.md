# TargetDP

TargetDP (Target Data Parallel) provides a simple abstraction of
thread-level parallelism which allows code to be run on either
CPUs (OpenMP) or GPUs (CUDA/HIP). TargetDP was first written by
Alan Gray in 2013/2014 and this version has been maintained as
part of the Ludwig code.

The idea is to provide an API which apes CUDA/HIP, and has a small
number of additional features to allow a host implementation using
OpenMP. This allows a single source to be used on both CPU and GPU
"targets". Relevant source code should include the file:
```
#include "target.h"
```
The API replaces e.g., `cuda` by `tdp`, so `cudaMalloc()` becomes
`tdpMalloc()` and so on.

There are three implementations:
```
target_x86.h
target_cuda.h
target_hip.h
```
representing respectively: serial and OpenMP, CUDA, and HIP targets.
The x86 implementation is also referred to as the host implementation.

## Memory

It is assumed that address spaces are potentially separate, and that
device memory is allocated in a manner distinct from that on the host.
Copies of data may be required between the two spaces, and these copies
are explicit.

In the host implementation memory allocation via
```
tdpHostAlloc()
tdpMalloc()
tdpMallocManaged()
```
result in a C `malloc()` allocation, which should be released with
`tdpFree()`. A `tdpMamcpy()` will result in a copy from the source
to the destination for all values of the `tdpMemcpyKind` argument.


## Kernels

### Execution

The model is one of execution of kernels, which may reside on the
GPU or the CPU depending on the current targets. Within kernels,
execution involves one or more independent blocks each of which
execute a number of threads per block.

Kernels are introduced with the `__global__` execution space qualifier
and are launched on the device via `tdpLaunchKernel()`. The kernel
is executed by a number of blocks and threads set be the kernel
launch parameters.

### Example

A typical example of a kernel might be
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

There are three relevant macros which may occur in a kernel:

```
* for_simt_parallel(index, ndata, stride)
```
This provides worksharing of the iterations of a one-dimensional loop with
index `index` which varies in the range `0 <= index < ndata` and with
stride `stride`. The `stride` must be present and should be set to 1
if a unit stride is required. The macro guarantees that the correct number
of interactions are performed independent of the number of threads.

```
* for_sind_v(iv, nsimdvl)
```
This macro is intended to introduce short inner loops with length of the
SIMD vector length `nsimdvl`. The loop index is `iv` which will run
`0 <= iv < nsindvl`.

```
* for_simd_v_reduction(iv, nsimdvl, clause)
```
Allows the addition of an OpenMP-style reduction clause if required.

For GPU implementations, `nsimdvl` should be set to unity.

### Kernel launch

A kernel is launched by specifying a number of blocks, and a number
of threads per block. At present, only the x-dimension is used in
the description; other dimensions should be set to unity.

```
dim3 nblk;    /* Number of blocks */
dim3 ntpb;    /* Number of threads per block */

ntpb.x = tdp_get_max_threads(); ntpb.y = 1; ntpb.z = 1;
nblk.x = (NARRAY + ntpb.x - 1)/ntpb.x; nblk.y = 1; nblk.z = 1;

tdpLaunchKernel(kernel_name, nblk, ntpb, 0, 0, ...)
```

## CUDA-like host-side API

Host execution space routines follow CUDA naming with the `cuda` replaced
be `tdp` (in the same fashion as HIP). A small subset of the CUDA API
is supported.

## Additional device-side TargetDP API

As targetDP is a C interface, convenience routines are provided for
a number of common operations.

### Type-specfic atomic operations

The following functions provide type-specfic atomic operations
which read the old value at the address provided by the first
argument, add the value of the second argument `val` and
write back the result of the relevant operation to the address.
The old value is returned.
```
__device__ int atomicAdd(int * sum, int val);
__device__ int atomicMax(int * maxval, int val);
__device__ int atomicMin(int * minval, int val);
__device__ double atomicAdd(double * sum, double val);
__device__ double atomicMax(double * maxval, double val);
__device__ double atomicMin(double * minval, double val);
```

## Comments on OpenMP target

If the target is OpenMP, then it is assumed that host and device memory
spaces are the same. All copies via `tdpMemcpy()` and related routines
are therefore host to hoast copies.

Threads are started by the kernel launch, and
Kernel execution should be regarded as being a single block running
at most `OMP_NUM_THREADS` threads.

## Comments on CUDA target

If the target is an NVIDIA GPU, then the model should be familiar to
someone having experience of CUDA.

## Comments on HIP target

The HIP implementation is similar to the CUDA implementation.
