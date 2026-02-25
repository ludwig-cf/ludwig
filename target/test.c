/*****************************************************************************
 *
 *  test.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2026 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "target.h"

#define NARRAY 132

__host__ int test0(void) {

  int mydevice;
  int ndevice = 0;
  struct tdpDeviceProp prop;
  tdpError_t ifail;

  ifail = tdpGetDeviceCount(&ndevice);
  if (ifail == tdpSuccess) {
    printf("Number of devices avail: %d\n", ndevice);
  }
  else {
    printf("No GPU device detected\n");
  }

  tdpAssert( tdpGetDevice(&mydevice) );
  ifail = tdpGetDeviceProperties(&prop, mydevice);
  if (ifail != tdpSuccess) printf("FAIL!\n");

  printf("Device id                 %d\n",  mydevice);
  printf("Device name               %s\n",  prop.name);
  printf("maxThreadsPerBlock        %d\n",  prop.maxThreadsPerBlock);
  printf("maxThreadsDim[0]          %d\n",  prop.maxThreadsDim[0]);
  printf("maxThreadsDim[1]          %d\n",  prop.maxThreadsDim[1]);
  printf("maxThreadsDim[2]          %d\n",  prop.maxThreadsDim[2]);
  printf("totalGlobalMem (bytes)    %ld\n", prop.totalGlobalMem);

  return 0;
}

/* Count the number of blocks */

__global__ void kerneltest1() {

  if (threadIdx.x == 0) {
    printf("blockidx.x %2d griddim %2d %2d %2d\n", blockIdx.x,
	   gridDim.x, gridDim.y, gridDim.z);
  }

  return;
}

int test1(void) {

  dim3 ntpb = {1, 1, 1};
  dim3 nblk = {4, 1, 1};

  tdpLaunchKernel(kerneltest1, nblk, ntpb, 0, 0);
  tdpAssert( tdpStreamSynchronize(0) );

  return 0;
}

/* Test 1: scale by constant */

__global__ void kerneltest2(int * n) {

  int p = 0;

  for_simt_parallel(p, NARRAY, 1) {
    n[p] = 2*n[p];
  }

  return;
}

/*****************************************************************************
 *
 *  test_atomicAdd
 *
 *****************************************************************************/

__device__ void test_atomicAdd(void) {

  /* int */
  {
    int val = 2;
    int sum = 1;
    int old = atomicAdd(&sum, val);

    assert(old == 1);
    assert(sum == 3);
  }

  /* double */
  {
    double val = 1.0;
    double sum = 2.0;
    double old = atomicAdd(&sum, val);

    assert(fabs(old - 2.0) < DBL_EPSILON);
    assert(fabs(sum - 3.0) < DBL_EPSILON);
  }

  return;
}

/*****************************************************************************
 *
 *  test_atomicMax
 *
 *****************************************************************************/

__device__ void test_atomicMax(void) {

  /* int */
  {
    int a = -2;
    int b = -1;
    int c = atomicMax(&a, b);

    assert(c == -2);
    assert(a == -1);
  }

  /* double */
  {
    double a = 1.0;
    double b = 2.0;
    double c = atomicMax(&a, b);

    assert(fabs(c - 1.0) < DBL_EPSILON);
    assert(fabs(a - 2.0) < DBL_EPSILON);
  }

  return;
}

/*****************************************************************************
 *
 *  test_atomicMin
 *
 *****************************************************************************/

__device__ void test_atomicMin(void) {

  /* int */
  {
    int a = -1;
    int b = -2;
    int c = atomicMin(&a, b);

    assert(c == -1);
    assert(a == -2);
  }

  /* double */
  {
    double a = 2.0;
    double b = 1.0;
    double c = atomicMin(&a, b);

    assert(fabs(c - 2.0) < DBL_EPSILON);
    assert(fabs(a - 1.0) < DBL_EPSILON);
  }

  return;
}

/*****************************************************************************
 *
 *  test_kernel
 *
 *****************************************************************************/

__global__ void test_kernel(void) {

  test_atomicAdd();
  test_atomicMax();
  test_atomicMin();

  return;
}


int main(int argc, char * argv[]) {

  int ndevice = 0;

  dim3 nblk, ntpb;
  int p;
  int bufsz;
  int * n_h; /* host */
  int * n_d; /* device */

  if (argc > 1) printf("%s: takes no arguments\n", argv[0]);

  test0();
  test1();

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  bufsz = NARRAY*sizeof(int);

  n_h = (int *) calloc(NARRAY, sizeof(int));
  tdpAssert(tdpMalloc((void **) &n_d, bufsz));

  for (p = 0; p < NARRAY; p++) {
    n_h[p] = p;
  }

  tdpAssert(tdpMemcpy(n_d, n_h, bufsz, tdpMemcpyHostToDevice));

  ntpb.x = tdp_get_max_threads(); ntpb.y = 1; ntpb.z = 1;
  nblk.x = (NARRAY + ntpb.x - 1)/ntpb.x; nblk.y = 1; nblk.z = 1;

  if (ndevice == 0) nblk.x = 1; /* OpenMP */

  tdpLaunchKernel(kerneltest2, nblk, ntpb, 0, 0, n_d);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  tdpAssert(tdpMemcpy(n_h, n_d, bufsz, tdpMemcpyDeviceToHost));

  for (p = 0; p < NARRAY; p++) {
    if (n_h[p] != 2*p) printf("Wrong %3d %3d\n", p, n_h[p]);
  }

  /* Serial tests */
  {
    dim3 nsblk = {1, 1, 1};
    dim3 nstpb = {1, 1, 1};

    tdpLaunchKernel(test_kernel, nsblk, nstpb, 0, 0);
    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}
