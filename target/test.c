/*****************************************************************************
 *
 *  test.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "target.h"

#define NARRAY 132

__host__ int test0(void) {

  int mydevice;
  struct tdpDeviceProp prop;
  tdpError_t ifail;

  tdpGetDevice(&mydevice);
  ifail = tdpGetDeviceProperties(&prop, mydevice);
  if (ifail != tdpSuccess) printf("FAIL!\n");

  printf("maxThreadsPerBlock %d\n", prop.maxThreadsPerBlock);
  printf("maxThreadsDim[0]   %d\n", prop.maxThreadsDim[0]);
  printf("maxThreadsDim[1]   %d\n", prop.maxThreadsDim[1]);
  printf("maxThreadsDim[2]   %d\n", prop.maxThreadsDim[2]);

  return 0;
}

/* Test 1: scale by constant */

__global__ void kerneltest1(int * n) {

  int p;

  for_simt_parallel(p, NARRAY, 1) {
    n[p] = 2*n[p];
  }

  return;
}

int main(int argc, char * argv[]) {

  dim3 nblk, ntpb;
  int p;
  int bufsz;
  int * n_h; /* host */
  int * n_d; /* device */

  test0();

  bufsz = NARRAY*sizeof(int);

  n_h = (int *) calloc(NARRAY, sizeof(int));
  tdpAssert(tdpMalloc((void **) &n_d, bufsz));

  for (p = 0; p < NARRAY; p++) {
    n_h[p] = p;
  }

  tdpAssert(tdpMemcpy(n_d, n_h, bufsz, tdpMemcpyHostToDevice));

  ntpb.x = tdp_get_max_threads(); ntpb.y = 1; ntpb.z = 1;
  nblk.x = (NARRAY + ntpb.x - 1)/ntpb.x; nblk.y = 1; nblk.z = 1;

  tdpLaunchKernel(kerneltest1, nblk, ntpb, 0, 0, n_d);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  tdpAssert(tdpMemcpy(n_h, n_d, bufsz, tdpMemcpyDeviceToHost));

  for (p = 0; p < NARRAY; p++) {
    assert(n_h[p] == 2*p);
  }

  return 0;
}
