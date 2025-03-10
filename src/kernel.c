/*****************************************************************************
 *
 *  kernel.c
 *
 *  Help for kernel execution.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "kernel.h"

/****************************************************************************
 *
 *  kernel_launch_param
 *
 ****************************************************************************/

int kernel_launch_param(int iterations, dim3 * nblk, dim3 * ntpb) {

  int ndevice = 0;

  assert(iterations > 0);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  ntpb->x = tdp_get_max_threads();
  ntpb->y = 1;
  ntpb->z = 1;

  if (ndevice == 0) {
    nblk->x = 1; /* Default to one block in OpenMP */
    nblk->y = 1;
    nblk->z = 1;
  }
  else {
    nblk->x = (iterations + ntpb->x - 1)/ntpb->x;
    nblk->y = 1;
    nblk->z = 1;
  }

  return 0;
}
