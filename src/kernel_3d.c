/*****************************************************************************
 *
 *  kernel_3d.c
 *
 *  Help for 3d kernel execution.
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

#include "kernel_3d.h"

/*****************************************************************************
 *
 *  kernel_3d
 *
 *****************************************************************************/

kernel_3d_t kernel_3d(cs_t * cs, cs_limits_t lim) {

  kernel_3d_t k3d = (kernel_3d_t) {0};

  assert(cs);

  cs_nhalo(cs,  &k3d.nhalo);
  cs_nlocal(cs,  k3d.nlocal);

  /* limits of the kernel, and total number of iterations ... */
  k3d.lim        = lim;
  k3d.nklocal[X] = lim.imax - lim.imin + 1;
  k3d.nklocal[Y] = lim.jmax - lim.jmin + 1;
  k3d.nklocal[Z] = lim.kmax - lim.kmin + 1;

  k3d.kiterations = k3d.nklocal[X]*k3d.nklocal[Y]*k3d.nklocal[Z];

  /* First iteration of the kernel "lower left" */

  k3d.kindex0 = cs_index(cs, lim.imin, lim.jmin, lim.kmin);

  return k3d;
}

/****************************************************************************
 *
 *  kernel_3d_launch_param
 *
 *  A "class" method FIXME relocate this.
 *
 ****************************************************************************/

int kernel_3d_launch_param(int iterations, dim3 * nblk, dim3 * ntpb) {

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

/*****************************************************************************
 *
 *  kernel_3d_mask
 *
 *****************************************************************************/

__host__ __device__
int kernel_3d_mask(const kernel_3d_t * k3d, int ic, int jc, int kc) {

  if (ic < k3d->lim.imin || ic > k3d->lim.imax ||
      jc < k3d->lim.jmin || jc > k3d->lim.jmax ||
      kc < k3d->lim.kmin || kc > k3d->lim.kmax) return 0;

  return 1;
}
