/*****************************************************************************
 *
 *  kernel_3d.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_KERNEL_3D_H
#define LUDWIG_KERNEL_3D_H

#include "coords.h"
#include "cs_limits.h"
#include "memory.h"

typedef struct kernel_3d_s kernel_3d_t;

struct kernel_3d_s {
  int nhalo;       /* physical system - number of halo sites */
  int nlocal[3];   /* local system extent */

  int kindex0;     /* first index for kernel executtion */
  int kiterations; /* Number of iterations required for kernel (1d) */

  int nklocal[3];  /* local kernel extent */
  cs_limits_t lim; /* coordinate limits of the kernel (inclusive) */
};

kernel_3d_t kernel_3d(cs_t * cs, cs_limits_t lim);

/* A "class" method */

int kernel_3d_launch_param(int iterations, dim3 * nblk, dim3 * ntpb);

/*****************************************************************************
 *
 *  __host__ __device__ static inline functions
 *
 *****************************************************************************/

#include <assert.h>

/*****************************************************************************
 *
 *  kernel_3d_ic
 *
 *****************************************************************************/

__host__ __device__ static inline int kernel_3d_ic(const kernel_3d_t * k3d,
						   int kindex) {

  assert(k3d);

  return k3d->lim.imin + kindex/(k3d->nklocal[Y]*k3d->nklocal[Z]);
}

/*****************************************************************************
 *
 *  kernel_3d_jc
 *
 *****************************************************************************/

__host__ __device__ static inline int kernel_3d_jc(const kernel_3d_t * k3d,
						   int kindex) {

  assert(k3d);

  int xstr = k3d->nklocal[Y]*k3d->nklocal[Z];
  int   ic = kindex/xstr;
  int   jc = k3d->lim.jmin + (kindex - ic*xstr)/k3d->nklocal[Z];

  assert(1 - k3d->nhalo <= jc && jc <= k3d->nlocal[Y] + k3d->nhalo);

  return jc;
}

/*****************************************************************************
 *
 *  kernel_3d_kc
 *
 *****************************************************************************/

__host__ __device__ static inline int kernel_3d_kc(const kernel_3d_t * k3d,
						   int kindex) {
  assert(k3d);

  int ystr = k3d->nklocal[Z];
  int xstr = k3d->nklocal[Y]*k3d->nklocal[Z];

  int ic = kindex/xstr;
  int jc = (kindex - ic*xstr)/ystr;
  int kc = k3d->lim.kmin + kindex - ic*xstr - jc*ystr;

  assert(1 - k3d->nhalo <= kc && kc <= k3d->nlocal[Z] + k3d->nhalo);

  return kc;
}

/*****************************************************************************
 *
 *  kernel_3d_cs_index
 *
 *  In lieu of cs_index().
 *
 *****************************************************************************/

__host__ __device__
static inline int kernel_3d_cs_index(const kernel_3d_t * k3d,
				     int ic, int jc, int kc) {
  int nh = k3d->nhalo;

  int zstr = 1;
  int ystr = zstr*(k3d->nlocal[Z] + 2*nh);
  int xstr = ystr*(k3d->nlocal[Y] + 2*nh);

  return xstr*(nh + ic - 1) + ystr*(nh + jc - 1) + zstr*(nh + kc - 1); 
}

#endif

