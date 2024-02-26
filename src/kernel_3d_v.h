/*****************************************************************************
 *
 *  kernel_3d_v.h
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

#ifndef LUDWIG_KERNEL_3D_V_H
#define LUDWIG_KERNEL_3D_V_H

#include "coords.h"
#include "cs_limits.h"
#include "memory.h"

typedef struct kernel_3d_v_s kernel_3d_v_t;

struct kernel_3d_v_s {
  int nhalo;       /* physical system - number of halo sites */
  int nlocal[3];   /* local system extent */

  int kindex0;     /* first index for kernel executtion */
  int kiterations; /* Number of iterations required for kernel (1d) */

  int nklocal[3];  /* local kernel extent */
  cs_limits_t lim; /* coordinate limits of the kernel (inclusive) */
};

kernel_3d_v_t kernel_3d_v(cs_t * cs, cs_limits_t lim);

/*****************************************************************************
 *
 *  __host__ __device__ static inline functions
 *
 *****************************************************************************/

#include <assert.h>

/*****************************************************************************
 *
 *  kernel_3d_v_coords
 *
 *****************************************************************************/

__host__ __device__ static inline void kernel_3d_v_coords(const kernel_3d_v_t * k3v,
					    int kindex0,
					    int ic[NSIMDVL],
					    int jc[NSIMDVL],
					    int kc[NSIMDVL]) {
  int iv;
  int index;
  int xs;
  int * __restrict__ icv = ic;
  int * __restrict__ jcv = jc;
  int * __restrict__ kcv = kc;

  assert(k3v);
  xs = k3v->nklocal[Y]*k3v->nklocal[Z];

  for_simd_v(iv, NSIMDVL) {
    index = k3v->kindex0 + kindex0 + iv;

    icv[iv] = index/xs;
    jcv[iv] = (index - icv[iv]*xs)/k3v->nklocal[Z];
    kcv[iv] = index - icv[iv]*xs - jcv[iv]*k3v->nklocal[Z];
  }

  for_simd_v(iv, NSIMDVL) {
    icv[iv] = icv[iv] - (k3v->nhalo - 1);
    jcv[iv] = jcv[iv] - (k3v->nhalo - 1);
    kcv[iv] = kcv[iv] - (k3v->nhalo - 1);

    assert(1 - k3v->nhalo <= icv[iv]);
    assert(1 - k3v->nhalo <= jcv[iv]);
    assert(1 - k3v->nhalo <= kcv[iv]);
    assert(icv[iv] <= k3v->nlocal[X] + k3v->nhalo);
    assert(jcv[iv] <= k3v->nlocal[Y] + k3v->nhalo);
    assert(kcv[iv] <= k3v->nlocal[Z] + k3v->nhalo);
  }

  return;
}

/*****************************************************************************
 *
 *  kernel_3d_v_mask
 *
 *****************************************************************************/

__host__ __device__ static inline void kernel_3d_v_mask(const kernel_3d_v_t * k3v,
					  int ic[NSIMDVL],
					  int jc[NSIMDVL],
					  int kc[NSIMDVL],
					  int mask[NSIMDVL]) {
  int iv;
  int * __restrict__ icv = ic;
  int * __restrict__ jcv = jc;
  int * __restrict__ kcv = kc;
  int * __restrict__ maskv = mask;

  for_simd_v(iv, NSIMDVL) {
    maskv[iv] = 1;
  }

  for_simd_v(iv, NSIMDVL) {
    if (icv[iv] < k3v->lim.imin || icv[iv] > k3v->lim.imax ||
	jcv[iv] < k3v->lim.jmin || jcv[iv] > k3v->lim.jmax ||
	kcv[iv] < k3v->lim.kmin || kcv[iv] > k3v->lim.kmax) {
      maskv[iv] = 0;
    }
  }

  return;
}

/*****************************************************************************
 *
 *  kernel_3d_v_cs_index
 *
 *****************************************************************************/

__host__ __device__ static inline void kernel_3d_v_cs_index(const kernel_3d_v_t * k3v,
					      const int ic[NSIMDVL],
					      const int jc[NSIMDVL],
					      const int kc[NSIMDVL],
					      int index[NSIMDVL]) {
  int iv;
  int nh;
  int xstr, ystr;
  const int * __restrict__ icv = ic;
  const int * __restrict__ jcv = jc;
  const int * __restrict__ kcv = kc;

  assert(k3v);

  nh   = k3v->nhalo;
  ystr = k3v->nlocal[Z] + 2*nh;
  xstr = ystr*(k3v->nlocal[Y] + 2*nh);

  for_simd_v(iv, NSIMDVL) {
    index[iv] = xstr*(nh + icv[iv] - 1) + ystr*(nh + jcv[iv] - 1) + nh + kcv[iv] - 1; 
  }

  return;
}

#endif

