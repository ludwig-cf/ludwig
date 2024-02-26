/*****************************************************************************
 *
 *  kernel_3d_v.c
 *
 *  Vectorised kernel (3d) helper.
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

#include <assert.h>

#include "kernel_3d_v.h"

/*****************************************************************************
 *
 *  kernel_3d_v
 *
 *****************************************************************************/

kernel_3d_v_t kernel_3d_v(cs_t * cs, cs_limits_t lim) {

  kernel_3d_v_t k3v = (kernel_3d_v_t) {0};
  assert(cs);

  cs_nhalo(cs, &k3v.nhalo);
  cs_nlocal(cs, k3v.nlocal);

  /* Limits as requested */
  k3v.lim = lim;

  /* The kernel must execute a whole number of vector blocks, which
   * means we have to include the nhalo regions in (y, z). Points
   * not belonging to the limits as requested must be masked out. */

  {
    cs_limits_t klim = {
      lim.imin, lim.imax,
      1 - k3v.nhalo, k3v.nlocal[Y] + k3v.nhalo,
      1 - k3v.nhalo, k3v.nlocal[Z] + k3v.nhalo
    }; 

    k3v.nklocal[X] = klim.imax - klim.imin + 1;
    k3v.nklocal[Y] = klim.jmax - klim.jmin + 1;
    k3v.nklocal[Z] = klim.kmax - klim.kmin + 1;

    /* Offset of first site must be start of a SIMD vector block */

    k3v.kindex0 = cs_index(cs, klim.imin, klim.jmin, klim.kmin);
    k3v.kindex0 = (k3v.kindex0/NSIMDVL)*NSIMDVL;

    /* Extent of the contiguous block ... */
    k3v.kiterations = k3v.nklocal[X]*k3v.nklocal[Y]*k3v.nklocal[Z];
  }

  return k3v;
}
