/****************************************************************************
 *
 *  advection_bcs.c
 *
 *  Advection boundary conditions.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2016 The University of Edinburgh
*
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "wall.h"
#include "coords.h"
#include "kernel.h"
#include "leesedwards.h"
#include "advection_s.h"
#include "psi_gradients.h"
#include "map_s.h"
#include "timer.h"
#include "advection_bcs.h"

__global__
void advection_bcs_no_flux_kernel_v(kernel_ctxt_t * ktx, advflux_t * flux,
				    map_t * map);

/*****************************************************************************
 *
 *  advection_bcs_no_normal_flux
 *
 *  Kernel driver for no-flux boundary conditions.
 *
 *****************************************************************************/

__host__
int advection_bcs_no_normal_flux(int nf, advflux_t * flux, map_t * map) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(flux);
  assert(map);

  coords_nlocal(nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(ADVECTION_BCS_KERNEL);

  __host_launch(advection_bcs_no_flux_kernel_v, nblk, ntpb, ctxt->target,
		flux->target, map->target);

  targetSynchronize();

  TIMER_stop(ADVECTION_BCS_KERNEL);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  advection_bcs_no_flux_kernel_v
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

__global__
void advection_bcs_no_flux_kernel_v(kernel_ctxt_t * ktx,
				    advflux_t * flux,
				    map_t * map) {
  int kindex;
  __shared__ int kiter;

  kiter = kernel_vector_iterations(ktx);

  __target_simt_parallel_for(kindex, kiter, NSIMDVL) {

    int n;
    int iv;
    int index0;
    int ic[NSIMDVL], jc[NSIMDVL], kc[NSIMDVL];
    int ix[NSIMDVL];
    int index[NSIMDVL];
    int maskv[NSIMDVL];     /* = 0 if vector entry not within kernel limits */

    double mask[NSIMDVL];   /* mask for current (ic, jc, kc) */
    double maskw[NSIMDVL];  /* mask for flux->fw */
    double maske[NSIMDVL];  /* mask for flux->fe */
    double masky[NSIMDVL];  /* mask for flux->fy */
    double maskz[NSIMDVL];  /* mask for flux->fz */

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_mask_v(ktx, ic, jc, kc, maskv);
      
    kernel_coords_index_v(ktx, ic, jc, kc, index);
    __targetILP__(iv) mask[iv] = (map->status[index[iv]] == MAP_FLUID);  

    __targetILP__(iv) ix[iv] = ic[iv] - maskv[iv];
    kernel_coords_index_v(ktx, ix, jc, kc, index);
    __targetILP__(iv) maskw[iv] = (map->status[index[iv]] == MAP_FLUID);    

    __targetILP__(iv) ix[iv] = ic[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ix, jc, kc, index);
    __targetILP__(iv) maske[iv] = (map->status[index[iv]] == MAP_FLUID);

    __targetILP__(iv) ix[iv] = jc[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ic, ix, kc, index);
    __targetILP__(iv) masky[iv] = (map->status[index[iv]] == MAP_FLUID);

    __targetILP__(iv) ix[iv] = kc[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ic, jc, ix, index);
    __targetILP__(iv) maskz[iv] = (map->status[index[iv]] == MAP_FLUID);

    index0 = kernel_baseindex(ktx, kindex);

    for (n = 0;  n < flux->nf; n++) {
      __targetILP__(iv) {
	index[iv] = addr_rank1(flux->nsite, flux->nf, index0 + iv, n);
      }
      __targetILP__(iv) {
	if (maskv[iv]) flux->fw[index[iv]] *= mask[iv]*maskw[iv];
      }
      __targetILP__(iv) {
	if (maskv[iv]) flux->fe[index[iv]] *= mask[iv]*maske[iv];
      }
      __targetILP__(iv) {
	if (maskv[iv]) flux->fy[index[iv]] *= mask[iv]*masky[iv];
      }
      __targetILP__(iv) {
	if (maskv[iv]) flux->fz[index[iv]] *= mask[iv]*maskz[iv];
      }
    }
    /* Next sites */
  }

  return;
}


/*****************************************************************************
 *
 *  advective_bcs_no_flux
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

int advective_bcs_no_flux(int nf, double * fx, double * fy, double * fz,
			  map_t * map) {
  int n;
  int nlocal[3];
  int ic, jc, kc, index, indexf;
  int status;

  double mask, maskx, masky, maskz;

  assert(nf > 0);
  assert(fx);
  assert(fy);
  assert(fz);
  assert(map);

  coords_nlocal(nlocal);

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic + 1, jc, kc);
	map_status(map, index, &status);
	maskx = (status == MAP_FLUID);

	index = coords_index(ic, jc + 1, kc);
	map_status(map, index, &status);
	masky = (status == MAP_FLUID);

	index = coords_index(ic, jc, kc + 1);
	map_status(map, index, &status);
	maskz = (status == MAP_FLUID);

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	mask = (status == MAP_FLUID);

	for (n = 0;  n < nf; n++) {

	  indexf = addr_rank1(coords_nsites(), nf, index, n);
	  fx[indexf] *= mask*maskx;
	  fy[indexf] *= mask*masky;
	  fz[indexf] *= mask*maskz;

	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  advective_bcs_no_flux_d3qx
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

int advective_bcs_no_flux_d3qx(int nf, double ** flx, map_t * map) {

  int n;
  int nsites;
  int nlocal[3];
  int ic, jc, kc, index0, index1;
  int status;
  int c;
  double mask[PSI_NGRAD];

  assert(nf > 0);
  assert(flx);
  assert(map);

  nsites = coords_nsites();
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = coords_index(ic, jc, kc);
	map_status(map, index0, &status);
	mask[0] = (status == MAP_FLUID);

	for (c = 1; c < PSI_NGRAD; c++) {

	  index1 = coords_index(ic + psi_gr_cv[c][X], jc + psi_gr_cv[c][Y], kc + psi_gr_cv[c][Z]);
	  map_status(map, index1, &status);
	  mask[c] = (status == MAP_FLUID);

	  for (n = 0;  n < nf; n++) {
	    flx[addr_rank1(nsites, nf, index0, n)][c - 1] *= mask[0]*mask[c];
	  }
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  advection_bcs_wall
 *
 *  For the case of flat walls, we kludge the order parameter advection
 *  by borrowing the adjacent fluid value.
 *
 *  The official explanation is this may be viewed as a no gradient
 *  condition on the order parameter near the wall.
 *
 *  This allows third and fourth order (x-direction) advective fluxes
 *  to be computed at interface one cell away from wall. Fluxes at
 *  the wall will always be zero.
 *
 ****************************************************************************/

int advection_bcs_wall(field_t * fphi) {

  int ic, jc, kc, index, index1;
  int nlocal[3];
  int nf;
  double q[NQAB];

  if (wall_at_edge(X) == 0) return 0;

  assert(fphi);

  field_nf(fphi, &nf);
  coords_nlocal(nlocal);
  assert(nf <= NQAB);

  if (cart_coords(X) == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index  = coords_index(ic, jc, kc);
	index1 = coords_index(ic-1, jc, kc);

	field_scalar_array(fphi, index, q);
	field_scalar_array_set(fphi, index1, q);
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {

    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	index1 = coords_index(ic+1, jc, kc);

	field_scalar_array(fphi, index, q);
	field_scalar_array_set(fphi, index1, q);

      }
    }
  }

  return 0;
}
