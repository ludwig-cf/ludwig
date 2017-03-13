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
 *  (c) 2009-2017 The University of Edinburgh
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

  cs_nlocal(flux->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(flux->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(ADVECTION_BCS_KERNEL);

  tdpLaunchKernel(advection_bcs_no_flux_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, flux->target, map->target);

  tdpDeviceSynchronize();

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

  __target_simt_for(kindex, kiter, NSIMDVL) {

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
      __targetILP__(iv) flux->fw[index[iv]] *= mask[iv]*maskw[iv];
      __targetILP__(iv) flux->fe[index[iv]] *= mask[iv]*maske[iv];
      __targetILP__(iv) flux->fy[index[iv]] *= mask[iv]*masky[iv];
      __targetILP__(iv) flux->fz[index[iv]] *= mask[iv]*maskz[iv];
    }
    /* Next sites */
  }

  return;
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

  cs_nsites(map->cs, &nsites);
  cs_nlocal(map->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = cs_index(map->cs, ic, jc, kc);
	map_status(map, index0, &status);
	mask[0] = (status == MAP_FLUID);

	for (c = 1; c < PSI_NGRAD; c++) {

	  index1 = cs_index(map->cs, ic + psi_gr_cv[c][X], jc + psi_gr_cv[c][Y], kc + psi_gr_cv[c][Z]);
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
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  double q[NQAB];
  cs_t * cs = NULL; /* To be required */

  /* if (wall_at_edge(X) == 0) return 0;*/

  assert(0); /* Sort out line above */
  assert(fphi);

  field_nf(fphi, &nf);
  cs_nlocal(cs, nlocal);
  cs_cartsz(cs, mpi_cartsz);
  cs_cart_coords(cs, mpi_cartcoords);

  assert(nf <= NQAB);

  if (mpi_cartcoords[X] == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index  = cs_index(cs, ic, jc, kc);
	index1 = cs_index(cs, ic-1, jc, kc);

	field_scalar_array(fphi, index, q);
	field_scalar_array_set(fphi, index1, q);
      }
    }
  }

  if (mpi_cartcoords[X] == mpi_cartsz[X] - 1) {

    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	index1 = cs_index(cs, ic+1, jc, kc);

	field_scalar_array(fphi, index, q);
	field_scalar_array_set(fphi, index1, q);

      }
    }
  }

  return 0;
}
