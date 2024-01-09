/****************************************************************************
 *
 *  advection_bcs.c
 *
 *  Advection boundary conditions.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2023 The University of Edinburgh
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
#include "timer.h"
#include "advection_bcs.h"

__global__
void advection_bcs_no_flux_kernel_v(kernel_ctxt_t * ktx, advflux_t * flux,
				    map_t * map);
__global__
void advflux_cs_no_flux_kernel(kernel_ctxt_t * ktx, advflux_t * flux,
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

  for_simt_parallel(kindex, kiter, NSIMDVL) {

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
    for_simd_v(iv, NSIMDVL) mask[iv] = (map->status[index[iv]] == MAP_FLUID);  

    for_simd_v(iv, NSIMDVL) ix[iv] = ic[iv] - maskv[iv];
    kernel_coords_index_v(ktx, ix, jc, kc, index);
    for_simd_v(iv, NSIMDVL) maskw[iv] = (map->status[index[iv]] == MAP_FLUID);    

    for_simd_v(iv, NSIMDVL) ix[iv] = ic[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ix, jc, kc, index);
    for_simd_v(iv, NSIMDVL) maske[iv] = (map->status[index[iv]] == MAP_FLUID);

    for_simd_v(iv, NSIMDVL) ix[iv] = jc[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ic, ix, kc, index);
    for_simd_v(iv, NSIMDVL) masky[iv] = (map->status[index[iv]] == MAP_FLUID);

    for_simd_v(iv, NSIMDVL) ix[iv] = kc[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ic, jc, ix, index);
    for_simd_v(iv, NSIMDVL) maskz[iv] = (map->status[index[iv]] == MAP_FLUID);

    index0 = kernel_baseindex(ktx, kindex);

    for (n = 0;  n < flux->nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	index[iv] = addr_rank1(flux->nsite, flux->nf, index0 + iv, n);
      }
      for_simd_v(iv, NSIMDVL) flux->fw[index[iv]] *= mask[iv]*maskw[iv];
      for_simd_v(iv, NSIMDVL) flux->fe[index[iv]] *= mask[iv]*maske[iv];
      for_simd_v(iv, NSIMDVL) flux->fy[index[iv]] *= mask[iv]*masky[iv];
      for_simd_v(iv, NSIMDVL) flux->fz[index[iv]] *= mask[iv]*maskz[iv];
    }
    /* Next sites */
  }

  return;
}

/*****************************************************************************
 *
 *  advflux_cs_no_normal_flux
 *
 *  Kernel driver for no-flux boundary conditions.
 *
 *****************************************************************************/

__host__ int advflux_cs_no_normal_flux(advflux_t * flux, map_t * map) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(flux);
  assert(map);

  cs_nlocal(flux->cs, nlocal);

  limits.imin = 0; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(flux->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(ADVECTION_BCS_KERNEL);

  tdpLaunchKernel(advflux_cs_no_flux_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, flux->target, map->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  TIMER_stop(ADVECTION_BCS_KERNEL);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  advection_cs_no_flux_kernel
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

__global__ void advflux_cs_no_flux_kernel(kernel_ctxt_t * ktx,
					  advflux_t * flux, map_t * map) {
  int kindex;
  __shared__ int kiter;

  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    int n;
    int index0, index1;
    int ic, jc, kc;
    double m0, mask;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index0 = kernel_coords_index(ktx, ic, jc, kc);
    m0     = (map->status[index0] == MAP_FLUID);  

    index1 = kernel_coords_index(ktx, ic+1, jc, kc);
    mask   = m0*(map->status[index1] == MAP_FLUID);  

    for (n = 0; n < flux->nf; n++) {
      flux->fx[addr_rank1(flux->nsite, flux->nf, index0, n)] *= mask;
    }

    index1 = kernel_coords_index(ktx, ic, jc+1, kc);
    mask   = m0*(map->status[index1] == MAP_FLUID);  

    for (n = 0; n < flux->nf; n++) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0, n)] *= mask;
    }

    index1 = kernel_coords_index(ktx, ic, jc, kc+1);
    mask   = m0*(map->status[index1] == MAP_FLUID);  

    for (n = 0; n < flux->nf; n++) {
      flux->fz[addr_rank1(flux->nsite, flux->nf, index0, n)] *= mask;
    }

    /* Next site */
  }

  return;
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
 *  TODO: relocate this to wall field interaction?
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

  /* Only required if there are walls in the x-direction */

  assert(fphi);

  pe_fatal(fphi->pe, "advection_bcs_wall internal error: not implemented\n");

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
