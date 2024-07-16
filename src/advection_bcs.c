/****************************************************************************
 *
 *  advection_bcs.c
 *
 *  Advection boundary conditions.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2024 The University of Edinburgh
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

__global__ void advection_bcs_no_flux_kernel_v(kernel_3d_v_t k3v,
					       advflux_t * flux, map_t * map);
__global__ void advflux_cs_no_flux_kernel(kernel_3d_t k3d, advflux_t * flux,
					  map_t * map);

/*****************************************************************************
 *
 *  advection_bcs_no_normal_flux
 *
 *  Kernel driver for no-flux boundary conditions.
 *
 *****************************************************************************/

int advection_bcs_no_normal_flux(advflux_t * flux, map_t * map) {

  int nlocal[3] = {0};

  assert(flux);
  assert(map);

  cs_nlocal(flux->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 0, nlocal[Y], 0, nlocal[Z]};
    kernel_3d_v_t k3v = kernel_3d_v(flux->cs, lim, NSIMDVL);

    kernel_3d_launch_param(k3v.kiterations, &nblk, &ntpb);

    TIMER_start(ADVECTION_BCS_KERNEL);

    tdpLaunchKernel(advection_bcs_no_flux_kernel_v, nblk, ntpb, 0, 0,
		    k3v, flux->target, map->target);

    tdpAssert( tdpDeviceSynchronize() );

    TIMER_stop(ADVECTION_BCS_KERNEL);
  }

  return 0;
}

/*****************************************************************************
 *
 *  advection_bcs_no_flux_kernel_v
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

__global__ void advection_bcs_no_flux_kernel_v(kernel_3d_v_t k3v,
					       advflux_t * flux,
					       map_t * map) {
  int kindex = 0;

  for_simt_parallel(kindex, k3v.kiterations, NSIMDVL) {

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

    kernel_3d_v_coords(&k3v, kindex, ic, jc, kc);
    kernel_3d_v_mask(&k3v, ic, jc, kc, maskv);

    kernel_3d_v_cs_index(&k3v, ic, jc, kc, index);
    for_simd_v(iv, NSIMDVL) mask[iv] = (map->status[index[iv]] == MAP_FLUID);

    for_simd_v(iv, NSIMDVL) ix[iv] = ic[iv] - maskv[iv];
    kernel_3d_v_cs_index(&k3v, ix, jc, kc, index);
    for_simd_v(iv, NSIMDVL) maskw[iv] = (map->status[index[iv]] == MAP_FLUID);

    for_simd_v(iv, NSIMDVL) ix[iv] = ic[iv] + maskv[iv];
    kernel_3d_v_cs_index(&k3v, ix, jc, kc, index);
    for_simd_v(iv, NSIMDVL) maske[iv] = (map->status[index[iv]] == MAP_FLUID);

    for_simd_v(iv, NSIMDVL) ix[iv] = jc[iv] + maskv[iv];
    kernel_3d_v_cs_index(&k3v, ic, ix, kc, index);
    for_simd_v(iv, NSIMDVL) masky[iv] = (map->status[index[iv]] == MAP_FLUID);

    for_simd_v(iv, NSIMDVL) ix[iv] = kc[iv] + maskv[iv];
    kernel_3d_v_cs_index(&k3v, ic, jc, ix, index);
    for_simd_v(iv, NSIMDVL) maskz[iv] = (map->status[index[iv]] == MAP_FLUID);

    index0 = k3v.kindex0 + kindex;

    for (int n = 0;  n < flux->nf; n++) {
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

  int nlocal[3] = {0};

  assert(flux);
  assert(map);

  cs_nlocal(flux->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {0, nlocal[X], 0, nlocal[Y], 0, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(flux->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    TIMER_start(ADVECTION_BCS_KERNEL);

    tdpLaunchKernel(advflux_cs_no_flux_kernel, nblk, ntpb, 0, 0,
		    k3d, flux->target, map->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    TIMER_stop(ADVECTION_BCS_KERNEL);
  }

  return 0;
}

/*****************************************************************************
 *
 *  advection_cs_no_flux_kernel
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

__global__ void advflux_cs_no_flux_kernel(kernel_3d_t k3d,
					  advflux_t * flux, map_t * map) {
  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index0 = kernel_3d_cs_index(&k3d, ic, jc, kc);
    int index1 = kernel_3d_cs_index(&k3d, ic+1, jc, kc);

    double m0   =    (map->status[index0] == MAP_FLUID);
    double mask = m0*(map->status[index1] == MAP_FLUID);

    for (int n = 0; n < flux->nf; n++) {
      flux->fx[addr_rank1(flux->nsite, flux->nf, index0, n)] *= mask;
    }

    index1 = kernel_3d_cs_index(&k3d, ic, jc+1, kc);
    mask   = m0*(map->status[index1] == MAP_FLUID);

    for (int n = 0; n < flux->nf; n++) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0, n)] *= mask;
    }

    index1 = kernel_3d_cs_index(&k3d, ic, jc, kc+1);
    mask   = m0*(map->status[index1] == MAP_FLUID);

    for (int n = 0; n < flux->nf; n++) {
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
