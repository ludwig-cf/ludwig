/*****************************************************************************
 *
 *  cahn_hilliard.c
 *
 *  The time evolution of the order parameter(s) is described
 *  by the Cahn Hilliard equation
 *
 *     d_t phi + div (u phi - M grad mu) = 0.
 *
 *  Here we allow that there are more than one order parameters phi
 *  e.g., we have composition and surfactant.
 *
 *  The equation is solved here via finite difference. The velocity
 *  field u is assumed known from the hydrodynamic sector. M is the
 *  order parameter mobility. The chemical potential mu is set via
 *  the choice of free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributions:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "advection_s.h"
#include "advection_bcs.h"
#include "cahn_hilliard.h"

__host__ int ch_update_forward_step(ch_t * ch, field_t * phif);
__host__ int ch_flux_mu1(ch_t * ch, fe_t * fe);

__global__ void ch_flux_mu1_kernel(kernel_3d_t k3d, ch_t * ch, fe_t * fe,
				   ch_info_t info);
__global__ void ch_update_kernel_2d(kernel_3d_t k3d, ch_t * ch,
				    field_t * field, ch_info_t info, int xs, int ys);
__global__ void ch_update_kernel_3d(kernel_3d_t k3d, ch_t * ch,
				    field_t * field, ch_info_t info, int xs, int ys);

/*****************************************************************************
 *
 *  ch_create
 *
 *****************************************************************************/

__host__ int ch_create(pe_t * pe, cs_t * cs, ch_info_t info, ch_t ** ch) {

  ch_t * obj = NULL;
  int ndevice = 0;

  assert(pe);
  assert(cs);
  assert(ch);

  obj = (ch_t *) calloc(1, sizeof(ch_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(ch_t) failed\n");

  obj->info = (ch_info_t *) calloc(1, sizeof(ch_info_t));
  assert(obj->info);
  if (obj->info == NULL) pe_fatal(pe, "calloc(ch_info_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  advflux_cs_create(pe, cs, info.nfield, &obj->flux);
  assert(obj->flux);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    tdpAssert(tdpMalloc((void **) &obj->target, sizeof(ch_t)));
    tdpAssert(tdpMemset(obj->target, 0, sizeof(ch_t)));

    /* Coords */
    tdpAssert(tdpMemcpy(&obj->target->cs, &cs->target, sizeof(cs_t *),
			tdpMemcpyHostToDevice));
    /* Flux */
    tdpAssert(tdpMemcpy(&obj->target->flux, &obj->flux->target,
			sizeof(advflux_t *), tdpMemcpyHostToDevice));
  }

  pe_retain(pe);
  ch_info_set(obj, info);

  *ch = obj;

  return 0;
}

/*****************************************************************************
 *
 *  ch_free
 *
 *****************************************************************************/

__host__ int ch_free(ch_t * ch) {

  assert(ch);

  {
    int ndevice = 0;

    tdpAssert( tdpGetDeviceCount(&ndevice) );
    if (ndevice > 0) tdpAssert(tdpFree(ch->target));
  }

  advflux_free(ch->flux);
  pe_free(ch->pe);

  free(ch->info);
  free(ch);

  return 0;
}

/*****************************************************************************
 *
 *  ch_info
 *
 *  Description to pe_info(). Assert exactly two fields.
 *
 *****************************************************************************/

__host__ int ch_info(ch_t * ch) {

  assert(ch);
  assert(ch->info->nfield == 2);

  pe_info(ch->pe, "Number of fields      = %2d\n", ch->info->nfield);
  pe_info(ch->pe, "Mobility (phi)        = %12.5e\n", ch->info->mobility[0]);
  pe_info(ch->pe, "Mobility (psi)        = %12.5e\n", ch->info->mobility[1]);

  return 0;
}

/*****************************************************************************
 *
 *  ch_info_set
 *
 *****************************************************************************/

__host__ int ch_info_set(ch_t * ch, ch_info_t info) {

  assert(ch);

  *ch->info = info;

  return 0;
}

/*****************************************************************************
 *
 *  ch_solver
 *
 *  Compute the fluxes (advective/diffusive) and compute the update
 *  to the order parameter field(s).
 *
 *  hydro is allowed to be NULL, in which case the dynamics is
 *  just relaxational (no velocity field).
 *
 *  map is allowed to be NULL, in which case there are no boundaries.
 *
 *****************************************************************************/

__host__ int ch_solver(ch_t * ch, fe_t * fe, field_t * phi, hydro_t * hydro,
		       map_t * map) {

  assert(ch);
  assert(fe);
  assert(phi);

  /* Compute any advective fluxes first, then diffusive fluxes. */

  if (hydro) {
    hydro_u_halo(hydro); /* Reposition to main to prevent repeat */
    advflux_cs_compute(ch->flux, hydro, phi);
  }
  else {
    advflux_cs_zero(ch->flux); /* Reset flux to zero */
  }

  ch_flux_mu1(ch, fe);

  if (map) advflux_cs_no_normal_flux(ch->flux, map);
  ch_update_forward_step(ch, phi);

  return 0;
}

/*****************************************************************************
 *
 *  ch_flux_mu1
 *
 *  Kernel driver for diffusive flux computation.
 *
 *****************************************************************************/

__host__ int ch_flux_mu1(ch_t * ch, fe_t * fe) {

  int nlocal[3] = {0};
  fe_t * fetarget = NULL;

  assert(ch);
  assert(fe);

  cs_nlocal(ch->cs, nlocal);
  fe->func->target(fe, &fetarget);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {0, nlocal[X], 0, nlocal[Y], 0, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(ch->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(ch_flux_mu1_kernel, nblk, ntpb, 0, 0,
		    k3d, ch->target, fetarget, *ch->info);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}

/*****************************************************************************
 *
 *  ch_flux_mu1_kernel
 *
 *  Unvectorised kernel.
 *
 *  Accumulate [add to a previously computed advective flux] the
 *  'diffusive' contribution related to the chemical potential. It's
 *  computed everywhere regardless of fluid/solid status.
 *
 *  This is a two point stencil the in the chemical potential,
 *  and the mobility is constant.
 *
 *****************************************************************************/

__global__ void ch_flux_mu1_kernel(kernel_3d_t k3d, ch_t * ch, fe_t * fe,
				   ch_info_t info) {

  int kindex = 0;

  assert(ch);
  assert(fe);
  assert(fe->func->mu);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic, jc, kc;
    int index0, index1;
    double flux;
    double mu0[3], mu1[3];

    assert(info.nfield == ch->flux->nf);

    ic = kernel_3d_ic(&k3d, kindex);
    jc = kernel_3d_jc(&k3d, kindex);
    kc = kernel_3d_kc(&k3d, kindex);

    index0 = kernel_3d_cs_index(&k3d, ic, jc, kc);

    fe->func->mu(fe, index0, mu0);

    /* between ic and ic+1 */

    index1 = kernel_3d_cs_index(&k3d, ic+1, jc, kc);
    fe->func->mu(fe, index1, mu1);
    for (int n = 0; n < info.nfield; n++) {
      flux = info.mobility[n]*(mu1[n] - mu0[n]);
      ch->flux->fx[addr_rank1(ch->flux->nsite, info.nfield, index0, n)] -= flux;
    }

    /* y direction */

    index1 = kernel_3d_cs_index(&k3d, ic, jc+1, kc);
    fe->func->mu(fe, index1, mu1);
    for (int n = 0; n < info.nfield; n++) {
      flux = info.mobility[n]*(mu1[n] - mu0[n]);
      ch->flux->fy[addr_rank1(ch->flux->nsite, info.nfield, index0, n)] -= flux;
    }

    /* z direction */

    index1 = kernel_3d_cs_index(&k3d, ic, jc, kc+1);
    fe->func->mu(fe, index1, mu1);
    for (int n = 0; n < info.nfield; n++) {
      flux = info.mobility[n]*(mu1[n] - mu0[n]);
      ch->flux->fz[addr_rank1(ch->flux->nsite, info.nfield, index0, n)] -= flux;
    }

    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  ch_update_forward_step
 *
 *  Update order parameter at each site in turn via the divergence of the
 *  fluxes. This is an Euler forward step:
 *
 *  phi new = phi old - dt*(flux_out - flux_in)
 *
 *  The time step is the LB time step dt = 1.
 *
 *****************************************************************************/

__host__ int ch_update_forward_step(ch_t * ch, field_t * phif) {

  int nlocal[3];
  int xs, ys, zs;

  cs_nlocal(ch->cs, nlocal);
  cs_strides(ch->cs, &xs, &ys, &zs);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(ch->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    if (nlocal[Z] == 1) {
      tdpLaunchKernel(ch_update_kernel_2d, nblk, ntpb, 0, 0,
		      k3d, ch->target, phif->target, *ch->info, xs, ys);
    }
    else {
      tdpLaunchKernel(ch_update_kernel_3d, nblk, ntpb, 0, 0,
		      k3d, ch->target, phif->target, *ch->info, xs, ys);
    }

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}

/******************************************************************************
 *
 *  ch_update_kernel_2d
 *
 *  Two dimensions (no z-fluxes).
 *
 *****************************************************************************/

__global__ void ch_update_kernel_2d(kernel_3d_t k3d, ch_t * ch,
				    field_t * field, ch_info_t info,
				    int xs, int ys) {

  int kindex = 0;

  assert(ch);
  assert(field);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);

    int index = kernel_3d_cs_index(&k3d, ic, jc, 1);

    for (int n = 0; n < info.nfield; n++) {
      double phi = field->data[addr_rank1(ch->flux->nsite, info.nfield, index, n)];

      phi -= ( ch->flux->fx[addr_rank1(ch->flux->nsite, info.nfield, index, n)]
	     - ch->flux->fx[addr_rank1(ch->flux->nsite, info.nfield, index - xs, n)]
	     + ch->flux->fy[addr_rank1(ch->flux->nsite, info.nfield, index, n)]
	     - ch->flux->fy[addr_rank1(ch->flux->nsite, info.nfield, index - ys, n)]);

      field->data[addr_rank1(ch->flux->nsite, info.nfield, index, n)] = phi;
    }
    /* Next site */
  }

  return;
}

/******************************************************************************
 *
 *  ch_update_kernel_3d
 *
 *****************************************************************************/

__global__ void ch_update_kernel_3d(kernel_3d_t k3d, ch_t * ch,
				    field_t * field, ch_info_t info,
				    int xs, int ys) {

  int kindex = 0;

  assert(ch);
  assert(field);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);

    for (int n = 0; n < info.nfield; n++) {
      double phi = field->data[addr_rank1(ch->flux->nsite, info.nfield, index, n)];

      phi -= ( ch->flux->fx[addr_rank1(ch->flux->nsite, info.nfield, index, n)]
	     - ch->flux->fx[addr_rank1(ch->flux->nsite, info.nfield, index - xs, n)]
	     + ch->flux->fy[addr_rank1(ch->flux->nsite, info.nfield, index, n)]
             - ch->flux->fy[addr_rank1(ch->flux->nsite, info.nfield, index - ys, n)]
             + ch->flux->fz[addr_rank1(ch->flux->nsite, info.nfield, index, n)]
	     - ch->flux->fz[addr_rank1(ch->flux->nsite, info.nfield, index - 1,  n)]);

      field->data[addr_rank1(ch->flux->nsite, info.nfield, index, n)] = phi;
    }
    /* Next site */
  }

  return;
}
