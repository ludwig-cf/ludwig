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

#include "field_s.h"
#include "advection_s.h"
#include "advection_bcs.h"
#include "coords_s.h"
#include "cahn_hilliard.h"

__host__ int ch_update_forward_step(ch_t * ch, field_t * phif);
__host__ int ch_flux_mu1(ch_t * ch, fe_t * fe);
__host__ int ch_store_flux(ch_t * ch, field_t * field);
__host__ int ch_store_mu(ch_t * ch, fe_t * fe, field_t * mu);
__host__ int ch_store_fed(ch_t * ch, fe_t * fe, field_t * fed);

__global__ void ch_store_flux_kernel(kernel_ctxt_t * ktx, ch_t * ch, field_t * field,
				   ch_info_t info, int xs, int ys);

__global__ void ch_flux_mu1_kernel(kernel_ctxt_t * ktx, ch_t * ch, fe_t * fe,
				   ch_info_t info);
__global__ void ch_store_mu_kernel(kernel_ctxt_t * ktx, ch_t * ch, fe_t * fe,
				   field_t * mu);
__global__ void ch_store_fed_kernel(kernel_ctxt_t * ktx, ch_t * ch, fe_t * fe,
				   field_t * fed);

__global__ void ch_update_kernel_2d(kernel_ctxt_t * ktx, ch_t * ch,
				    field_t * field, ch_info_t info, int xs, int ys);
__global__ void ch_update_kernel_3d(kernel_ctxt_t * ktx, ch_t * ch,
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

  tdpGetDeviceCount(&ndevice);

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

    tdpGetDeviceCount(&ndevice);
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
		       map_t * map, field_t * total_flux_psi, field_t * advective_flux_psi, field_t * mu, field_t * fed) {

  assert(ch);
  assert(fe);
  assert(phi);
  assert(total_flux_psi);
  assert(advective_flux_psi);
  assert(mu);
  assert(fed);

  /* Compute any advective fluxes first, then diffusive fluxes. */

  if (hydro) {
    hydro_u_halo(hydro); /* Reposition to main to prevent repeat */
    advflux_cs_compute(ch->flux, hydro, phi);

    if (map) advflux_cs_no_normal_flux(ch->flux, map);
    ch_store_flux(ch, advective_flux_psi); /* Computes flux from flux structure into advective_flux_psi/phi */

  }
  else {
    advflux_cs_zero(ch->flux); /* Reset flux to zero */
  }

  ch_flux_mu1(ch, fe); /* Compute total flux into total_flux_psi/phi */
  ch_store_mu(ch, fe, mu);
  ch_store_fed(ch, fe, fed);

  if (map) advflux_cs_no_normal_flux(ch->flux, map);
  ch_store_flux(ch, total_flux_psi);

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

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  fe_t * fetarget = NULL;
  kernel_ctxt_t * ctxt = NULL;

  assert(ch);
  assert(fe);

  cs_nlocal(ch->cs, nlocal);
  fe->func->target(fe, &fetarget);

  limits.imin = 0; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(ch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(ch_flux_mu1_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, ch->target, fetarget, *ch->info);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

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

__global__ void ch_flux_mu1_kernel(kernel_ctxt_t * ktx, ch_t * ch, fe_t * fe,
				   ch_info_t info) {

  int kindex;
  int kiterations;

  assert(ktx);
  assert(ch);
  assert(fe);
  assert(fe->func->mu);


  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index0, index1;
    double flux;
    double mu0[3], mu1[3];

    assert(info.nfield == ch->flux->nf);

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index0 = cs_index(ch->cs, ic, jc, kc);

    fe->func->mu(fe, index0, mu0);

    /* between ic and ic+1 */

    index1 = cs_index(ch->cs, ic+1, jc, kc);
    fe->func->mu(fe, index1, mu1);
    for (int n = 0; n < info.nfield; n++) {
      flux = info.mobility[n]*(mu1[n] - mu0[n]);
      ch->flux->fx[addr_rank1(ch->flux->nsite, info.nfield, index0, n)] -= flux;
    }

    if (ic == 1 && jc == 1 & kc == 20) printf("mu = %f, mu1 = %f\n", mu0[0], mu1[0]);


    /* y direction */

    index1 = cs_index(ch->cs, ic, jc+1, kc);
    fe->func->mu(fe, index1, mu1);
    for (int n = 0; n < info.nfield; n++) {
      flux = info.mobility[n]*(mu1[n] - mu0[n]);
      ch->flux->fy[addr_rank1(ch->flux->nsite, info.nfield, index0, n)] -= flux;
    }

    /* z direction */

    index1 = cs_index(ch->cs, ic, jc, kc+1);
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
  dim3 nblk, ntpb;

  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  cs_nlocal(ch->cs, nlocal);
  cs_strides(ch->cs, &xs, &ys, &zs);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(ch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  if (nlocal[Z] == 1) {
    tdpLaunchKernel(ch_update_kernel_2d, nblk, ntpb, 0, 0,
		    ctxt->target, ch->target, phif->target, *ch->info, xs, ys);
  }
  else {
    tdpLaunchKernel(ch_update_kernel_3d, nblk, ntpb, 0, 0,
		    ctxt->target, ch->target, phif->target, *ch->info, xs, ys);
  }

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/******************************************************************************
 *
 *  ch_update_kernel_2d
 *
 *  Two dimensions (no z-fluxes).
 *
 *****************************************************************************/

__global__ void ch_update_kernel_2d(kernel_ctxt_t * ktx, ch_t * ch,
				    field_t * field, ch_info_t info, int xs, int ys) {

  int kindex;
  int kiterations;

  assert(ktx);
  assert(ch);
  assert(field);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);

    int index = cs_index(ch->cs, ic, jc, 1);

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

__global__ void ch_update_kernel_3d(kernel_ctxt_t * ktx, ch_t * ch,
				    field_t * field, ch_info_t info,
				    int xs, int ys) {

  int kindex;
  int kiterations;

  assert(ktx);
  assert(ch);
  assert(field);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);

    int index = cs_index(ch->cs, ic, jc, kc);

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

/*****************************************************************************
 *
 *  ch_store_flux
 *
 *****************************************************************************/

__host__ int ch_store_flux(ch_t * ch, field_t * field) {

  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;

  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  cs_nlocal(ch->cs, nlocal);
  cs_strides(ch->cs, &xs, &ys, &zs);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(ch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(ch_store_flux_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, ch->target, field->target, *ch->info, xs, ys);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/******************************************************************************
 *
 *  ch_store_flux
 *
 *****************************************************************************/

__global__ void ch_store_flux_kernel(kernel_ctxt_t * ktx, ch_t * ch,
				    field_t * field, ch_info_t info,
				    int xs, int ys) {

  int kindex;
  int kiterations;

  assert(ktx);
  assert(ch);
  assert(field->nf==3);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);

    int index = cs_index(ch->cs, ic, jc, kc);

    
    field->data[addr_rank1(ch->flux->nsite, 3, index, 0)] =
      	      -	ch->flux->fx[addr_rank1(ch->flux->nsite, 2, index, 1)]
              + ch->flux->fx[addr_rank1(ch->flux->nsite, 2, index - xs, 1)];

    field->data[addr_rank1(ch->flux->nsite, 3, index, 1)] =
      	      -	ch->flux->fy[addr_rank1(ch->flux->nsite, 2, index, 1)]
              + ch->flux->fy[addr_rank1(ch->flux->nsite, 2, index - ys, 1)];

    field->data[addr_rank1(ch->flux->nsite, 3, index, 2)] =
      	      -	ch->flux->fz[addr_rank1(ch->flux->nsite, 2, index, 1)]
              + ch->flux->fz[addr_rank1(ch->flux->nsite, 2, index - 1, 1)];

    /* Next site */
  }

  return;
}


/*****************************************************************************
 *
 *  ch_store_mu
 *
 *****************************************************************************/

__host__ int ch_store_mu(ch_t * ch, fe_t * fe, field_t * mu) {

  int nlocal[3];
  dim3 nblk, ntpb;
  fe_t * fetarget = NULL;

  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(ch);
  assert(fe);

  cs_nlocal(ch->cs, nlocal);
  fe->func->target(fe, &fetarget);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(ch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(ch_store_mu_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, ch->target, fetarget, mu->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/******************************************************************************
 *
 *  ch_store_mu_kernel
 *
 *****************************************************************************/

__global__ void ch_store_mu_kernel(kernel_ctxt_t * ktx, ch_t * ch, fe_t * fe, field_t * mu) {

  int kindex;
  int kiterations;
  int index;
  double mu0[2];

  assert(ktx);
  assert(ch);
  assert(mu);
  assert(fe);
  assert(mu->nf==2);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);

    int index = cs_index(ch->cs, ic, jc, kc);

    fe->func->mu(fe, index, mu0);
    
    mu->data[addr_rank1(ch->flux->nsite, 2, index, 0)] = mu0[0];
    mu->data[addr_rank1(ch->flux->nsite, 2, index, 1)] = mu0[1];

    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  ch_store_mu
 *
 *****************************************************************************/

__host__ int ch_store_fed(ch_t * ch, fe_t * fe, field_t * fed) {

  int nlocal[3];
  dim3 nblk, ntpb;
  fe_t * fetarget = NULL;

  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(ch);
  assert(fe);

  cs_nlocal(ch->cs, nlocal);
  fe->func->target(fe, &fetarget);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(ch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(ch_store_fed_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, ch->target, fetarget, fed->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/******************************************************************************
 *
 *  ch_store_fed_kernel
 *
 *****************************************************************************/

__global__ void ch_store_fed_kernel(kernel_ctxt_t * ktx, ch_t * ch, fe_t * fe, field_t * fed) {

  int kindex;
  int kiterations;
  int index;
  double fed0[2];

  assert(ktx);
  assert(ch);
  assert(fed);
  assert(fe);
  assert(fed->nf==2);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);

    int index = cs_index(ch->cs, ic, jc, kc);

    fe->func->fed(fe, index, fed0);
    
    fed->data[addr_rank1(ch->flux->nsite, 2, index, 0)] = fed0[0];
    fed->data[addr_rank1(ch->flux->nsite, 2, index, 1)] = fed0[1];

    /* Next site */
  }

  return;
}


