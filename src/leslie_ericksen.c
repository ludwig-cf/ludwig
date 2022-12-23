/*****************************************************************************
 *
 *  leslie_ericksen.c
 *
 *  Updates a vector order parameter according to something looking
 *  like a Leslie-Ericksen equation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "advection_s.h"
#include "leslie_ericksen.h"

__global__ static void leslie_update_kernel(kernel_ctxt_t * ktx,
					    fe_polar_t * fe,
					    field_t * fp,
					    hydro_t * hydro,
					    advflux_t * flux,
					    leslie_param_t param);

__global__ static void leslie_self_advection_kernel(kernel_ctxt_t * ktx,
						    field_t * p,
						    hydro_t * hydro,
						    double swim);

/*****************************************************************************
 *
 *  leslie_ericksen_create
 *
 *****************************************************************************/

int leslie_ericksen_create(pe_t * pe, cs_t * cs, fe_polar_t * fe, field_t * p,
			   const leslie_param_t * param,
			   leslie_ericksen_t ** pobj) {

  leslie_ericksen_t * obj = NULL;

  obj = (leslie_ericksen_t *) calloc(1, sizeof(leslie_ericksen_t));
  assert(obj);

  /* Initialise */

  obj->pe = pe;
  obj->cs = cs;
  obj->fe = fe;
  obj->p  = p;
  obj->param = *param;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  leslie_ericksen_free
 *
 *****************************************************************************/

int leslie_ericksen_free(leslie_ericksen_t ** pobj) {

  assert(pobj && *pobj);

  free(*pobj);
  *pobj = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  leslie_ericksen_update
 *
 *  The hydro is allowed to be NULL, in which case the dynamics is
 *  purely relaxational.
 *
 *  Note there is a side effect on the velocity field here if the
 *  self-advection term is not zero.
 *
 *****************************************************************************/

__host__ int leslie_ericksen_update(leslie_ericksen_t * obj, hydro_t * hydro) {

  int nlocal[3] = {0};
  advflux_t * flux = NULL;

  assert(obj);
  assert(hydro);

  cs_nlocal(obj->cs, nlocal);

  /* Device version should allocate flux obj only once on creation. */
  advflux_cs_create(obj->pe, obj->cs, 3, &flux);

  if (hydro) {
    /* Add self-advection term if present; halo swap; compute
     * advective fluxes */
    leslie_ericksen_self_advection(obj, hydro);
    hydro_u_halo(hydro);
    advflux_cs_compute(flux, hydro, obj->p);
  }

  {
    /* Update driver */
    dim3 nblk, ntpb;
    kernel_info_t limits = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_ctxt_t * ctxt = NULL;

    kernel_ctxt_create(obj->cs, 1, limits, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(leslie_update_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, obj->fe, obj->p->target, hydro->target,
		    flux->target, obj->param);
    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
  }

  advflux_free(flux);

  return 0;
}

/*****************************************************************************
 *
 *  leslie_update_kernel
 *
 *  hydro is allowed to be NULL, in which case we have relaxational
 *  dynamics only.
 *
 *****************************************************************************/

__global__ static void leslie_update_kernel(kernel_ctxt_t * ktx,
					    fe_polar_t * fe,
					    field_t * fp,
					    hydro_t * hydro,
					    advflux_t * flux,
					    leslie_param_t param) {
  int kindex = 0;
  int kiterations = 0;
  const double dt = 1.0;

  assert(ktx);
  assert(fe);
  assert(fp);
  assert(flux);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1)  {

    int ic    = kernel_coords_ic(ktx, kindex);
    int jc    = kernel_coords_jc(ktx, kindex);
    int kc    = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);

    double p[3] = {0};
    double h[3] = {0};          /* molecular field (vector) */
    double w[3][3] = {0};       /* Velocity gradient tensor */
    double d[3][3] = {0};       /* Symmetric velocity gradient tensor */
    double omega[3][3] = {0};   /* Antisymmetric ditto */

    field_vector(fp, index, p);
    fe_polar_mol_field(fe, index, h);
    if (hydro) hydro_u_gradient_tensor(hydro, ic, jc, kc, w);

    /* Note that the convection for Leslie Ericksen is that
     * w_ab = d_a u_b, which is the transpose of what the
     * above returns. Hence an extra minus sign in the
     * omega term in the following. */

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	d[ia][ib]     =  0.5*(w[ia][ib] + w[ib][ia]);
	omega[ia][ib] = -0.5*(w[ia][ib] - w[ib][ia]);
      }
    }

    /* updates involve the following fluxes */

    int im1 = cs_index(flux->cs, ic-1, jc, kc);
    int jm1 = cs_index(flux->cs, ic, jc-1, kc);
    int km1 = cs_index(flux->cs, ic, jc, kc-1);

    for (int ia = 0; ia < 3; ia++) {

      double sum = 0.0;
      for (int ib = 0; ib < 3; ib++) {
	sum += param.lambda*d[ia][ib]*p[ib] - omega[ia][ib]*p[ib];
      }

      p[ia] += dt*(- flux->fx[addr_rank1(flux->nsite, 3, index, ia)]
		   + flux->fx[addr_rank1(flux->nsite, 3, im1,   ia)]
		   - flux->fy[addr_rank1(flux->nsite, 3, index, ia)]
		   + flux->fy[addr_rank1(flux->nsite, 3, jm1,   ia)]
		   - flux->fz[addr_rank1(flux->nsite, 3, index, ia)]
		   + flux->fz[addr_rank1(flux->nsite, 3, km1,   ia)]
		   + sum + param.Gamma*h[ia]);
    }

    field_vector_set(fp, index, p);
  }

  return;
}

/*****************************************************************************
 *
 *  leslie_self_advection_kernel
 *
 *****************************************************************************/

__global__ static void leslie_self_advection_kernel(kernel_ctxt_t * ktx,
						    field_t * p,
						    hydro_t * hydro,
						    double swim) {
  int kindex = 0;
  int kiterations = 0;

  assert(ktx);
  assert(p);
  assert(hydro);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic    = kernel_coords_ic(ktx, kindex);
    int jc    = kernel_coords_jc(ktx, kindex);
    int kc    = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);
    double p3[3] = {0};
    double u3[3] = {0};

    field_vector(p, index, p3);
    hydro_u(hydro, index, u3);

    u3[X] += swim*p3[X];
    u3[Y] += swim*p3[Y];
    u3[Z] += swim*p3[Z];

    hydro_u_set(hydro, index, u3);
  }

  return;
}

/*****************************************************************************
 *
 *  leslie_self_advection
 *
 *  Driver to add the self-advection velocity to the current hydro->u.
 *  This then appears in advective fluxes.
 *
 *  1. There is a somewhat open question on whether one really wants the
 *  self advection to appear anywhere else as a side-effect.
 *  2. If we have a halo in p, then we can avoid a halo in u.
 *
 *****************************************************************************/

__host__ int leslie_ericksen_self_advection(leslie_ericksen_t * obj,
					    hydro_t * hydro) {
  int nlocal[3] = {0};

  assert(obj);
  assert(hydro);

  cs_nlocal(obj->cs, nlocal);

  /* Don't bother if swim = 0 */

  if (fabs(obj->param.swim) > 0.0) {

    dim3 nblk, ntpb;
    kernel_info_t limits = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_ctxt_t * ctxt = NULL;

    kernel_ctxt_create(obj->cs, 1, limits, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(leslie_self_advection_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, obj->p->target, hydro->target,
		    obj->param.swim);
    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
  }

  return 0;
}
