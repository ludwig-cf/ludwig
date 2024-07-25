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
 *  (c) 2010-2024 The University of Edinburgh
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

__global__ static void leslie_update_kernel(kernel_3d_t k3d,
					    fe_polar_t * fe,
					    field_t * fp,
					    hydro_t * hydro,
					    advflux_t * flux,
					    leslie_param_t param);

__global__ static void leslie_self_advection_kernel(kernel_3d_t k3d,
						    field_t * p,
						    hydro_t * hydro,
						    double swim);

__device__ static void leslie_u_gradient_tensor(const kernel_3d_t * k3d,
						hydro_t * hydro,
						int ic, int jc, int kc,
						double w[3][3]);

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
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(obj->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(leslie_update_kernel, nblk, ntpb, 0, 0,
		    k3d, obj->fe->target, obj->p->target,
		    hydro->target, flux->target, obj->param);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
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

__global__ static void leslie_update_kernel(kernel_3d_t k3d,
					    fe_polar_t * fe,
					    field_t * fp,
					    hydro_t * hydro,
					    advflux_t * flux,
					    leslie_param_t param) {
  int kindex = 0;
  const double dt = 1.0;

  assert(fe);
  assert(fp);
  assert(flux);

  for_simt_parallel(kindex, k3d.kiterations, 1)  {

    int ic    = kernel_3d_ic(&k3d, kindex);
    int jc    = kernel_3d_jc(&k3d, kindex);
    int kc    = kernel_3d_kc(&k3d, kindex);
    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);

    double p[3] = {0};
    double h[3] = {0};          /* molecular field (vector) */
    double w[3][3] = {0};       /* Velocity gradient tensor */
    double d[3][3] = {0};       /* Symmetric velocity gradient tensor */
    double omega[3][3] = {0};   /* Antisymmetric ditto */

    field_vector(fp, index, p);
    fe_polar_mol_field(fe, index, h);
    if (hydro) leslie_u_gradient_tensor(&k3d, hydro, ic, jc, kc, w);

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

__global__ static void leslie_self_advection_kernel(kernel_3d_t k3d,
						    field_t * p,
						    hydro_t * hydro,
						    double swim) {
  int kindex = 0;

  assert(p);
  assert(hydro);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic    = kernel_3d_ic(&k3d, kindex);
    int jc    = kernel_3d_jc(&k3d, kindex);
    int kc    = kernel_3d_kc(&k3d, kindex);
    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
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

    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(obj->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(leslie_self_advection_kernel, nblk, ntpb, 0, 0,
		    k3d, obj->p->target, hydro->target, obj->param.swim);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}

/*****************************************************************************
 *
 *  leslie_u_gradient_tensor
 *
 *  A copy of the hydro_u_gradient_tensor() routine with no Lees-Edwards
 *  conditions.
 *
 *  A __device__ version is required.
 *
 *****************************************************************************/

__device__ static void leslie_u_gradient_tensor(const kernel_3d_t * k3d,
						hydro_t * hydro,
						int ic, int jc, int kc,
						double w[3][3]) {
  assert(hydro);

  int m1 = kernel_3d_cs_index(k3d, ic - 1, jc, kc);
  int p1 = kernel_3d_cs_index(k3d, ic + 1, jc, kc);

  w[X][X] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, X)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, X)]);
  w[Y][X] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, Y)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, Y)]);
  w[Z][X] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, Z)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, Z)]);

  m1 = kernel_3d_cs_index(k3d, ic, jc - 1, kc);
  p1 = kernel_3d_cs_index(k3d, ic, jc + 1, kc);

  w[X][Y] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, X)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, X)]);
  w[Y][Y] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, Y)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, Y)]);
  w[Z][Y] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, Z)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, Z)]);

  m1 = kernel_3d_cs_index(k3d, ic, jc, kc - 1);
  p1 = kernel_3d_cs_index(k3d, ic, jc, kc + 1);

  w[X][Z] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, X)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, X)]);
  w[Y][Z] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, Y)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, Y)]);
  w[Z][Z] = 0.5*(hydro->u->data[addr_rank1(hydro->nsite, NHDIM, p1, Z)] -
		 hydro->u->data[addr_rank1(hydro->nsite, NHDIM, m1, Z)]);

  /* Enforce tracelessness */

  {
    double tr = (1.0/3.0)*(w[X][X] + w[Y][Y] + w[Z][Z]);
    w[X][X] -= tr;
    w[Y][Y] -= tr;
    w[Z][Z] -= tr;
  }

  return;
}
