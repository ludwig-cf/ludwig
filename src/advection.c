/*****************************************************************************
 *
 *  advection.c
 *
 *  Computes advective order parameter fluxes from the current
 *  velocity field (from hydrodynamics) and the the current
 *  order parameter(s).
 *
 *  Fluxes are all computed at the interface of the control cells
 *  surrounding each lattice site. Unique face fluxes guarantee
 *  conservation of the order parameter.
 *
 *  To deal with Lees-Edwards boundaries positioned at x = constant
 *  we have to allow the 'east' face flux to be stored separately
 *  to the 'west' face flux. There's no effect in the y- or z-
 *  directions.
 *
 *  Any solid-fluid boundary conditions are dealt with post-hoc by
 *  in advection_bcs.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017  The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "field_s.h"
#include "advection_s.h"
#include "psi_gradients.h"
#include "hydro_s.h"
#include "timer.h"

__host__
int advection_le_1st(advflux_t * flux, hydro_t * hydro, field_t * field);
__host__
int advection_le_2nd(advflux_t * flux, hydro_t * hydro, field_t * field);
__host__
int advection_le_3rd(advflux_t * flux, hydro_t * hydro, field_t * field);

static int advection_le_4th(advflux_t * flux, hydro_t * hydro, int nf,
			    double * f);
static int advection_le_5th(advflux_t * flux, hydro_t * hydro, int nf,
			    double * f);

__global__
void advection_le_1st_kernel(kernel_ctxt_t * ktx, advflux_t * flux,
			     hydro_t * hydro, field_t * field);
__global__
void advection_2nd_kernel(kernel_ctxt_t * ktx, advflux_t * flux,
			  hydro_t * hydro, field_t * field);
__global__
void advection_2nd_kernel_v(kernel_ctxt_t * ktx, advflux_t * flux,
			    hydro_t * hydro, field_t * field);
__global__
void advection_3rd_kernel_v(kernel_ctxt_t * ktx, advflux_t * flux, 
			    hydro_t * hydro, field_t * field);
__global__
void advection_le_3rd_kernel_v(kernel_ctxt_t * ktx, lees_edw_t * le,
			       advflux_t * flux,
			       hydro_t * hydro, field_t * field);


/* SCHEDULED FOR DELETION! */
static int order_ = 1; /* Default is upwind (bad!) */

/*****************************************************************************
 *
 *  advection_order_set
 *
 *****************************************************************************/

int advection_order_set(const int n) {

  order_ = n;
  return 0;
}

/*****************************************************************************
 *
 *  advection_order
 *
 *****************************************************************************/

int advection_order(int * order) {

  assert(order);

  *order = order_;

  return 0;
}

/*****************************************************************************
 *
 *  advflux_cs_create
 *
 *****************************************************************************/

__host__ int advflux_cs_create(pe_t * pe, cs_t * cs, int nf, advflux_t ** pobj) {

  assert(pe);
  assert(cs);

  return advflux_create(pe, cs, NULL, nf, pobj);
}

/*****************************************************************************
 *
 *  advflux_le_create
 *
 *****************************************************************************/

__host__ int advflux_le_create(pe_t * pe, cs_t * cs, lees_edw_t * le, int nf,
			       advflux_t ** pobj) {

  assert(pe);
  assert(cs);
  assert(le);
  return advflux_create(pe, cs, le, nf, pobj);
}

/*****************************************************************************
 *
 *  advflux_create
 *
 *****************************************************************************/

__host__ int advflux_create(pe_t * pe, cs_t * cs, lees_edw_t * le, int nf,
			    advflux_t ** pobj) {

  int ndevice;
  int nsites;
  double * tmp;
  advflux_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (advflux_t *) calloc(1, sizeof(advflux_t));
  if (obj == NULL) pe_fatal(pe, "calloc(advflux) failed\n");

  if (cs) cs_nsites(cs, &nsites);
  if (le) lees_edw_nsites(le, &nsites);

  obj->pe = pe;
  obj->cs = cs;
  obj->le = le;
  obj->nf = nf;
  obj->nsite = nsites;

  obj->fe = (double *) calloc(nsites*nf, sizeof(double));
  obj->fw = (double *) calloc(nsites*nf, sizeof(double));
  obj->fy = (double *) calloc(nsites*nf, sizeof(double));
  obj->fz = (double *) calloc(nsites*nf, sizeof(double));

  if (obj->fe == NULL) pe_fatal(pe, "calloc(advflux->fe) failed\n");
  if (obj->fw == NULL) pe_fatal(pe, "calloc(advflux->fw) failed\n");
  if (obj->fy == NULL) pe_fatal(pe, "calloc(advflux->fy) failed\n");
  if (obj->fz == NULL) pe_fatal(pe, "calloc(advflux->fz) failed\n");


  /* Allocate target copy of structure (or alias) */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    cs_t * cstarget = NULL;
    lees_edw_t * letarget = NULL;

    tdpMalloc((void **) &obj->target, sizeof(advflux_t));
    tdpMemset(obj->target, 0, sizeof(advflux_t));
    tdpMalloc((void **) &tmp, nf*nsites*sizeof(double));
    tdpMemcpy(&obj->target->fe, &tmp, sizeof(double *), tdpMemcpyHostToDevice);

    tdpMalloc((void **) &tmp, nf*nsites*sizeof(double));
    tdpMemcpy(&obj->target->fw, &tmp, sizeof(double *), tdpMemcpyHostToDevice);

    tdpMalloc((void **) &tmp, nf*nsites*sizeof(double));
    tdpMemcpy(&obj->target->fy, &tmp, sizeof(double *), tdpMemcpyHostToDevice);

    tdpMalloc((void **) &tmp, nf*nsites*sizeof(double));
    tdpMemcpy(&obj->target->fz, &tmp, sizeof(double *), tdpMemcpyHostToDevice);

    tdpMemcpy(&obj->target->nf, &obj->nf, sizeof(int), tdpMemcpyHostToDevice);
    tdpMemcpy(&obj->target->nsite, &obj->nsite, sizeof(int),
	      tdpMemcpyHostToDevice);

    if (cs) cs_target(cs, &cstarget);
    if (le) lees_edw_target(le, &letarget);
    tdpMemcpy(&obj->target->cs, &cstarget, sizeof(cs_t *),
	      tdpMemcpyHostToDevice);
    tdpMemcpy(&obj->target->le, &letarget, sizeof(lees_edw_t *),
	      tdpMemcpyHostToDevice);
  }

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  advflux_free
 *
 *****************************************************************************/

__host__ int advflux_free(advflux_t * obj) {

  int ndevice;
  double * tmp;

  assert(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpMemcpy(&tmp, &obj->target->fe, sizeof(double *), tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &obj->target->fw, sizeof(double *), tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &obj->target->fy, sizeof(double *), tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &obj->target->fz, sizeof(double *), tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpFree(obj->target);
  }

  free(obj->fe);
  free(obj->fw);
  free(obj->fy);
  free(obj->fz);
  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  advection_memcpy
 *
 *****************************************************************************/

__host__ int advection_memcpy(advflux_t * obj) {

  int ndevice;

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {
    assert(0); /* Please fill me in if needed. */
  }

  return 0;
}

/*****************************************************************************
 *
 *  advection_x
 *
 *****************************************************************************/

int advection_x(advflux_t * obj, hydro_t * hydro, field_t * field) {

  int nf;

  assert(obj);
  assert(hydro);
  assert(field);

  field_nf(field, &nf);

  /* For given LE , and given order, compute fluxes */

  TIMER_start(ADVECTION_X_KERNEL);
  switch (order_) {
  case 1:
    advection_le_1st(obj, hydro, field);
    break;
  case 2:
    advection_le_2nd(obj, hydro, field);
    break;
  case 3:
    advection_le_3rd(obj, hydro, field);
    break;
  case 4:
    advection_le_4th(obj, hydro, nf, field->data);
    break;
  case 5:
    advection_le_5th(obj, hydro, nf, field->data);
    break; 
  default:
    pe_fatal(obj->pe, "Unexpected advection scheme order\n");
  }
  TIMER_stop(ADVECTION_X_KERNEL);

  return 0;
}

/*****************************************************************************
 *
 *  advection_le_1st
 *
 *  Kernel driver routine
 *
 *****************************************************************************/

__host__ int advection_le_1st(advflux_t * flux, hydro_t * hydro,
			      field_t * field) {
  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(flux);
  assert(hydro);
  assert(field);

  cs_nlocal(flux->cs, nlocal);

  /* Limits */

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(flux->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(advection_le_1st_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, flux->target, hydro->target, field->target);

  tdpDeviceSynchronize();

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  advection_le_1st_kernel
 *
 *  The advective fluxes are computed via first order upwind
 *  allowing for LE planes.
 * 
 *  The following are set (as for all the upwind routines):
 *
 *  fluxw  ('west') is the flux in x-direction between cells ic-1, ic
 *  fluxe  ('east') is the flux in x-direction between cells ic, ic+1
 *  fluxy           is the flux in y-direction between cells jc, jc+1
 *  fluxz           is the flux in z-direction between cells kc, kc+1
 *
 *****************************************************************************/

__global__ void advection_le_1st_kernel(kernel_ctxt_t * ktx,
					advflux_t * flux,
					hydro_t * hydro,
					field_t * field) {

  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(flux);
  assert(hydro);
  assert(field);

  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    int ia;
    int n;
    int ic, jc, kc;
    int index0, index1, index;
    int icm1, icp1;
    double u0[3], u1[3], u;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index0 = kernel_coords_index(ktx, ic, jc, kc);

    icm1 = lees_edw_ic_to_buff(flux->le, ic, -1);
    icp1 = lees_edw_ic_to_buff(flux->le, ic, +1);

    /* west face (icm1 and ic) */

    for (ia = 0; ia < 3; ia++) {
      u0[ia] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index0, ia)];
    }

    index1 = kernel_coords_index(ktx, icm1, jc, kc);

    u1[X] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1, X)];
    u = 0.5*(u0[X] + u1[X]);

    index = index0;
    if (u > 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fw[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }


    /* east face (ic and icp1) */

    index1 = kernel_coords_index(ktx, icp1, jc, kc);

    u1[X] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1, X)];
    u = 0.5*(u0[X] + u1[X]);

    index = index0;
    if (u < 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fe[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }

    /* y direction */

    index1 = kernel_coords_index(ktx, ic, jc+1, kc);

    u1[Y] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1, Y)];
    u = 0.5*(u0[Y] + u1[Y]);

    index = index0;
    if (u < 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }

    /* z direction */

    index1 = kernel_coords_index(ktx, ic, jc, kc+1);

    u1[Z] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1, Z)];
    u = 0.5*(u0[Z] + u1[Z]);

    index = index0;
    if (u < 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fz[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }

    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  advection_le_2nd
 *
 *  Kernel driver routine
 *
 *****************************************************************************/

__host__ int advection_le_2nd(advflux_t * flux, hydro_t * hydro,
			      field_t * field) {
  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(flux);
  assert(hydro);
  assert(field);

  cs_nlocal(flux->cs, nlocal);

  /* Limits */

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(flux->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(advection_2nd_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, flux->target, hydro->target, field->target);

  tdpDeviceSynchronize();

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  advection_2nd_kernel
 *
 *  'Centred difference' advective fluxes, allowing LE planes.
 *
 *  Symmetric two-point stencil.
 *
 *  Non-vectorised version retained for interest.
 *
 *****************************************************************************/

__global__ void advection_2nd_kernel(kernel_ctxt_t * ktx, advflux_t * flux,
				     hydro_t * hydro, field_t * field) {
  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(flux);
  assert(hydro);
  assert(field);

  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    int ia, n;
    int ic, jc, kc;
    int icm1, icp1;
    int index0, index1;
    double u, u0[3], u1[3];

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index0 = kernel_coords_index(ktx, ic, jc, kc);

    icm1 = lees_edw_ic_to_buff(flux->le, ic, -1);
    icp1 = lees_edw_ic_to_buff(flux->le, ic, +1);

    for (ia = 0; ia < NHDIM; ia++) {
      u0[ia] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index0, ia)];
    }

    /* west face (ic - 1 and ic) */

    index1 = kernel_coords_index(ktx, icm1, jc, kc);
    u1[X] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1, X)];
    u = 0.5*(u0[X] + u1[X]);

    for (n = 0; n < field->nf; n++) {
      flux->fw[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index1, n)]
       + field->data[addr_rank1(field->nsites, field->nf, index0, n)]);
    }

    /* east face (ic and ic + 1) */

    index1 = kernel_coords_index(ktx, icp1, jc, kc);
    u1[X] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1, X)];
    u = 0.5*(u0[X] + u1[X]);

    for (n = 0; n < flux->nf; n++) {
      flux->fe[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index0, n)] +
	 field->data[addr_rank1(field->nsites, field->nf, index1, n)]);
    }

    /* y direction */

    index1 = kernel_coords_index(ktx, ic, jc+1, kc);
    u1[Y] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1, Y)];
    u = 0.5*(u0[Y] + u1[Y]);

    for (n = 0; n < flux->nf; n++) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index0, n)] +
	 field->data[addr_rank1(field->nsites, field->nf, index1, n)]);
    }

    /* z direction */

    index1 = kernel_coords_index(ktx, ic, jc, kc+1);
    u1[Z] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1, Z)];
    u = 0.5*(u0[Z] + u1[Z]);

    for (n = 0; n < flux->nf; n++) {
      flux->fz[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index0, n)] +
	 field->data[addr_rank1(field->nsites, field->nf, index1, n)]);
    }
    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  advection_2nd_kernel_v
 *
 *  Vectorisaed version
 *
 *****************************************************************************/

__global__ void advection_2nd_kernel_v(kernel_ctxt_t * ktx, advflux_t * flux,
				       hydro_t * hydro, field_t * field) {
  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(flux);
  assert(hydro);
  assert(field);

  kiter = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiter, NSIMDVL) {

    int ia, iv, n;
    int ic[NSIMDVL], jc[NSIMDVL], kc[NSIMDVL];
    int p1[NSIMDVL], m1[NSIMDVL];
    int index0[NSIMDVL], index1[NSIMDVL];
    int maskv[NSIMDVL];
    double u0[NHDIM][NSIMDVL], u1[NHDIM][NSIMDVL];

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_coords_index_v(ktx, ic, jc, kc, index0);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    for_simd_v(iv, NSIMDVL) {
      m1[iv] = lees_edw_ic_to_buff(flux->le, ic[iv], -maskv[iv]);
    }
    for_simd_v(iv, NSIMDVL) {
      p1[iv] = lees_edw_ic_to_buff(flux->le, ic[iv], +maskv[iv]);
    }


    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u0[ia][iv] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index0[iv], ia)];
      }
    }

    /* west face (ic - 1 and ic) */

    lees_edw_index_v(flux->le, m1, jc, kc, index1);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1[iv], ia)];
      }
    }

    for (n = 0; n < field->nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	flux->fw[addr_rank1(flux->nsite, flux->nf, index0[iv], n)] =
	  0.5*(u0[X][iv] + u1[X][iv])*maskv[iv]*0.5*
	  (field->data[addr_rank1(field->nsites, field->nf, index1[iv], n)]
	   + field->data[addr_rank1(field->nsites, field->nf, index0[iv], n)]);
      }
    }

    /* east face (ic and ic + 1) */

    lees_edw_index_v(flux->le, p1, jc, kc, index1);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1[iv], ia)];
      }
    }

    for (n = 0; n < flux->nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	flux->fe[addr_rank1(flux->nsite, flux->nf, index0[iv], n)] =
	  0.5*(u0[X][iv] + u1[X][iv])*maskv[iv]*0.5*
	  (field->data[addr_rank1(field->nsites, field->nf, index0[iv], n)] +
	   field->data[addr_rank1(field->nsites, field->nf, index1[iv], n)]);
      }
    }

    /* y direction */

    for_simd_v(iv, NSIMDVL) p1[iv] = jc[iv] + maskv[iv];
    lees_edw_index_v(flux->le, ic, p1, kc, index1);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1[iv], ia)];
      }
    }

    for (n = 0; n < flux->nf; n++) {
      for_simd_v(iv, NSIMDVL) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0[iv], n)] =
	0.5*(u0[Y][iv] + u1[Y][iv])*maskv[iv]*0.5*
	(field->data[addr_rank1(field->nsites, field->nf, index0[iv], n)] +
	 field->data[addr_rank1(field->nsites, field->nf, index1[iv], n)]);
      }
    }

    /* z direction */

    for_simd_v(iv, NSIMDVL) p1[iv] = kc[iv] + maskv[iv];
    lees_edw_index_v(flux->le, ic, jc, p1, index1);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite, NHDIM, index1[iv], ia)];
      }
    }

    for (n = 0; n < flux->nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	flux->fz[addr_rank1(flux->nsite, flux->nf, index0[iv], n)] =
	  0.5*(u0[Z][iv] + u1[Z][iv])*maskv[iv]*0.5*
	  (field->data[addr_rank1(field->nsites, field->nf, index0[iv], n)] +
	   field->data[addr_rank1(field->nsites, field->nf, index1[iv], n)]);
      }
    }
    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  advection_le_3rd
 *
 *  Kernel driver
 *
 *****************************************************************************/

__host__ int advection_le_3rd(advflux_t * flux, hydro_t * hydro,
			      field_t * field) {
  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  lees_edw_t * letarget = NULL;

  assert(flux);
  assert(hydro);
  assert(field->data);

  cs_nlocal(flux->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(flux->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  if (flux->le) {
    lees_edw_target(flux->le, &letarget);
    tdpLaunchKernel(advection_le_3rd_kernel_v, nblk, ntpb, 0, 0,
		    ctxt->target,
		    letarget, flux->target, hydro->target, field->target);
  }
  else {
    tdpLaunchKernel(advection_3rd_kernel_v, nblk, ntpb, 0, 0,
		    ctxt->target, flux->target, hydro->target, field->target);
  }
  tdpDeviceSynchronize();

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  advection_le_3rd_kernel_v
 *
 *  Advective fluxes, allowing for LE planes.
 *
 *  In fact, formally second order wave-number extended scheme
 *  folowing Li, J. Comp. Phys. 113 235--255 (1997).
 *
 *  The stencil is three points, biased in upwind direction,
 *  with weights a1, a2, a3.
 *
 *****************************************************************************/

__global__ void advection_le_3rd_kernel_v(kernel_ctxt_t * ktx,
					  lees_edw_t * le,
					  advflux_t * flux, 
					  hydro_t * hydro,
					  field_t * fld) {
  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(le);
  assert(flux);
  assert(hydro);
  assert(fld);

  kiter = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiter, NSIMDVL) {

    int ia, iv, n;
    int ic[NSIMDVL], jc[NSIMDVL], kc[NSIMDVL];
    int maskv[NSIMDVL];
    int index0[NSIMDVL], index1[NSIMDVL], index2[NSIMDVL], index3[NSIMDVL];
    int m2[NSIMDVL], m1[NSIMDVL], p1[NSIMDVL], p2[NSIMDVL];
    double u0[3][NSIMDVL], u1[3][NSIMDVL], u[NSIMDVL];

    double fd1[NSIMDVL];
    double fd2[NSIMDVL];
    double fd3[NSIMDVL];
    
    const double a1 = -0.213933;
    const double a2 =  0.927865;
    const double a3 =  0.286067;

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_coords_index_v(ktx, ic, jc, kc, index0);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u0[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index0[iv],ia)];
      }
    }

    /* Flux at west face (between icm1 and ic) */

    for_simd_v(iv, NSIMDVL) m2[iv] = lees_edw_ic_to_buff(le, ic[iv], -2*maskv[iv]);
    for_simd_v(iv, NSIMDVL) m1[iv] = lees_edw_ic_to_buff(le, ic[iv], -1*maskv[iv]);
    for_simd_v(iv, NSIMDVL) p1[iv] = lees_edw_ic_to_buff(le, ic[iv], +1*maskv[iv]);
    for_simd_v(iv, NSIMDVL) p2[iv] = lees_edw_ic_to_buff(le, ic[iv], +2*maskv[iv]);

    lees_edw_index_v(le, m2, jc, kc, index2);
    lees_edw_index_v(le, m1, jc, kc, index1);
    lees_edw_index_v(le, p1, jc, kc, index3);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index1[iv],ia)];
      }
    }

    for_simd_v(iv, NSIMDVL) u[iv] = 0.5*maskv[iv]*(u0[X][iv] + u1[X][iv]);

    for (n = 0; n < fld->nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	if (u[iv] > 0.0) {
	  fd1[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index2[iv],n)];
	  fd2[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index1[iv],n)];
	  fd3[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index0[iv],n)];
	}
	else {
	  fd1[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index3[iv],n)];
	  fd2[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index0[iv],n)];
	  fd3[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index1[iv],n)];
	}
      }

      for_simd_v(iv, NSIMDVL) {
	flux->fw[addr_rank1(flux->nsite, flux->nf, index0[iv], n)] =
	  u[iv]*(a1*fd1[iv] + a2*fd2[iv] + a3*fd3[iv]);
      }
    }
	
    /* east face (ic and icp1) */

    lees_edw_index_v(le, p2, jc, kc, index2);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index3[iv],ia)];
      }
    }

    for_simd_v(iv, NSIMDVL) u[iv] = 0.5*maskv[iv]*(u0[X][iv] + u1[X][iv]);

    for (n = 0; n < fld->nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	if (u[iv] < 0.0) {
	  fd1[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index2[iv],n)];
	  fd2[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index3[iv],n)];
	  fd3[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index0[iv],n)];
	}
	else {
	  fd1[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index1[iv],n)];
	  fd2[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index0[iv],n)];
	  fd3[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index3[iv],n)];
	}
      }

      for_simd_v(iv, NSIMDVL) {
	flux->fe[addr_rank1(flux->nsite,flux->nf,index0[iv],n)] =
	  u[iv]*(a1*fd1[iv] + a2*fd2[iv] + a3*fd3[iv]);
      }  
    }


    /* y direction: jc+1 or ignore */

    for_simd_v(iv, NSIMDVL) m1[iv] = jc[iv] - 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p1[iv] = jc[iv] + 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p2[iv] = jc[iv] + 2*maskv[iv];

    lees_edw_index_v(le, ic, m1, kc, index3);
    lees_edw_index_v(le, ic, p1, kc, index1);
    lees_edw_index_v(le, ic, p2, kc, index2);
 
    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(fld->nsites,NHDIM,index1[iv],ia)];
      }
    }

    for_simd_v(iv, NSIMDVL) u[iv] = 0.5*maskv[iv]*(u0[Y][iv] + u1[Y][iv]);

    for (n = 0; n < fld->nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	if (u[iv] < 0.0) {
	  fd1[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index2[iv],n)];
	  fd2[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index1[iv],n)];
	  fd3[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index0[iv],n)];
	}
	else {
	  fd1[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index3[iv],n)];
	  fd2[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index0[iv],n)];
	  fd3[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index1[iv],n)];
	}
      }

      for_simd_v(iv, NSIMDVL) {
	flux->fy[addr_rank1(flux->nsite,flux->nf,index0[iv],n)] =
	  u[iv]*(a1*fd1[iv] + a2*fd2[iv] + a3*fd3[iv]);
      }
    }
	
    /* z direction: kc+1 or ignore */

    for_simd_v(iv, NSIMDVL) m1[iv] = kc[iv] - 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p1[iv] = kc[iv] + 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p2[iv] = kc[iv] + 2*maskv[iv];

    lees_edw_index_v(le, ic, jc, m1, index3);
    lees_edw_index_v(le, ic, jc, p1, index1);
    lees_edw_index_v(le, ic, jc, p2, index2);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index1[iv],ia)];
      }
    }

    for_simd_v(iv, NSIMDVL) u[iv] = 0.5*maskv[iv]*(u0[Z][iv] + u1[Z][iv]);

    for (n = 0; n < fld->nf; n++) {	    
      for_simd_v(iv, NSIMDVL) {
	if (u[iv] < 0.0) {
	  fd1[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index2[iv],n)];
	  fd2[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index1[iv],n)];
	  fd3[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index0[iv],n)];
	}
	else {
	  fd1[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index3[iv],n)];
	  fd2[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index0[iv],n)];
	  fd3[iv] = fld->data[addr_rank1(fld->nsites,fld->nf,index1[iv],n)];
	}
      }

      for_simd_v(iv, NSIMDVL) {
	flux->fz[addr_rank1(flux->nsite,flux->nf,index0[iv],n)] =
	  u[iv]*(a1*fd1[iv] + a2*fd2[iv] + a3*fd3[iv]);
      }
    }
    /* Next sites */
  }

  return;
}

/*****************************************************************************
 *
 *  advection_3rd_kernel_v
 *
 *  No Less-Edwards planes.
 *
 *****************************************************************************/

__global__ void advection_3rd_kernel_v(kernel_ctxt_t * ktx,
				       advflux_t * flux, 
				       hydro_t * hydro,
				       field_t * field) {

  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(flux);
  assert(hydro);
  assert(field);

  kiter = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiter, NSIMDVL) {

    int ia, iv;
    int n, nf;
    int ic[NSIMDVL], jc[NSIMDVL], kc[NSIMDVL];
    int maskv[NSIMDVL];
    int index0[NSIMDVL], index1[NSIMDVL], index2[NSIMDVL], index3[NSIMDVL];
    int m2[NSIMDVL], m1[NSIMDVL], p1[NSIMDVL], p2[NSIMDVL];
    double u0[3][NSIMDVL], u1[3][NSIMDVL], u[NSIMDVL];

    double fd1[NSIMDVL];
    double fd2[NSIMDVL];
    double fd3[NSIMDVL];
    
    const double a1 = -0.213933;
    const double a2 =  0.927865;
    const double a3 =  0.286067;

    nf = field->nf;

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_coords_index_v(ktx, ic, jc, kc, index0);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u0[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index0[iv],ia)];
      }
    }

    /* Flux at west face (between icm1 and ic) */

    for_simd_v(iv, NSIMDVL) m2[iv] = ic[iv] - 2*maskv[iv];
    for_simd_v(iv, NSIMDVL) m1[iv] = ic[iv] - 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p1[iv] = ic[iv] + 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p2[iv] = ic[iv] + 2*maskv[iv];

    kernel_coords_index_v(ktx, m2, jc, kc, index2);
    kernel_coords_index_v(ktx, m1, jc, kc, index1);
    kernel_coords_index_v(ktx, p1, jc, kc, index3);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index1[iv],ia)];
      }
    }

    for_simd_v(iv, NSIMDVL) u[iv] = 0.5*maskv[iv]*(u0[X][iv] + u1[X][iv]);

    for (n = 0; n < nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	if (u[iv] > 0.0) {
	  fd1[iv] = field->data[addr_rank1(field->nsites,nf,index2[iv],n)];
	  fd2[iv] = field->data[addr_rank1(field->nsites,nf,index1[iv],n)];
	  fd3[iv] = field->data[addr_rank1(field->nsites,nf,index0[iv],n)];
	}
	else {
	  fd1[iv] = field->data[addr_rank1(field->nsites,nf,index3[iv],n)];
	  fd2[iv] = field->data[addr_rank1(field->nsites,nf,index0[iv],n)];
	  fd3[iv] = field->data[addr_rank1(field->nsites,nf,index1[iv],n)];
	}
      }

      for_simd_v(iv, NSIMDVL) {
	flux->fw[addr_rank1(flux->nsite, flux->nf, index0[iv], n)] =
	  u[iv]*(a1*fd1[iv] + a2*fd2[iv] + a3*fd3[iv]);
      }
    }
	
    /* east face (ic and icp1) */

    kernel_coords_index_v(ktx, p2, jc, kc, index2);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index3[iv],ia)];
      }
    }

    for_simd_v(iv, NSIMDVL) u[iv] = 0.5*maskv[iv]*(u0[X][iv] + u1[X][iv]);

    for (n = 0; n < nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	if (u[iv] < 0.0) {
	  fd1[iv] = field->data[addr_rank1(field->nsites,nf,index2[iv],n)];
	  fd2[iv] = field->data[addr_rank1(field->nsites,nf,index3[iv],n)];
	  fd3[iv] = field->data[addr_rank1(field->nsites,nf,index0[iv],n)];
	}
	else {
	  fd1[iv] = field->data[addr_rank1(field->nsites,nf,index1[iv],n)];
	  fd2[iv] = field->data[addr_rank1(field->nsites,nf,index0[iv],n)];
	  fd3[iv] = field->data[addr_rank1(field->nsites,nf,index3[iv],n)];
	}
      }

      for_simd_v(iv, NSIMDVL) {
	flux->fe[addr_rank1(flux->nsite,flux->nf,index0[iv],n)] =
	  u[iv]*(a1*fd1[iv] + a2*fd2[iv] + a3*fd3[iv]);
      }  
    }


    /* y direction: jc+1 or ignore */

    for_simd_v(iv, NSIMDVL) m1[iv] = jc[iv] - 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p1[iv] = jc[iv] + 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p2[iv] = jc[iv] + 2*maskv[iv];

    kernel_coords_index_v(ktx, ic, m1, kc, index3);
    kernel_coords_index_v(ktx, ic, p1, kc, index1);
    kernel_coords_index_v(ktx, ic, p2, kc, index2);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(field->nsites,NHDIM,index1[iv],ia)];
      }
    }

    for_simd_v(iv, NSIMDVL) u[iv] = 0.5*maskv[iv]*(u0[Y][iv] + u1[Y][iv]);

    for (n = 0; n < nf; n++) {
      for_simd_v(iv, NSIMDVL) {
	if (u[iv] < 0.0) {
	  fd1[iv] = field->data[addr_rank1(field->nsites,nf,index2[iv],n)];
	  fd2[iv] = field->data[addr_rank1(field->nsites,nf,index1[iv],n)];
	  fd3[iv] = field->data[addr_rank1(field->nsites,nf,index0[iv],n)];
	}
	else {
	  fd1[iv] = field->data[addr_rank1(field->nsites,nf,index3[iv],n)];
	  fd2[iv] = field->data[addr_rank1(field->nsites,nf,index0[iv],n)];
	  fd3[iv] = field->data[addr_rank1(field->nsites,nf,index1[iv],n)];
	}
      }

      for_simd_v(iv, NSIMDVL) {
	flux->fy[addr_rank1(field->nsites,nf,index0[iv],n)] =
	  u[iv]*(a1*fd1[iv] + a2*fd2[iv] + a3*fd3[iv]);
      }
    }
	
    /* z direction: kc+1 or ignore */

    for_simd_v(iv, NSIMDVL) m1[iv] = kc[iv] - 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p1[iv] = kc[iv] + 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p2[iv] = kc[iv] + 2*maskv[iv];

    kernel_coords_index_v(ktx, ic, jc, m1, index3);
    kernel_coords_index_v(ktx, ic, jc, p1, index1);
    kernel_coords_index_v(ktx, ic, jc, p2, index2);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index1[iv],ia)];
      }
    }

    for_simd_v(iv, NSIMDVL) u[iv] = 0.5*maskv[iv]*(u0[Z][iv] + u1[Z][iv]);

    for (n = 0; n < nf; n++) {	    
      for_simd_v(iv, NSIMDVL) {
	if (u[iv] < 0.0) {
	  fd1[iv] = field->data[addr_rank1(field->nsites,nf,index2[iv],n)];
	  fd2[iv] = field->data[addr_rank1(field->nsites,nf,index1[iv],n)];
	  fd3[iv] = field->data[addr_rank1(field->nsites,nf,index0[iv],n)];
	}
	else {
	  fd1[iv] = field->data[addr_rank1(field->nsites,nf,index3[iv],n)];
	  fd2[iv] = field->data[addr_rank1(field->nsites,nf,index0[iv],n)];
	  fd3[iv] = field->data[addr_rank1(field->nsites,nf,index1[iv],n)];
	}
      }

      for_simd_v(iv, NSIMDVL) {
	flux->fz[addr_rank1(field->nsites,nf,index0[iv],n)] =
	  u[iv]*(a1*fd1[iv] + a2*fd2[iv] + a3*fd3[iv]);
      }
    }
    /* Next sites */
  }

  return;
}

/****************************************************************************
 *
 *  advection_le_4th
 *
 *  Advective fluxes, allowing for LE planes.
 *
 *  The stencil is four points.
 *
 ****************************************************************************/

static int advection_le_4th(advflux_t * flux, hydro_t * hydro, int nf,
			    double * f) {
  int nlocal[3];
  int ic, jc, kc;
  int n;
  int index0, index1, index2, index3;
  int icm2, icm1, icp1, icp2;
  double u0[3], u1[3], u;
  lees_edw_t * le = NULL;

  const double a1 = (1.0/16.0); /* Interpolation weight */
  const double a2 = (9.0/16.0); /* Interpolation weight */

  assert(flux);
  assert(hydro);
  assert(f);

  le = flux->le;

  lees_edw_nlocal(le, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm2 = lees_edw_ic_to_buff(le, ic, -2);
    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);
    icp2 = lees_edw_ic_to_buff(le, ic, +2);

    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(le, ic, jc, kc);
	hydro_u(hydro, index0, u0);

	/* west face (icm1 and ic) */

	index1 = lees_edw_index(le, icm1, jc, kc);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[X] + u1[X]);

	index2 = lees_edw_index(le, icm2, jc, kc);
	index3 = lees_edw_index(le, icp1, jc, kc);

	for (n = 0; n < nf; n++) {
	  flux->fw[addr_rank1(flux->nsite, nf, index0,  n)] =
	    u*(- a1*f[addr_rank1(flux->nsite, nf, index2, n)]
	       + a2*f[addr_rank1(flux->nsite, nf, index1, n)]
	       + a2*f[addr_rank1(flux->nsite, nf, index0, n)]
	       - a1*f[addr_rank1(flux->nsite, nf, index3, n)]);
	}

	/* east face */

	index1 = lees_edw_index(le, icp1, jc, kc);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[X] + u1[X]);

	index2 = lees_edw_index(le, icm1, jc, kc);
	index3 = lees_edw_index(le, icp2, jc, kc);

	for (n = 0; n < nf; n++) {
	  flux->fe[addr_rank1(flux->nsite, nf, index0, n)] =
	    u*(- a1*f[addr_rank1(flux->nsite, nf, index2, n)]
	       + a2*f[addr_rank1(flux->nsite, nf, index0, n)]
	       + a2*f[addr_rank1(flux->nsite, nf, index1, n)]
	       - a1*f[addr_rank1(flux->nsite, nf, index3, n)]);
	}

	/* y-direction */

	index1 = lees_edw_index(le, ic, jc+1, kc);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[Y] + u1[Y]);

	index2 = lees_edw_index(le, ic, jc-1, kc);
	index3 = lees_edw_index(le, ic, jc+2, kc);

	for (n = 0; n < nf; n++) {
	  flux->fy[addr_rank1(flux->nsite, nf, index0, n)] =
	    u*(- a1*f[addr_rank1(flux->nsite, nf, index2, n)]
	       + a2*f[addr_rank1(flux->nsite, nf, index0, n)]
	       + a2*f[addr_rank1(flux->nsite, nf, index1, n)]
	       - a1*f[addr_rank1(flux->nsite, nf, index3, n)]);
	}

	/* z-direction */

	index1 = lees_edw_index(le, ic, jc, kc+1);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[Z] + u1[Z]);

	index2 = lees_edw_index(le, ic, jc, kc-1);
	index3 = lees_edw_index(le, ic, jc, kc+2);

	for (n = 0; n < nf; n++) {
	  flux->fz[addr_rank1(flux->nsite, nf, index0, n)] =
	    u*(- a1*f[addr_rank1(flux->nsite, nf, index2, n)]
	       + a2*f[addr_rank1(flux->nsite, nf, index0, n)]
	       + a2*f[addr_rank1(flux->nsite, nf, index1, n)]
	       - a1*f[addr_rank1(flux->nsite, nf, index3, n)]);
	}

	/* Next interface. */
      }
    }
  }

  return 0;
}

/****************************************************************************
 *
 *  advection_le_5th
 *
 *  Advective fluxes, allowing for LE planes.
 *
 *  Formally fourth-order accurate wavenumber-extended scheme of
 *  Li, J. Comp. Phys. 133 235-255 (1997).
 *
 *  The stencil is five points, biased in the upwind direction,
 *  with weights a1--a5.
 *
 ****************************************************************************/

static int advection_le_5th(advflux_t * flux, hydro_t * hydro, int nf,
			    double * f) {
  int nlocal[3];
  int ic, jc, kc;
  int n;
  int index0, index1;
  int icm2, icm1, icp1, icp2, icm3, icp3;
  double u0[3], u1[3], u;
  lees_edw_t * le = NULL;

  const double a1 =  0.055453;
  const double a2 = -0.305147;
  const double a3 =  0.916054;
  const double a4 =  0.361520;
  const double a5 = -0.027880;

  assert(flux);
  assert(hydro);
  assert(f);

  le = flux->le;
  lees_edw_nlocal(le, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm3 = lees_edw_ic_to_buff(le, ic, -3);
    icm2 = lees_edw_ic_to_buff(le, ic, -2);
    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);
    icp2 = lees_edw_ic_to_buff(le, ic, +2);
    icp3 = lees_edw_ic_to_buff(le, ic, +3);

    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

        index0 = lees_edw_index(le, ic, jc, kc);
        hydro_u(hydro, index0, u0);

        /* west face (icm1 and ic) */

        index1 = lees_edw_index(le, icm1, jc, kc);
        hydro_u(hydro, index1, u1);
        u = 0.5*(u0[X] + u1[X]);

        if (u > 0.0) {
          for (n = 0; n < nf; n++) {
            flux->fw[addr_rank1(flux->nsite, nf, index0, n)] = u*
	      (a1*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icm3,jc,kc), n)] +
	       a2*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icm2,jc,kc), n)] +
               a3*f[addr_rank1(flux->nsite, nf, index1, n)] +
               a4*f[addr_rank1(flux->nsite, nf, index0, n)] +
	       a5*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icp1,jc,kc), n)]);
          }
        }
        else {
          for (n = 0; n < nf; n++) {
            flux->fw[addr_rank1(flux->nsite, nf, index0, n)] = u*
	      (a1*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icp2,jc,kc), n)] +
	       a2*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icp1,jc,kc), n)] +
               a3*f[addr_rank1(flux->nsite, nf, index0, n)] +
               a4*f[addr_rank1(flux->nsite, nf, index1, n)] +
	       a5*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icm2,jc,kc), n)]);
          }
	}

        /* east face */

        index1 = lees_edw_index(le, icp1, jc, kc);
        hydro_u(hydro, index1, u1);
        u = 0.5*(u0[X] + u1[X]);

        if (u < 0.0) {
          for (n = 0; n < nf; n++) {
            flux->fe[addr_rank1(flux->nsite, nf, index0, n)] = u*
	      (a1*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icp3,jc,kc), n)] +
	       a2*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icp2,jc,kc), n)] +
               a3*f[addr_rank1(flux->nsite, nf, index1, n)] +
               a4*f[addr_rank1(flux->nsite, nf, index0, n)] +
	       a5*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icm1,jc,kc), n)]);
          }
        }
        else {
          for (n = 0; n < nf; n++) {
            flux->fe[addr_rank1(flux->nsite, nf, index0, n)] = u*
	      (a1*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icm2,jc,kc), n)] +
	       a2*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icm1,jc,kc), n)] +
               a3*f[addr_rank1(flux->nsite, nf, index0, n)] +
               a4*f[addr_rank1(flux->nsite, nf, index1, n)] +
	       a5*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,icp2,jc,kc), n)]);
          }
        }

        /* y-direction */

        index1 = lees_edw_index(le, ic, jc+1, kc);
        hydro_u(hydro, index1, u1);
        u = 0.5*(u0[Y] + u1[Y]);

        if (u < 0.0) {
          for (n = 0; n < nf; n++) {
            flux->fy[addr_rank1(flux->nsite, nf, index0, n)] = u*
	      (a1*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc+3,kc), n)] +
	       a2*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc+2,kc), n)] +
               a3*f[addr_rank1(flux->nsite, nf, index1, n)] +
               a4*f[addr_rank1(flux->nsite, nf, index0, n)] +
	       a5*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc-1,kc), n)]);
          }
        }
        else {
          for (n = 0; n < nf; n++) {
            flux->fy[addr_rank1(flux->nsite, nf, index0, n)] = u*
	      (a1*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc-2,kc), n)] +
	       a2*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc-1,kc), n)] +
               a3*f[addr_rank1(flux->nsite, nf, index0, n)] +
               a4*f[addr_rank1(flux->nsite, nf, index1, n)] +
	       a5*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc+2,kc), n)]);
          }
        }

        /* z-direction */

        index1 = lees_edw_index(le, ic, jc, kc+1);
        hydro_u(hydro, index1, u1);
        u = 0.5*(u0[Z] + u1[Z]);

        if (u < 0.0) {
          for (n = 0; n < nf; n++) {
            flux->fz[addr_rank1(flux->nsite, nf, index0, n)] = u*
	      (a1*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc,kc+3), n)] +
	       a2*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc,kc+2), n)] +
               a3*f[addr_rank1(flux->nsite, nf, index1, n)] +
               a4*f[addr_rank1(flux->nsite, nf, index0, n)] +
	       a5*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc,kc-1), n)]);
          }
        }
        else {
          for (n = 0; n < nf; n++) {
            flux->fz[addr_rank1(flux->nsite, nf, index0, n)] = u*
	      (a1*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc,kc-2), n)] +
	       a2*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc,kc-1), n)] +
               a3*f[addr_rank1(flux->nsite, nf, index0, n)] +
               a4*f[addr_rank1(flux->nsite, nf, index1, n)] +
	       a5*f[addr_rank1(flux->nsite, nf, lees_edw_index(le,ic,jc,kc+2), n)]);
          }
        }

        /* Next interface. */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  advective_fluxes_d3qx
 *
 *  General routine for nf fields at starting address f.
 *  No Lees Edwards boundaries.
 *
 *  The storage of the field(s) for all the related routines is
 *  assumed to be f[index][nf], where index is the spatial index.
 *
 *****************************************************************************/

int advective_fluxes_d3qx(hydro_t * hydro, int nf, double * f, 
					double ** flx) {

  assert(hydro);
  assert(nf > 0);
  assert(f);
  assert(flx);

  advective_fluxes_2nd_d3qx(hydro, nf, f, flx);

  return 0;
}

/*****************************************************************************
 *
 *  advective_fluxes_2nd_d3qx
 *
 *  'Centred difference' advective fluxes. No LE planes.
 *  TODO: belongs to Nernst Planck.
 *
 *  Symmetric two-point stencil.
 *
 *****************************************************************************/

int advective_fluxes_2nd_d3qx(hydro_t * hydro, int nf, double * f, 
					double ** flx) {

  int nlocal[3];
  int ic, jc, kc, c;
  int n;
  int index0, index1;
  int nsites;
  double u0[3], u1[3], u;

  assert(hydro);
  assert(hydro->cs);
  assert(nf > 0);
  assert(f);
  assert(flx);

  cs_nlocal(hydro->cs, nlocal);
  cs_nsites(hydro->cs, &nsites);
  /* assert(nhalo >= 1); */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = cs_index(hydro->cs, ic, jc, kc);
	hydro_u(hydro, index0, u0);

        for (c = 1; c < PSI_NGRAD; c++) {

	  index1 = cs_index(hydro->cs, ic + psi_gr_cv[c][X], jc + psi_gr_cv[c][Y], kc + psi_gr_cv[c][Z]);
	  hydro_u(hydro, index1, u1);

	  u = 0.5*((u0[X] + u1[X])*psi_gr_cv[c][X] + (u0[Y] + u1[Y])*psi_gr_cv[c][Y] + (u0[Z] + u1[Z])*psi_gr_cv[c][Z]);

	  for (n = 0; n < nf; n++) {
	    flx[addr_rank1(nsites, nf, index0, n)][c - 1]
	      = u*0.5*(f[addr_rank1(nsites, nf, index1, n)]
		       + f[addr_rank1(nsites, nf, index0, n)]);
	  }
	}

	/* Next site */
      }
    }
  }

  return 0;
}
