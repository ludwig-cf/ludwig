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
 *  conservation of the order parameter when computing an update
 *  involving the divergence of fluxes.
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
 *  (c) 2010-2023  The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <stdlib.h>

#include "advection_s.h"
#include "hydro.h"
#include "timer.h"

/* Non-Lees Edwards implementation */

__global__ void advflux_cs_0th_kernel(advflux_t * flux);
__global__ void advflux_cs_1st_kernel(kernel_ctxt_t * ktx, advflux_t * flux,
				      hydro_t * hydro, field_t * field);
__global__ void advflux_cs_2nd_kernel(kernel_ctxt_t * ktx, advflux_t * flux,
				      hydro_t * hydro, field_t * field);
__global__ void advflux_cs_3rd_kernel_v(kernel_ctxt_t * ktx, advflux_t * flux, 
					hydro_t * hydro, field_t * field);

/* Lees-Edwards (via "advection_x()") */

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

__global__ void advflux_zero_kernel(kernel_ctxt_t * ktx, advflux_t * flx);

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
  int nsites = 0;
  double * tmp = NULL;
  advflux_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (advflux_t *) calloc(1, sizeof(advflux_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(advflux) failed\n");

  if (cs) cs_nsites(cs, &nsites);
  if (le) lees_edw_nsites(le, &nsites);

  obj->pe = pe;
  obj->cs = cs;
  obj->le = le;
  obj->nf = nf;
  obj->nsite = nsites;

  if (nf < 1 || nsites < 1 || INT_MAX/nf < nsites) {
    pe_info(pe,  "advflux_create: failure in int32_t indexing\n");
    return -1;
  }

  if (obj->le == NULL) {
    /* If no Lees Edwards, we only require an fx = fw */

    obj->fx = (double *) calloc((size_t) nsites*nf, sizeof(double));
    obj->fy = (double *) calloc((size_t) nsites*nf, sizeof(double));
    obj->fz = (double *) calloc((size_t) nsites*nf, sizeof(double));

    if (obj->fx == NULL) pe_fatal(pe, "calloc(advflux->fx) failed\n");
    if (obj->fy == NULL) pe_fatal(pe, "calloc(advflux->fy) failed\n");
    if (obj->fz == NULL) pe_fatal(pe, "calloc(advflux->fz) failed\n");
  }
  else {

    /* Require fe and fw */
    obj->fw = (double *) calloc((size_t) nsites*nf, sizeof(double));
    obj->fe = (double *) calloc((size_t) nsites*nf, sizeof(double));
    obj->fy = (double *) calloc((size_t) nsites*nf, sizeof(double));
    obj->fz = (double *) calloc((size_t) nsites*nf, sizeof(double));
    if (obj->fw == NULL) pe_fatal(pe, "calloc(advflux->fw) failed\n");
    if (obj->fe == NULL) pe_fatal(pe, "calloc(advflux->fe) failed\n");
    if (obj->fy == NULL) pe_fatal(pe, "calloc(advflux->fy) failed\n");
    if (obj->fz == NULL) pe_fatal(pe, "calloc(advflux->fz) failed\n");
  }

  /* Allocate target copy of structure (or alias) */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    cs_t * cstarget = NULL;
    lees_edw_t * letarget = NULL;
    size_t nsz = (size_t) nsites*nf*sizeof(double);

    tdpAssert(tdpMalloc((void **) &obj->target, sizeof(advflux_t)));
    tdpAssert(tdpMemset(obj->target, 0, sizeof(advflux_t)));

    if (obj->le == NULL) {
      /* Just fx */
      tdpAssert(tdpMalloc((void **) &tmp, nsz));
      tdpAssert(tdpMemcpy(&obj->target->fx, &tmp, sizeof(double *),
			  tdpMemcpyHostToDevice));
    }
    else {
      /* Lees Edwards fe, fw */
      tdpAssert(tdpMalloc((void **) &tmp, nsz));
      tdpAssert(tdpMemcpy(&obj->target->fe, &tmp, sizeof(double *),
			  tdpMemcpyHostToDevice));
      tdpAssert(tdpMalloc((void **) &tmp, nsz));
      tdpAssert(tdpMemcpy(&obj->target->fw, &tmp, sizeof(double *),
			  tdpMemcpyHostToDevice));
    }

    tdpAssert(tdpMalloc((void **) &tmp, nsz));
    tdpAssert(tdpMemcpy(&obj->target->fy, &tmp, sizeof(double *),
			tdpMemcpyHostToDevice));

    tdpAssert(tdpMalloc((void **) &tmp, nsz));
    tdpAssert(tdpMemcpy(&obj->target->fz, &tmp, sizeof(double *),
			tdpMemcpyHostToDevice));

    tdpAssert(tdpMemcpy(&obj->target->nf, &obj->nf, sizeof(int),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&obj->target->nsite, &obj->nsite, sizeof(int),
			tdpMemcpyHostToDevice));

    /* Other target pointers */
    if (cs) cs_target(cs, &cstarget);
    if (le) lees_edw_target(le, &letarget);
    tdpAssert(tdpMemcpy(&obj->target->cs, &cstarget, sizeof(cs_t *),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&obj->target->le, &letarget, sizeof(lees_edw_t *),
			tdpMemcpyHostToDevice));
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
    if (obj->le == NULL) {
    tdpMemcpy(&tmp, &obj->target->fx, sizeof(double *), tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    }
    else {
    tdpMemcpy(&tmp, &obj->target->fe, sizeof(double *), tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &obj->target->fw, sizeof(double *), tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    }
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
 *  advflux_zero
 *
 *****************************************************************************/

__host__ int advflux_zero(advflux_t * flux) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(flux);

  cs_nlocal(flux->cs, nlocal);

  /* Limits */

  limits.imin = 0; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(flux->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(advflux_zero_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, flux->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  advflux_zero_kernel_v
 *
 *****************************************************************************/

__global__ void advflux_zero_kernel(kernel_ctxt_t * ktx, advflux_t * flux) {

  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(flux);

  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    int ia, index;
    int ic, jc, kc;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

    for (ia = 0; ia < flux->nf; ia++) {
      flux->fw[addr_rank1(flux->nsite, flux->nf, index, ia)] = 0.0;
      flux->fe[addr_rank1(flux->nsite, flux->nf, index, ia)] = 0.0;
      flux->fy[addr_rank1(flux->nsite, flux->nf, index, ia)] = 0.0;
      flux->fz[addr_rank1(flux->nsite, flux->nf, index, ia)] = 0.0;
    }
  }

  return;
}

/*****************************************************************************
 *
 *  advflux_memcpy
 *
 *****************************************************************************/

__host__ int advflux_memcpy(advflux_t * adv, tdpMemcpyKind flag) {

  int ndevice;

  assert(adv);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(adv->target == adv);
  }
  else {
    /* Copy data items */
    size_t nsz = (size_t) adv->nsite*adv->nf;
    double * fe = NULL;
    double * fw = NULL;
    double * fy = NULL;
    double * fz = NULL;

    tdpAssert(tdpMemcpy(&fe, &adv->target->fe, sizeof(double *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(&fw, &adv->target->fw, sizeof(double *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(&fy, &adv->target->fy, sizeof(double *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(&fz, &adv->target->fz, sizeof(double *),
			tdpMemcpyDeviceToHost));
    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpAssert(tdpMemcpy(fe, adv->fe, nsz*sizeof(double), flag));
      tdpAssert(tdpMemcpy(fw, adv->fw, nsz*sizeof(double), flag));
      tdpAssert(tdpMemcpy(fy, adv->fy, nsz*sizeof(double), flag));
      tdpAssert(tdpMemcpy(fz, adv->fz, nsz*sizeof(double), flag));
      break;
    case tdpMemcpyDeviceToHost:
      tdpAssert(tdpMemcpy(adv->fe, fe, nsz*sizeof(double), flag));
      tdpAssert(tdpMemcpy(adv->fw, fw, nsz*sizeof(double), flag));
      tdpAssert(tdpMemcpy(adv->fy, fy, nsz*sizeof(double), flag));
      tdpAssert(tdpMemcpy(adv->fz, fz, nsz*sizeof(double), flag));
      break;
    default:
      pe_fatal(adv->pe, "advflux_memcpy: Bad tdpMemcpyKind\n");
    }
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

  TIMER_start(ADVECTION_X_KERNEL);

  if (obj->le == NULL) {

    /* No Lees-Edwards planes: treat spearately */
    advflux_cs_compute(obj, hydro, field);

  }
  else {

    /* For given LE , and given order, compute fluxes */

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
      /* No kernel available yet, hence copies ... */
      advflux_memcpy(obj, tdpMemcpyDeviceToHost);
      advection_le_4th(obj, hydro, nf, field->data);
      advflux_memcpy(obj, tdpMemcpyHostToDevice);
      break;
    case 5:
      /* Ditto */
      advflux_memcpy(obj, tdpMemcpyDeviceToHost);
      advection_le_5th(obj, hydro, nf, field->data);
      advflux_memcpy(obj, tdpMemcpyHostToDevice);
      break; 
    default:
      pe_fatal(obj->pe, "Unexpected advection scheme order\n");
    }
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

    hydro_u(hydro, index0, u0);

    index1 = kernel_coords_index(ktx, icm1, jc, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[X] + u1[X]);

    index = index0;
    if (u > 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fw[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }


    /* east face (ic and icp1) */

    index1 = kernel_coords_index(ktx, icp1, jc, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[X] + u1[X]);

    index = index0;
    if (u < 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fe[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }

    /* y direction */

    index1 = kernel_coords_index(ktx, ic, jc+1, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[Y] + u1[Y]);

    index = index0;
    if (u < 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }

    /* z direction */

    index1 = kernel_coords_index(ktx, ic, jc, kc+1);
    hydro_u(hydro, index1, u1);

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

    int n;
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

    hydro_u(hydro, index0, u0);

    /* west face (ic - 1 and ic) */

    index1 = kernel_coords_index(ktx, icm1, jc, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[X] + u1[X]);

    for (n = 0; n < field->nf; n++) {
      flux->fw[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index1, n)]
       + field->data[addr_rank1(field->nsites, field->nf, index0, n)]);
    }

    /* east face (ic and ic + 1) */

    index1 = kernel_coords_index(ktx, icp1, jc, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[X] + u1[X]);

    for (n = 0; n < flux->nf; n++) {
      flux->fe[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index0, n)] +
	 field->data[addr_rank1(field->nsites, field->nf, index1, n)]);
    }

    /* y direction */

    index1 = kernel_coords_index(ktx, ic, jc+1, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[Y] + u1[Y]);

    for (n = 0; n < flux->nf; n++) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index0, n)] +
	 field->data[addr_rank1(field->nsites, field->nf, index1, n)]);
    }

    /* z direction */

    index1 = kernel_coords_index(ktx, ic, jc, kc+1);
    hydro_u(hydro, index1, u1);

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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index0[iv], ia);
	u0[ia][iv] = hydro->u->data[haddr];
      }

    }

    /* west face (ic - 1 and ic) */

    lees_edw_index_v(flux->le, m1, jc, kc, index1);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
    assert(0); /* Moved to advflux_cs_compute */
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index0[iv], ia);
	u0[ia][iv] = hydro->u->data[haddr];
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index3[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
 *  advflux_cs_compute
 *
 *  Kernel driver routine (no Lees Edwards SPBC).
 *
 *  So we compute exactly three fluxes: fx, fy, fz.
 *
 *****************************************************************************/

__host__ int advflux_cs_compute(advflux_t * flux, hydro_t * h, field_t * f) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(flux);
  assert(flux->fx);
  assert(h);
  assert(f);

  cs_nlocal(flux->cs, nlocal);

  /* Limits */

  limits.imin = 0; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(flux->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  switch (order_) {
  case 1:
    tdpLaunchKernel(advflux_cs_1st_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, flux->target, h->target, f->target);
    break;
  case 2:
    tdpLaunchKernel(advflux_cs_2nd_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, flux->target, h->target, f->target);
    break;
  case 3:
    tdpLaunchKernel(advflux_cs_3rd_kernel_v, nblk, ntpb, 0, 0,
		    ctxt->target, flux->target, h->target, f->target);
    break;
  default:
    pe_fatal(flux->pe, "advflux_cs_compute: Unexpected advection scheme\n");
  }

  tdpDeviceSynchronize();

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  advflux_cs_zero
 *
 *  Zero fluxes
 *
 *****************************************************************************/

__host__ int advflux_cs_zero(advflux_t * flux) {

  dim3 nblk, ntpb;

  assert(flux);
  assert(flux->fx);

  kernel_launch_param(flux->nsite, &nblk, &ntpb);

  tdpLaunchKernel(advflux_cs_0th_kernel, nblk, ntpb, 0, 0, flux->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  return 0;
}

/*****************************************************************************
 *
 *  advflux_cs_0th_kernel
 *
 *****************************************************************************/

__global__ void advflux_cs_0th_kernel(advflux_t * flux) {

  int kindex;

  assert(flux);

  for_simt_parallel(kindex, flux->nsite, 1) {
    int n1;

    for (n1 = 0; n1 < flux->nf; n1++) {
      flux->fx[addr_rank1(flux->nsite, flux->nf, kindex, n1)] = 0.0;
      flux->fy[addr_rank1(flux->nsite, flux->nf, kindex, n1)] = 0.0;
      flux->fz[addr_rank1(flux->nsite, flux->nf, kindex, n1)] = 0.0;
    }
  }

  return;
}

/*****************************************************************************
 *
 *  advflux_cs_1st_kernel
 *
 *  The advective fluxes are computed via first order upwind
 * 
 *  The following are set (as for all the upwind routines):
 *
 *  fx  ('east') is the flux in x-direction between cells ic, ic+1
 *  fy           is the flux in y-direction between cells jc, jc+1
 *  fz           is the flux in z-direction between cells kc, kc+1
 *
 *****************************************************************************/

__global__ void advflux_cs_1st_kernel(kernel_ctxt_t * ktx,
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

    int n;
    int ic, jc, kc;
    int index0, index1, index;
    double u0[3], u1[3], u;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index0 = kernel_coords_index(ktx, ic, jc, kc);
    hydro_u(hydro, index0, u0);

    /* x-direction (between ic and ic+1) */

    index1 = kernel_coords_index(ktx, ic+1, jc, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[X] + u1[X]);

    index = index0;
    if (u < 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fx[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }

    /* y direction */

    index1 = kernel_coords_index(ktx, ic, jc+1, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[Y] + u1[Y]);

    index = index0;
    if (u < 0.0) index = index1;

    for (n = 0; n < field->nf; n++) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0, n)]
	= u*field->data[addr_rank1(field->nsites, field->nf, index, n)];
    }

    /* z direction */

    index1 = kernel_coords_index(ktx, ic, jc, kc+1);
    hydro_u(hydro, index1, u1);

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
 *  advflux_cs_2nd_kernel
 *
 *  'Centred difference' advective fluxes.
 *
 *  Symmetric two-point stencil.
 *
 *  Non-vectorised version.
 *
 *****************************************************************************/

__global__ void advflux_cs_2nd_kernel(kernel_ctxt_t * ktx, advflux_t * flux,
				      hydro_t * hydro, field_t * field) {
  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(flux);
  assert(hydro);
  assert(field);

  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    int n;
    int ic, jc, kc;
    int index0, index1;
    double u, u0[3], u1[3];

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index0 = kernel_coords_index(ktx, ic, jc, kc);
    hydro_u(hydro, index0, u0);

    /* x-direction (between ic and ic + 1) */

    index1 = kernel_coords_index(ktx, ic+1, jc, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[X] + u1[X]);

    for (n = 0; n < flux->nf; n++) {
      flux->fx[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index0, n)] +
	 field->data[addr_rank1(field->nsites, field->nf, index1, n)]);
    }

    /* y direction */

    index1 = kernel_coords_index(ktx, ic, jc+1, kc);
    hydro_u(hydro, index1, u1);

    u = 0.5*(u0[Y] + u1[Y]);

    for (n = 0; n < flux->nf; n++) {
      flux->fy[addr_rank1(flux->nsite, flux->nf, index0, n)] = 0.5*u*
	(field->data[addr_rank1(field->nsites, field->nf, index0, n)] +
	 field->data[addr_rank1(field->nsites, field->nf, index1, n)]);
    }

    /* z direction */

    index1 = kernel_coords_index(ktx, ic, jc, kc+1);
    hydro_u(hydro, index1, u1);

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
 *  advflux_cs_3rd_kernel_v
 *
 *  No Less-Edwards planes. Vectorised version.
 *
 *****************************************************************************/

__global__ void advflux_cs_3rd_kernel_v(kernel_ctxt_t * ktx,
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
    int m1[NSIMDVL], p1[NSIMDVL], p2[NSIMDVL];
    double u0[3][NSIMDVL], u1[3][NSIMDVL], u[NSIMDVL];

    double fd1[NSIMDVL];
    double fd2[NSIMDVL];
    double fd3[NSIMDVL];
    
    const double a1 = -0.213933;
    const double a2 =  0.927865;
    const double a3 =  0.286067;

    nf = field->nf;

    /* Always require u(ic,jc,kc) */

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_coords_index_v(ktx, ic, jc, kc, index0);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	int haddr = addr_rank1(hydro->nsite, NHDIM, index0[iv], ia);
	u0[ia][iv] = hydro->u->data[haddr];
      }
    }

    /* x-direction (betwenn ic and ic+1) */

    for_simd_v(iv, NSIMDVL) m1[iv] = ic[iv] - 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p1[iv] = ic[iv] + 1*maskv[iv];
    for_simd_v(iv, NSIMDVL) p2[iv] = ic[iv] + 2*maskv[iv];

    kernel_coords_index_v(ktx, m1, jc, kc, index1);
    kernel_coords_index_v(ktx, p1, jc, kc, index3);
    kernel_coords_index_v(ktx, p2, jc, kc, index2);

    for (ia = 0; ia < NHDIM; ia++) {
      for_simd_v(iv, NSIMDVL) {
	int haddr = addr_rank1(hydro->nsite, NHDIM, index3[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
	flux->fx[addr_rank1(flux->nsite,flux->nf,index0[iv],n)] =
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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
	int haddr = addr_rank1(hydro->nsite, NHDIM, index1[iv], ia);
	u1[ia][iv] = hydro->u->data[haddr];
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

