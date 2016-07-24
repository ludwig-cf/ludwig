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
 *  (c) 2010-2016  The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "field_s.h"
#include "advection_s.h"
#include "psi_gradients.h"
#include "hydro_s.h"
#include "timer.h"

__host__ int advection_le_1st(advflux_t * flux, hydro_t * hydro,
			    field_t * field);
static int advection_le_2nd(advflux_t * flux, hydro_t * hydro, int nf,
			    double * f);
__host__ int advection_le_3rd(advflux_t * flux, hydro_t * hydro,
			    field_t * field);
static int advection_le_4th(advflux_t * flux, hydro_t * hydro, int nf,
			    double * f);
static int advection_le_5th(advflux_t * flux, hydro_t * hydro, int nf,
			    double * f);

__global__
void advection_le_1st_kernel(kernel_ctxt_t * ktx, advflux_t * flux,
			     hydro_t * hydro, field_t * field);
__global__
void advection_le_3rd_kernel_v(kernel_ctxt_t * ktx, advflux_t * flux,
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
 *  advflux_create
 *
 *****************************************************************************/

__host__ int advflux_create(int nf, advflux_t ** pobj) {

  int ndevice;
  int nsites;
  double * tmp;
  advflux_t * obj = NULL;

  assert(pobj);

  obj = (advflux_t *) calloc(1, sizeof(advflux_t));
  if (obj == NULL) fatal("calloc(advflux) failed\n");

  nsites = le_nsites();
  obj->nf = nf;
  obj->nsite = nsites;

  obj->fe = (double *) calloc(nsites*nf, sizeof(double));
  obj->fw = (double *) calloc(nsites*nf, sizeof(double));
  obj->fy = (double *) calloc(nsites*nf, sizeof(double));
  obj->fz = (double *) calloc(nsites*nf, sizeof(double));

  if (obj->fe == NULL) fatal("calloc(advflux->fe) failed\n");
  if (obj->fw == NULL) fatal("calloc(advflux->fw) failed\n");
  if (obj->fy == NULL) fatal("calloc(advflux->fy) failed\n");
  if (obj->fz == NULL) fatal("calloc(advflux->fz) failed\n");


  /* Allocate target copy of structure (or alias) */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {

    targetMalloc((void **) &obj->target, sizeof(advflux_t));

    targetCalloc((void **) &tmp, nf*nsites*sizeof(double));
    copyToTarget(&obj->target->fe, &tmp, sizeof(double *)); 

    targetCalloc((void **) &tmp, nf*nsites*sizeof(double));
    copyToTarget(&obj->target->fw, &tmp, sizeof(double *)); 

    targetCalloc((void **) &tmp, nf*nsites*sizeof(double));
    copyToTarget(&obj->target->fy, &tmp, sizeof(double *)); 

    targetCalloc((void **) &tmp, nf*nsites*sizeof(double));
    copyToTarget(&obj->target->fz, &tmp, sizeof(double *)); 

    copyToTarget(&obj->target->nsite, &obj->nsite, sizeof(int));
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

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    copyFromTarget(&tmp, &obj->target->fe, sizeof(double *)); 
    targetFree(tmp);
    copyFromTarget(&tmp, &obj->target->fw, sizeof(double *)); 
    targetFree(tmp);
    copyFromTarget(&tmp, &obj->target->fy, sizeof(double *)); 
    targetFree(tmp);
    copyFromTarget(&tmp, &obj->target->fz, sizeof(double *)); 
    targetFree(tmp);
    targetFree(obj->target);
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

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {
    assert(0); /* Please fill me in */
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
    advection_le_2nd(obj, hydro, nf, field->data);
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
    fatal("Unexpected advection scheme order\n");
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

  coords_nlocal(nlocal);

  /* Limits */

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  __host_launch(advection_le_1st_kernel, nblk, ntpb, ctxt->target,
		flux->target, hydro->target, field->target);

  targetSynchronize();

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

  __target_simt_parallel_for(kindex, kiter, 1) {

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

#ifdef __NVCC__
    icm1 = ic-1;
    icp1 = ic+1;
#else
    /* enable LE planes (not yet supported for CUDA) */
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
#endif

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
 *  'Centred difference' advective fluxes, allowing for LE planes.
 *
 *  Symmetric two-point stencil.
 *
 *****************************************************************************/

static int advection_le_2nd(advflux_t * flux, hydro_t * hydro, int nf,
			    double * f) {
  int nlocal[3];
  int ic, jc, kc;
  int n;
  int index0, index1;
  int icp1, icm1;
  double u0[3], u1[3], u;

  assert(flux);
  assert(hydro);
  assert(f);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 1);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, jc, kc);
	hydro_u(hydro, index0, u0);

	/* west face (icm1 and ic) */

	index1 = le_site_index(icm1, jc, kc);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[X] + u1[X]);

	for (n = 0; n < nf; n++) {
	  flux->fw[addr_rank1(le_nsites(), nf, index0, n)]
	    = u*0.5*(f[addr_rank1(le_nsites(), nf, index1, n)]
		     + f[addr_rank1(le_nsites(), nf, index0, n)]);
	}

	/* east face (ic and icp1) */

	index1 = le_site_index(icp1, jc, kc);

	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[X] + u1[X]);

	for (n = 0; n < nf; n++) {
	  flux->fe[addr_rank1(le_nsites(), nf, index0, n)]
	    = u*0.5*(f[addr_rank1(le_nsites(), nf, index1, n)]
		     + f[addr_rank1(le_nsites(), nf, index0, n)]);
	}

	/* y direction */

	index1 = le_site_index(ic, jc+1, kc);

	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[Y] + u1[Y]);

	for (n = 0; n < nf; n++) {
	  flux->fy[addr_rank1(le_nsites(), nf, index0, n)]
	    = u*0.5*(f[addr_rank1(le_nsites(), nf, index1, n)]
		     + f[addr_rank1(le_nsites(), nf, index0, n)]);
	}

	/* z direction */

	index1 = le_site_index(ic, jc, kc+1);

	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[Z] + u1[Z]);

	for (n = 0; n < nf; n++) {
	  flux->fz[addr_rank1(le_nsites(), nf, index0, n)]
	    = u*0.5*(f[addr_rank1(le_nsites(), nf, index1, n)]
		     + f[addr_rank1(le_nsites(), nf, index0, n)]);
	}
	/* Next site */
      }
    }
  }

  return 0;
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

  assert(flux);
  assert(hydro);
  assert(field->data);
  assert(coords_nhalo() >= 2);

  coords_nlocal(nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  __host_launch(advection_le_3rd_kernel_v, nblk, ntpb, ctxt->target,
		flux->target, hydro->target, field->target);
  targetSynchronize();

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

  __target_simt_parallel_for(kindex, kiter, NSIMDVL) {

    int ia, iv;
    int n, nf, nsites;
    int ic[NSIMDVL], jc[NSIMDVL], kc[NSIMDVL];
    int maskv[NSIMDVL];
    int index0[NSIMDVL], index1[NSIMDVL], index2[NSIMDVL];
    int icm2[NSIMDVL], icm1[NSIMDVL], icp1[NSIMDVL], icp2[NSIMDVL];
    double u0[3][NSIMDVL], u1[3][NSIMDVL], u[NSIMDVL];

    double fd1[NSIMDVL];
    double fd2[NSIMDVL];
    
    const double a1 = -0.213933;
    const double a2 =  0.927865;
    const double a3 =  0.286067;

    nf = field->nf;
    nsites = field->nsites;

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_coords_index_v(ktx, ic, jc, kc, index0);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) {
	u0[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index0[iv],ia)];
      }
    }

    /* west face (icm1 and ic) */

#ifdef __NVCC__
    __targetILP__(iv) icm2[iv] = ic[iv]-2;
    __targetILP__(iv) icm1[iv] = ic[iv]-1;
    __targetILP__(iv) icp1[iv] = ic[iv]+1;
    __targetILP__(iv) icp2[iv] = ic[iv]+2;
#else
    /* enable LE planes (not yet supported for CUDA) */
    __targetILP__(iv) icm2[iv] = le_index_real_to_buffer(ic[iv], -2);
    __targetILP__(iv) icm1[iv] = le_index_real_to_buffer(ic[iv], -1);
    __targetILP__(iv) icp1[iv] = le_index_real_to_buffer(ic[iv], +1);
    __targetILP__(iv) icp2[iv] = le_index_real_to_buffer(ic[iv], +2);
#endif

    kernel_coords_index_v(ktx, icm1, jc, kc, index1);

    /* MAY WANT TO RE_INSTATE ia LOOP to get missing vector loads */
    __targetILP__(iv) {
      if (maskv[iv]) {
	u1[X][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index1[iv],X)];
      }
    }

    __targetILP__(iv) u[iv] = 0.5*(u0[X][iv] + u1[X][iv]);

    for (n = 0; n < nf; n++) {
      __targetILP__(iv) {
	if (maskv[iv]) {
	  if (u[iv] > 0.0) {
	    index2[iv] = le_site_index(icm2[iv], jc[iv], kc[iv]);
	    fd1[iv] = field->data[addr_rank1(nsites,nf,index1[iv],n)];
	    fd2[iv] = field->data[addr_rank1(nsites,nf,index0[iv],n)];
	  }
	  else {
	    index2[iv] = le_site_index(icp1[iv], jc[iv], kc[iv]);
	    fd1[iv] = field->data[addr_rank1(nsites,nf,index0[iv],n)];
	    fd2[iv] = field->data[addr_rank1(nsites,nf,index1[iv],n)];
	  }
	}
      }
	
      __targetILP__(iv) {
	if (maskv[iv]) {
	  flux->fw[addr_rank1(flux->nsite, flux->nf,index0[iv],n)] =
	    u[iv]*(a1*field->data[addr_rank1(nsites,nf,index2[iv],n)]
		   + a2*fd1[iv] + a3*fd2[iv]);
	}
      }
    }
	
    /* east face (ic and icp1) */

    kernel_coords_index_v(ktx, icp1, jc, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) {
	if (maskv[iv]) {
	  u1[ia][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index1[iv],ia)];
	}
      }
    }

    __targetILP__(iv) u[iv] = 0.5*(u0[X][iv] + u1[X][iv]);

    for (n = 0; n < nf; n++) {
      __targetILP__(iv) {
	if (maskv[iv]) {
	  if (u[iv] < 0.0) {
	    index2[iv] = le_site_index(icp2[iv], jc[iv], kc[iv]);
	    fd1[iv] = field->data[addr_rank1(nsites,nf,index1[iv],n)];
	    fd2[iv] = field->data[addr_rank1(nsites,nf,index0[iv],n)];
	  }
	  else {
	    index2[iv] = le_site_index(icm1[iv], jc[iv], kc[iv]);
	    fd1[iv] = field->data[addr_rank1(nsites,nf,index0[iv],n)];
	    fd2[iv] = field->data[addr_rank1(nsites,nf,index1[iv],n)];
	  }
	}
      }
	
      __targetILP__(iv) {
	if (maskv[iv]) {
	  flux->fe[addr_rank1(flux->nsite,flux->nf,index0[iv],n)] =
	    u[iv]*(a1*field->data[addr_rank1(nsites,nf,index2[iv],n)]
		   + a2*fd1[iv] + a3*fd2[iv]);
	}
      }  
    }


    /* y direction: jc+1 or ignore */
	
    __targetILP__(iv) {
      index1[iv] = le_site_index(ic[iv], jc[iv]+maskv[iv], kc[iv]);
    }

    __targetILP__(iv) {
      u1[Y][iv] = hydro->u[addr_rank1(nsites,NHDIM,index1[iv],Y)];
    }

    __targetILP__(iv) u[iv] = 0.5*(u0[Y][iv] + u1[Y][iv]);

    for (n = 0; n < nf; n++) {
      __targetILP__(iv) {
	if (u[iv] < 0.0) {
	  index2[iv] = le_site_index(ic[iv], jc[iv]+2*maskv[iv], kc[iv]);
	  fd1[iv] = field->data[addr_rank1(nsites,nf,index1[iv],n)];
	  fd2[iv] = field->data[addr_rank1(nsites,nf,index0[iv],n)];
	}
	else {
	  index2[iv] = le_site_index(ic[iv], jc[iv]-1*maskv[iv], kc[iv]);
	  fd1[iv] = field->data[addr_rank1(nsites,nf,index0[iv],n)];
	  fd2[iv] = field->data[addr_rank1(nsites,nf,index1[iv],n)];
	}
      }

      __targetILP__(iv) {
	if (maskv[iv]) {
	  flux->fy[addr_rank1(nsites,nf,index0[iv],n)] =
	    u[iv]*(a1*field->data[addr_rank1(nsites,nf,index2[iv],n)]
		   + a2*fd1[iv] + a3*fd2[iv]);
	}
      }
    }
	
    /* z direction: kc+1 or ignore */
	
    __targetILP__(iv) {
      index1[iv] = le_site_index(ic[iv], jc[iv], kc[iv]+maskv[iv]);
    }

    __targetILP__(iv) {
      u1[Z][iv] = hydro->u[addr_rank1(hydro->nsite,NHDIM,index1[iv],Z)];
    }

    __targetILP__(iv) u[iv] = 0.5*(u0[Z][iv] + u1[Z][iv]);

    for (n = 0; n < nf; n++) {	    
      __targetILP__(iv) {
	if (u[iv] < 0.0) {
	  index2[iv] = le_site_index(ic[iv], jc[iv], kc[iv]+2*maskv[iv]);
	  fd1[iv] = field->data[addr_rank1(nsites,nf,index1[iv],n)];
	  fd2[iv] = field->data[addr_rank1(nsites,nf,index0[iv],n)];
	}
	else {
	  index2[iv] = le_site_index(ic[iv], jc[iv], kc[iv]-1*maskv[iv]);
	  fd1[iv] = field->data[addr_rank1(nsites,nf,index0[iv],n)];
	  fd2[iv] = field->data[addr_rank1(nsites,nf,index1[iv],n)];
	}
      }

      __targetILP__(iv) {
	if (maskv[iv]) {
	  flux->fz[addr_rank1(nsites,nf,index0[iv],n)] =
	    u[iv]*(a1*field->data[addr_rank1(nsites,nf,index2[iv],n)]
		   + a2*fd1[iv] + a3*fd2[iv]);
	}
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
  int index0, index1;
  int icm2, icm1, icp1, icp2;
  double u0[3], u1[3], u;

  const double a1 = (1.0/16.0); /* Interpolation weight */
  const double a2 = (9.0/16.0); /* Interpolation weight */

  assert(flux);
  assert(hydro);
  assert(f);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  assert(0); /* SHIT NO TEST? */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm2 = le_index_real_to_buffer(ic, -2);
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    icp2 = le_index_real_to_buffer(ic, +2);

    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, jc, kc);
	hydro_u(hydro, index0, u0);

	/* west face (icm1 and ic) */

	index1 = le_site_index(icm1, jc, kc);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[X] + u1[X]);
	
	for (n = 0; n < nf; n++) {
	  flux->fw[nf*index0 + n] =
	    u*(- a1*f[nf*le_site_index(icm2, jc, kc) + n]
	       + a2*f[nf*index1 + n]
	       + a2*f[nf*index0 + n]
	       - a1*f[nf*le_site_index(icp1, jc, kc) + n]);
	}

	/* east face */

	index1 = le_site_index(icp1, jc, kc);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[X] + u1[X]);

	for (n = 0; n < nf; n++) {
	  flux->fe[nf*index0 + n] =
	    u*(- a1*f[nf*le_site_index(icm1, jc, kc) + n]
	       + a2*f[nf*index0 + n]
	       + a2*f[nf*index1 + n]
	       - a1*f[nf*le_site_index(icp2, jc, kc) + n]);
	}

	/* y-direction */

	index1 = le_site_index(ic, jc+1, kc);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[Y] + u1[Y]);

	for (n = 0; n < nf; n++) {
	  flux->fy[nf*index0 + n] =
	    u*(- a1*f[nf*le_site_index(ic, jc-1, kc) + n]
	       + a2*f[nf*index0 + n]
	       + a2*f[nf*index1 + n]
	       - a1*f[nf*le_site_index(ic, jc+2, kc) + n]);
	}

	/* z-direction */

	index1 = le_site_index(ic, jc, kc+1);
	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[Z] + u1[Z]);

	for (n = 0; n < nf; n++) {
	  flux->fz[nf*index0 + n] =
	    u*(- a1*f[nf*le_site_index(ic, jc, kc-1) + n]
	       + a2*f[nf*index0 + n]
	       + a2*f[nf*index1 + n]
	       - a1*f[nf*le_site_index(ic, jc, kc+2) + n]);
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
  int nsites;
  int nlocal[3];
  int ic, jc, kc;
  int n;
  int index0, index1;
  int icm2, icm1, icp1, icp2, icm3, icp3;
  double u0[3], u1[3], u;

  const double a1 =  0.055453;
  const double a2 = -0.305147;
  const double a3 =  0.916054;
  const double a4 =  0.361520;
  const double a5 = -0.027880;

  assert(flux);
  assert(hydro);
  assert(f);

  nsites = le_nsites();
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm3 = le_index_real_to_buffer(ic, -3);
    icm2 = le_index_real_to_buffer(ic, -2);
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    icp2 = le_index_real_to_buffer(ic, +2);
    icp3 = le_index_real_to_buffer(ic, +3);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

        index0 = le_site_index(ic, jc, kc);
        hydro_u(hydro, index0, u0);

        /* west face (icm1 and ic) */

        index1 = le_site_index(icm1, jc, kc);
        hydro_u(hydro, index1, u1);
        u = 0.5*(u0[X] + u1[X]);

        if (u > 0.0) {
          for (n = 0; n < nf; n++) {
            flux->fw[addr_rank1(nsites, nf, index0, n)] = u*
	      (a1*f[addr_rank1(nsites, nf, le_site_index(icm3, jc, kc), n)] +
	       a2*f[addr_rank1(nsites, nf, le_site_index(icm2, jc, kc), n)] +
               a3*f[addr_rank1(nsites, nf, index1, n)] +
               a4*f[addr_rank1(nsites, nf, index0, n)] +
	       a5*f[addr_rank1(nsites, nf, le_site_index(icp1, jc, kc), n)]);
          }
        }
        else {
          for (n = 0; n < nf; n++) {
            flux->fw[addr_rank1(nsites, nf, index0, n)] = u*
	      (a1*f[addr_rank1(nsites, nf, le_site_index(icp2, jc, kc), n)] +
	       a2*f[addr_rank1(nsites, nf, le_site_index(icp1, jc, kc), n)] +
               a3*f[addr_rank1(nsites, nf, index0, n)] +
               a4*f[addr_rank1(nsites, nf, index1, n)] +
	       a5*f[addr_rank1(nsites, nf, le_site_index(icm2, jc, kc), n)]);
          }
	}

        /* east face */

        index1 = le_site_index(icp1, jc, kc);
        hydro_u(hydro, index1, u1);
        u = 0.5*(u0[X] + u1[X]);

        if (u < 0.0) {
          for (n = 0; n < nf; n++) {
            flux->fe[addr_rank1(nsites, nf, index0, n)] = u*
	      (a1*f[addr_rank1(nsites, nf, le_site_index(icp3, jc, kc), n)] +
	       a2*f[addr_rank1(nsites, nf, le_site_index(icp2, jc, kc), n)] +
               a3*f[addr_rank1(nsites, nf, index1, n)] +
               a4*f[addr_rank1(nsites, nf, index0, n)] +
	       a5*f[addr_rank1(nsites, nf, le_site_index(icm1, jc, kc), n)]);
          }
        }
        else {
          for (n = 0; n < nf; n++) {
            flux->fe[addr_rank1(nsites, nf, index0, n)] = u*
	      (a1*f[addr_rank1(nsites, nf, le_site_index(icm2, jc, kc), n)] +
	       a2*f[addr_rank1(nsites, nf, le_site_index(icm1, jc, kc), n)] +
               a3*f[addr_rank1(nsites, nf, index0, n)] +
               a4*f[addr_rank1(nsites, nf, index1, n)] +
	       a5*f[addr_rank1(nsites, nf, le_site_index(icp2, jc, kc), n)]);
          }
        }

        /* y-direction */

        index1 = le_site_index(ic, jc+1, kc);
        hydro_u(hydro, index1, u1);
        u = 0.5*(u0[Y] + u1[Y]);

        if (u < 0.0) {
          for (n = 0; n < nf; n++) {
            flux->fy[addr_rank1(nsites, nf, index0, n)] = u*
	      (a1*f[addr_rank1(nsites, nf, le_site_index(ic, jc+3, kc), n)] +
	       a2*f[addr_rank1(nsites, nf, le_site_index(ic, jc+2, kc), n)] +
               a3*f[addr_rank1(nsites, nf, index1, n)] +
               a4*f[addr_rank1(nsites, nf, index0, n)] +
	       a5*f[addr_rank1(nsites, nf, le_site_index(ic, jc-1, kc), n)]);
          }
        }
        else {
          for (n = 0; n < nf; n++) {
            flux->fy[addr_rank1(nsites, nf, index0, n)] = u*
	      (a1*f[addr_rank1(nsites, nf, le_site_index(ic, jc-2, kc), n)] +
	       a2*f[addr_rank1(nsites, nf, le_site_index(ic, jc-1, kc), n)] +
               a3*f[addr_rank1(nsites, nf, index0, n)] +
               a4*f[addr_rank1(nsites, nf, index1, n)] +
	       a5*f[addr_rank1(nsites, nf, le_site_index(ic, jc+2, kc), n)]);
          }
        }

        /* z-direction */

        index1 = le_site_index(ic, jc, kc+1);
        hydro_u(hydro, index1, u1);
        u = 0.5*(u0[Z] + u1[Z]);

        if (u < 0.0) {
          for (n = 0; n < nf; n++) {
            flux->fz[addr_rank1(nsites, nf, index0, n)] = u*
	      (a1*f[addr_rank1(nsites, nf, le_site_index(ic, jc, kc+3), n)] +
	       a2*f[addr_rank1(nsites, nf, le_site_index(ic, jc, kc+2), n)] +
               a3*f[addr_rank1(nsites, nf, index1, n)] +
               a4*f[addr_rank1(nsites, nf, index0, n)] +
	       a5*f[addr_rank1(nsites, nf, le_site_index(ic, jc, kc-1), n)]);
          }
        }
        else {
          for (n = 0; n < nf; n++) {
            flux->fz[addr_rank1(nsites, nf, index0, n)] = u*
	      (a1*f[addr_rank1(nsites, nf, le_site_index(ic, jc, kc-2), n)] +
	       a2*f[addr_rank1(nsites, nf, le_site_index(ic, jc, kc-1), n)] +
               a3*f[addr_rank1(nsites, nf, index0, n)] +
               a4*f[addr_rank1(nsites, nf, index1, n)] +
	       a5*f[addr_rank1(nsites, nf, le_site_index(ic, jc, kc+2), n)]);
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
 *  advective_fluxes
 *
 *  General routine for nf fields at starting address f.
 *  No Lees Edwards boundaries.
 *
 *  The storage of the field(s) for all the related routines is
 *  assumed to be f[index][nf], where index is the spatial index.
 *
 *****************************************************************************/

int advective_fluxes(hydro_t * hydro, int nf, double * f, double * fe,
		     double * fy, double * fz) {

  assert(hydro);
  assert(nf > 0);
  assert(f);
  assert(fe);
  assert(fy);
  assert(fz);
  assert(le_get_nplane_total() == 0);

  advective_fluxes_2nd(hydro, nf, f, fe, fy, fz);

  return 0;
}

/*****************************************************************************
 *
 *  advective_fluxes_2nd
 *
 *  'Centred difference' advective fluxes. No LE planes.
 *
 *  Symmetric two-point stencil.
 *
 *****************************************************************************/

int advective_fluxes_2nd(hydro_t * hydro, int nf, double * f, double * fe,
			 double * fy, double * fz) {
  int nlocal[3];
  int ic, jc, kc;
  int n;
  int index0, index1;
  double u0[3], u1[3], u;

  assert(0); /* SHIT NO TEST? */
  assert(hydro);
  assert(nf > 0);
  assert(f);
  assert(fe);
  assert(fy);
  assert(fz);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 1);
  assert(le_get_nplane_total() == 0);

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = coords_index(ic, jc, kc);
	hydro_u(hydro, index0, u0);

	/* east face (ic and icp1) */

	index1 = coords_index(ic+1, jc, kc);

	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[X] + u1[X]);

	for (n = 0; n < nf; n++) {
	  fe[nf*index0 + n] = u*0.5*(f[nf*index1 + n] + f[nf*index0 + n]);
	}

	/* y direction */

	index1 = coords_index(ic, jc+1, kc);

	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[Y] + u1[Y]);

	for (n = 0; n < nf; n++) {
	  fy[nf*index0 + n] = u*0.5*(f[nf*index1 + n] + f[nf*index0 + n]);
	}

	/* z direction */

	index1 = coords_index(ic, jc, kc+1);

	hydro_u(hydro, index1, u1);
	u = 0.5*(u0[Z] + u1[Z]);

	for (n = 0; n < nf; n++) {
	  fz[nf*index0 + n] = u*0.5*(f[nf*index1 + n] + f[nf*index0 + n]);
	}

	/* Next site */
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
  assert(le_get_nplane_total() == 0);

  advective_fluxes_2nd_d3qx(hydro, nf, f, flx);

  return 0;
}

/*****************************************************************************
 *
 *  advective_fluxes_2nd_d3qx
 *
 *  'Centred difference' advective fluxes. No LE planes.
 *
 *  Symmetric two-point stencil.
 *
 *****************************************************************************/

int advective_fluxes_2nd_d3qx(hydro_t * hydro, int nf, double * f, 
					double ** flx) {

  int nsites;
  int nlocal[3];
  int ic, jc, kc, c;
  int n;
  int index0, index1;
  double u0[3], u1[3], u;

  assert(hydro);
  assert(nf > 0);
  assert(f);
  assert(flx);
  assert(le_get_nplane_total() == 0);

  nsites = coords_nsites();
  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 1);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = coords_index(ic, jc, kc);
	hydro_u(hydro, index0, u0);

        for (c = 1; c < PSI_NGRAD; c++) {

	  index1 = coords_index(ic + psi_gr_cv[c][X], jc + psi_gr_cv[c][Y], kc + psi_gr_cv[c][Z]);
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
