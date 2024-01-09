/*****************************************************************************
 *
 *  blue_phase_beris_edwards.c
 *
 *  Time evolution for the blue phase tensor order parameter via the
 *  Beris-Edwards equation with fluctuations.
 *
 *  We have
 *
 *  d_t Q_ab + div . (u Q_ab) + S(W, Q) = -Gamma H_ab + xi_ab
 *
 *  where S(W, Q) allows for the rotation of rod-like molecules.
 *  W_ab is the velocity gradient tensor. H_ab is the molecular
 *  field.
 *
 *  S(W, Q) = (xi D_ab + Omega_ab)(Q_ab + (1/3) d_ab)
 *          + (Q_ab + (1/3) d_ab)(xiD_ab - Omega_ab)
 *          - 2xi(Q_ab + (1/3) d_ab) Tr (QW)
 *
 *  D_ab = (1/2) (W_ab + W_ba) and Omega_ab = (1/2) (W_ab - W_ba);
 *  the final term renders the whole thing traceless.
 *  xi is defined with the free energy.
 *
 *  The noise term xi_ab is treated following Bhattacharjee et al.
 *  J. Chem. Phys. 133 044112 (2010). We need to define five constant
 *  matrices T_ab; these are used in association with five random
 *  variates at each lattice site to generate consistent noise. The
 *  variance is 2 kT Gamma from fluctuation dissipation.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Davide Marenduzzo supplied the inspiration.
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "leesedwards.h"
#include "advection_bcs.h"
#include "blue_phase.h"
#include "blue_phase_beris_edwards.h"
#include "advection_s.h"
#include "colloids.h"
#include "timer.h"

__host__ int beris_edw_update_driver(beris_edw_t * be, field_t * fq,
				     field_grad_t * fq_grad,
				     hydro_t * hydro,
				     map_t * map, noise_t * noise); 
__host__ int beris_edw_fix_swd(beris_edw_t * be, colloids_info_t * cinfo,
			       hydro_t * hydro, map_t * map);
__host__ int beris_edw_update_host(beris_edw_t * be, fe_t * fe, field_t * fq,
				   hydro_t * hydro, advflux_t * flux,
				   map_t * map, noise_t * noise);
__host__ int beris_edw_h_driver(beris_edw_t * be, fe_t * fe);

__global__
void beris_edw_h_kernel_v(kernel_ctxt_t * ktx, beris_edw_t * be, fe_t * fe);
__global__
void beris_edw_kernel_v(kernel_ctxt_t * ktx, beris_edw_t * be,
			field_t * fq, field_grad_t * fqgrad,
			hydro_t * hydro, advflux_t * flux,
			map_t * map, noise_t * noise);
__global__
void beris_edw_fix_swd_kernel(kernel_ctxt_t * ktx, colloids_info_t * cinfo,
			      hydro_t * hydro, map_t * map, int noffsetx,
			      int noffsety, int noffsetz);

struct beris_edw_s {
  beris_edw_param_t * param;       /* Parameters */ 
  cs_t * cs;                       /* Coordinate object */
  lees_edw_t * le;                 /* Lees Edwards */
  advflux_t * flux;                /* Advective fluxes */
  int nall;                        /* Allocated sites */
  double * h;                      /* Molecular Field */

  beris_edw_t * target;            /* Target memory */
};

static __constant__ beris_edw_param_t static_param;

/*****************************************************************************
 *
 *  beris_edw_create
 *
 *  Create; one-time initialisation of the constant noise matrices is
 *  also here.
 *
 *****************************************************************************/

__host__ int beris_edw_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			      beris_edw_t ** pobj) {

  int ndevice;
  int nsites = 0;
  advflux_t * flx = NULL;
  beris_edw_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(le);
  assert(pobj);

  obj = (beris_edw_t *) calloc(1, sizeof(beris_edw_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(beris_edw) failed\n");

  obj->param = (beris_edw_param_t *) calloc(1, sizeof(beris_edw_param_t));
  if (obj->param == NULL) pe_fatal(pe, "calloc(beris_edw_param_t) failed\n");

  advflux_le_create(pe, cs, le, NQAB, &flx);
  assert(flx);

  lees_edw_nsites(le, &nsites);
  obj->nall = nsites;

  if (nsites < 1 || INT_MAX/NQAB < nsites) {
    pe_info(pe, "beris_edw_create: failure in int32_t indexing\n");
    return -1;
  }

  obj->h = (double *) calloc(nsites*NQAB, sizeof(double));
  assert(obj->h);

  obj->cs = cs;
  obj->le = le;
  obj->flux = flx;

  beris_edw_tmatrix(obj->param->tmatrix);

  /* Allocate a target copy, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    double * htmp = NULL;
    beris_edw_param_t * tmp;
    lees_edw_t * letarget = NULL;

    tdpAssert(tdpMalloc((void **) &obj->target, sizeof(beris_edw_t)));
    tdpAssert(tdpMemset(obj->target, 0, sizeof(beris_edw_t)));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(static_param));
    tdpAssert(tdpMemcpy(&obj->target->param, &tmp, sizeof(beris_edw_param_t *),
			tdpMemcpyHostToDevice));

    lees_edw_target(le, &letarget);
    tdpAssert(tdpMemcpy(&obj->target->le, &letarget, sizeof(lees_edw_t *),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&obj->target->flux, &flx->target, sizeof(advflux_t *),
			tdpMemcpyHostToDevice));

    tdpAssert(tdpMemcpy(&obj->target->nall, &obj->nall, sizeof(int),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMalloc((void **) &htmp, nsites*NQAB*sizeof(double)));
    tdpAssert(tdpMemcpy(&obj->target->h, &htmp, sizeof(double *),
			tdpMemcpyHostToDevice));
  }

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_free
 *
 *****************************************************************************/

__host__ int beris_edw_free(beris_edw_t * be) {

  int ndevice;

  assert(be);

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    double * htmp;

    tdpAssert(tdpMemcpy(&htmp, &be->target->h, sizeof(double *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpFree(htmp));
    tdpAssert(tdpFree(be->target));
  }

  advflux_free(be->flux);
  free(be->h);
  free(be->param);
  free(be);

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_param_commit
 *
 *****************************************************************************/

__host__ int beris_edw_param_commit(beris_edw_t * be) {

  double kt;
  physics_t * phys = NULL;

  assert(be);

  physics_ref(&phys);

  physics_kt(phys, &kt);
  be->param->var = sqrt(2.0*kt*be->param->gamma);

  tdpMemcpyToSymbol(tdpSymbol(static_param), be->param,
		    sizeof(beris_edw_param_t), 0, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_param_set
 *
 *****************************************************************************/

__host__ int beris_edw_param_set(beris_edw_t * be, beris_edw_param_t * vals) {

  assert(be);
  assert(vals);

  *be->param = *vals;

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_update
 *
 *  Driver routine for the update.
 *
 *  Compute advective fluxes (plus appropriate boundary conditions),
 *  and perform update for one time step.
 *
 *  hydro is allowed to be NULL, in which case we only have relaxational
 *  dynamics.
 *
 *****************************************************************************/

__host__ int beris_edw_update(beris_edw_t * be,
			      fe_t * fe,
			      field_t * fq,
			      field_grad_t * fq_grad,
			      hydro_t * hydro,
			      colloids_info_t * cinfo,
			      map_t * map,
			      noise_t * noise) {
  int nf;

  assert(be);
  assert(fq);
  assert(map);

  field_nf(fq, &nf);
  assert(nf == NQAB);

  if (hydro) {
    beris_edw_fix_swd(be, cinfo, hydro, map);
    hydro_lees_edwards(hydro);
    advection_x(be->flux, hydro, fq);
    advection_bcs_no_normal_flux(nf, be->flux, map);
  }

  beris_edw_h_driver(be, fe);
  beris_edw_update_driver(be, fq, fq_grad, hydro, map, noise);

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_update_host
 *
 *  Update q via Euler forward step. Note here we only update the
 *  5 independent elements of the Q tensor.
 *
 *  hydro is allowed to be NULL, in which case we only have relaxational
 *  dynamics.
 *
 *  This is a explicit (ic,jc,kc) loop version retained for reference.
 *
 *  TODO: The assert(0) in the noise section indicates this requires
 *        a test. The rest of the code is unaffected.
 *
 *****************************************************************************/

__host__ int beris_edw_update_host(beris_edw_t * be, fe_t * fe, field_t * fq,
				   hydro_t * hydro, advflux_t * flux,
				   map_t * map, noise_t * noise) {
  int ic, jc, kc;
  int ia, ib, id;
  int index, indexj, indexk;
  int nlocal[3];
  int status;
  int noise_on = 0;

  double q[3][3];
  double w[3][3];
  double d[3][3];
  double h[3][3];
  double s[3][3];
  double omega[3][3];
  double trace_qw;
  double xi;
  double gamma;

  double chi[NQAB], chi_qab[3][3];
  double tmatrix[3][3][NQAB] = {0};
  double var = 0.0;

  const double dt = 1.0;
  const double r3 = 1.0/3.0;
  KRONECKER_DELTA_CHAR(d_);

  assert(be);
  assert(fe);
  assert(fe->func->htensor);
  assert(fq);
  assert(flux);
  assert(map);

  xi = be->param->xi;
  gamma = be->param->gamma;
  var = be->param->var;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = 0.0;
      chi_qab[ia][ib] = 0.0;
    }
  }

  /* Get kBT, variance of noise and set basis of traceless,
   * symmetric matrices for contraction */

  if (noise) noise_present(noise, NOISE_QAB, &noise_on);
  if (noise_on) {
    assert(0); /* check noise kt */
    beris_edw_tmatrix(tmatrix);
  }

  lees_edw_nlocal(be->le, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = lees_edw_index(be->le, ic, jc, kc);

	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_tensor(fq, index, q);
	fe->func->htensor(fe, index, h);

	if (hydro) {

	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

	  hydro_u_gradient_tensor(hydro, ic, jc, kc, w);

	  trace_qw = 0.0;

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      trace_qw += q[ia][ib]*w[ib][ia];
	      d[ia][ib]     = 0.5*(w[ia][ib] + w[ib][ia]);
	      omega[ia][ib] = 0.5*(w[ia][ib] - w[ib][ia]);
	    }
	  }
	  
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      s[ia][ib] = -2.0*xi*(q[ia][ib] + r3*d_[ia][ib])*trace_qw;
	      for (id = 0; id < 3; id++) {
		s[ia][ib] +=
		  (xi*d[ia][id] + omega[ia][id])*(q[id][ib] + r3*d_[id][ib])
		+ (q[ia][id] + r3*d_[ia][id])*(xi*d[id][ib] - omega[id][ib]);
	      }
	    }
	  }
	}

	/* Fluctuating tensor order parameter */

	if (noise_on) {
	  noise_reap_n(noise, index, NQAB, chi);
	  for (id = 0; id < NQAB; id++) {
	    chi[id] = var*chi[id];
	  }

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      chi_qab[ia][ib] = 0.0;
	      for (id = 0; id < NQAB; id++) {
		chi_qab[ia][ib] += chi[id]*tmatrix[ia][ib][id];
	      }
	    }
	  }
	}

	/* Here's the full hydrodynamic update. */
	  
	indexj = lees_edw_index(be->le, ic, jc-1, kc);
	indexk = lees_edw_index(be->le, ic, jc, kc-1);

	q[X][X] += dt*(s[X][X] + gamma*h[X][X] + chi_qab[X][X]
		       - flux->fe[addr_rank1(flux->nsite, NQAB, index, XX)]
		       + flux->fw[addr_rank1(flux->nsite, NQAB, index, XX)]
		       - flux->fy[addr_rank1(flux->nsite, NQAB, index, XX)]
		       + flux->fy[addr_rank1(flux->nsite, NQAB, indexj, XX)]
		       - flux->fz[addr_rank1(flux->nsite, NQAB, index, XX)]
		       + flux->fz[addr_rank1(flux->nsite, NQAB, indexk, XX)]);

	q[X][Y] += dt*(s[X][Y] + gamma*h[X][Y] + chi_qab[X][Y]
		       - flux->fe[addr_rank1(flux->nsite, NQAB, index, XY)]
		       + flux->fw[addr_rank1(flux->nsite, NQAB, index, XY)]
		       - flux->fy[addr_rank1(flux->nsite, NQAB, index, XY)]
		       + flux->fy[addr_rank1(flux->nsite, NQAB, indexj, XY)]
		       - flux->fz[addr_rank1(flux->nsite, NQAB, index,  XY)]
		       + flux->fz[addr_rank1(flux->nsite, NQAB, indexk, XY)]);

	q[X][Z] += dt*(s[X][Z] + gamma*h[X][Z] + chi_qab[X][Z]
		       - flux->fe[addr_rank1(flux->nsite, NQAB, index, XZ)]
		       + flux->fw[addr_rank1(flux->nsite, NQAB, index, XZ)]
		       - flux->fy[addr_rank1(flux->nsite, NQAB, index, XZ)]
		       + flux->fy[addr_rank1(flux->nsite, NQAB, indexj, XZ)]
		       - flux->fz[addr_rank1(flux->nsite, NQAB, index, XZ)]
		       + flux->fz[addr_rank1(flux->nsite, NQAB, indexk, XZ)]);

	q[Y][Y] += dt*(s[Y][Y] + gamma*h[Y][Y] + chi_qab[Y][Y]
		       - flux->fe[addr_rank1(flux->nsite, NQAB, index, YY)]
		       + flux->fw[addr_rank1(flux->nsite, NQAB, index, YY)]
		       - flux->fy[addr_rank1(flux->nsite, NQAB, index, YY)]
		       + flux->fy[addr_rank1(flux->nsite, NQAB, indexj, YY)]
		       - flux->fz[addr_rank1(flux->nsite, NQAB, index, YY)]
		       + flux->fz[addr_rank1(flux->nsite, NQAB, indexk, YY)]);

	q[Y][Z] += dt*(s[Y][Z] + gamma*h[Y][Z] + chi_qab[Y][Z]
		       - flux->fe[addr_rank1(flux->nsite, NQAB, index, YZ)]
		       + flux->fw[addr_rank1(flux->nsite, NQAB, index, YZ)]
		       - flux->fy[addr_rank1(flux->nsite, NQAB, index, YZ)]
		       + flux->fy[addr_rank1(flux->nsite, NQAB, indexj, YZ)]
		       - flux->fz[addr_rank1(flux->nsite, NQAB, index, YZ)]
		       + flux->fz[addr_rank1(flux->nsite, NQAB, indexk, YZ)]);

	field_tensor_set(fq, index, q);

	/* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_update_driver
 *
 *  Update q via Euler forward step. Note here we only update the
 *  5 independent elements of the Q tensor.
 *
 *  hydro is allowed to be NULL, in which case we only have relaxational
 *  dynamics.
 *
 *****************************************************************************/

__host__ int beris_edw_update_driver(beris_edw_t * be,
				     field_t * fq,
				     field_grad_t * fq_grad,
				     hydro_t * hydro,
				     map_t * map,
				     noise_t * noise) {
  int ison;
  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  hydro_t * hydrotarget = NULL;
  noise_t * noisetarget = NULL;

  assert(be);
  assert(fq);
  assert(map);

  cs_nlocal(be->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(be->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  beris_edw_param_commit(be);
  if (hydro) hydrotarget = hydro->target;

  ison = 0;
  if (noise) noise_present(noise, NOISE_QAB, &ison);
  if (ison) noisetarget = noise;

  TIMER_start(BP_BE_UPDATE_KERNEL);

  tdpLaunchKernel(beris_edw_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, be->target, fq->target, fq_grad->target,
		  hydrotarget, be->flux->target, map->target, noisetarget);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  TIMER_stop(BP_BE_UPDATE_KERNEL);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_kernel
 *
 *****************************************************************************/

__global__
void beris_edw_kernel_v(kernel_ctxt_t * ktx, beris_edw_t * be,
			field_t * fq, field_grad_t * fqgrad,
			hydro_t * hydro, advflux_t * flux,
			map_t * map, noise_t * noise) {

  int kindex;
  __shared__ int kiterations;

  const double dt = 1.0;
  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d_);

  assert(ktx);
  assert(be);
  assert(fq);
  assert(fqgrad);
  assert(flux);
  assert(map);

  kiterations = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiterations, NSIMDVL) {

    int iv;

    int ia, ib, id;
    int index;
    int ic[NSIMDVL], jc[NSIMDVL], kc[NSIMDVL];
    int indexj[NSIMDVL], indexk[NSIMDVL];
    int maskv[NSIMDVL];
    int status = 0;

    double q[3][3][NSIMDVL];
    double w[3][3][NSIMDVL] = {0};
    double d[3][3][NSIMDVL];
    double s[3][3][NSIMDVL];

    double omega[3][3][NSIMDVL];
    double trace_qw[NSIMDVL];
    double chi[NQAB], chi_qab[3][3][NSIMDVL];
    double tr[NSIMDVL];

    index = kernel_baseindex(ktx, kindex);
    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	for_simd_v(iv, NSIMDVL) s[ia][ib][iv] = 0.0;
	for_simd_v(iv, NSIMDVL) chi_qab[ia][ib][iv] = 0.0;
      }
    }

    /* Mask out non-fluid sites. */

    /* No vectorisation here at the moment */
    for (iv = 0; iv < NSIMDVL; iv++) {
      if (maskv[iv]) map_status(map, index+iv, &status);
      if (maskv[iv] && status != MAP_FLUID) maskv[iv] = 0;
    }

    /* Expand q tensor */

    for_simd_v(iv, NSIMDVL) q[X][X][iv] = fq->data[addr_rank1(fq->nsites,NQAB,index+iv,XX)];
    for_simd_v(iv, NSIMDVL) q[X][Y][iv] = fq->data[addr_rank1(fq->nsites,NQAB,index+iv,XY)];
    for_simd_v(iv, NSIMDVL) q[X][Z][iv] = fq->data[addr_rank1(fq->nsites,NQAB,index+iv,XZ)];
    for_simd_v(iv, NSIMDVL) q[Y][X][iv] = q[X][Y][iv];
    for_simd_v(iv, NSIMDVL) q[Y][Y][iv] = fq->data[addr_rank1(fq->nsites,NQAB,index+iv,YY)];
    for_simd_v(iv, NSIMDVL) q[Y][Z][iv] = fq->data[addr_rank1(fq->nsites,NQAB,index+iv,YZ)];
    for_simd_v(iv, NSIMDVL) q[Z][X][iv] = q[X][Z][iv];
    for_simd_v(iv, NSIMDVL) q[Z][Y][iv] = q[Y][Z][iv];
    for_simd_v(iv, NSIMDVL) q[Z][Z][iv] = 0.0 - q[X][X][iv] - q[Y][Y][iv];


    if (hydro) {
      /* Velocity gradient tensor, symmetric and antisymmetric parts */

      int im1[NSIMDVL];
      int ip1[NSIMDVL];

      for_simd_v(iv, NSIMDVL) im1[iv] = lees_edw_ic_to_buff(be->le, ic[iv], -1);
      for_simd_v(iv, NSIMDVL) ip1[iv] = lees_edw_ic_to_buff(be->le, ic[iv], +1);

      for_simd_v(iv, NSIMDVL) im1[iv] = lees_edw_index(be->le, im1[iv], jc[iv], kc[iv]);
      for_simd_v(iv, NSIMDVL) ip1[iv] = lees_edw_index(be->le, ip1[iv], jc[iv], kc[iv]);

      for_simd_v(iv, NSIMDVL) { 
	if (maskv[iv]) {
	  w[X][X][iv] = 0.5*
	    (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], X)] -
	     hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], X)]);
	    }
	  }
      for_simd_v(iv, NSIMDVL) { 
	if (maskv[iv]) {
	  w[Y][X][iv] = 0.5*
	    (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], Y)] -
	     hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], Y)]);
	}
      }
      for_simd_v(iv, NSIMDVL) { 
	if (maskv[iv]) {
	  w[Z][X][iv] = 0.5*
	    (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], Z)] -
	     hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], Z)]);
	}
      }

      for_simd_v(iv, NSIMDVL) {
	im1[iv] = lees_edw_index(be->le, ic[iv], jc[iv] - maskv[iv], kc[iv]);
      }
      for_simd_v(iv, NSIMDVL) {
	ip1[iv] = lees_edw_index(be->le, ic[iv], jc[iv] + maskv[iv], kc[iv]);
      }
	  
      for_simd_v(iv, NSIMDVL) { 
	w[X][Y][iv] = 0.5*
	  (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], X)] -
	   hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], X)]);
      }
      for_simd_v(iv, NSIMDVL) { 
	w[Y][Y][iv] = 0.5*
	  (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], Y)] -
	   hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], Y)]);
      }
      for_simd_v(iv, NSIMDVL) { 
	w[Z][Y][iv] = 0.5*
	  (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], Z)] -
	   hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], Z)]);
      }

      for_simd_v(iv, NSIMDVL) {
	im1[iv] = lees_edw_index(be->le, ic[iv], jc[iv], kc[iv] - maskv[iv]);
      }
      for_simd_v(iv, NSIMDVL) {
	ip1[iv] = lees_edw_index(be->le, ic[iv], jc[iv], kc[iv] + maskv[iv]);
      }
	  
      for_simd_v(iv, NSIMDVL) { 
	w[X][Z][iv] = 0.5*
	  (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], X)] -
	   hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], X)]);
      }
      for_simd_v(iv, NSIMDVL) { 
	w[Y][Z][iv] = 0.5*
	  (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], Y)] -
	   hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], Y)]);
      }
      for_simd_v(iv, NSIMDVL) { 
	w[Z][Z][iv] = 0.5*
	  (hydro->u->data[addr_rank1(hydro->nsite, NHDIM, ip1[iv], Z)] -
	   hydro->u->data[addr_rank1(hydro->nsite, NHDIM, im1[iv], Z)]);
      }

      /* Enforce tracelessness */
	  
      for_simd_v(iv, NSIMDVL) tr[iv] = r3*(w[X][X][iv] + w[Y][Y][iv] + w[Z][Z][iv]);
      for_simd_v(iv, NSIMDVL) w[X][X][iv] -= tr[iv];
      for_simd_v(iv, NSIMDVL) w[Y][Y][iv] -= tr[iv];
      for_simd_v(iv, NSIMDVL) w[Z][Z][iv] -= tr[iv];

      for_simd_v(iv, NSIMDVL) trace_qw[iv] = 0.0;

      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  for_simd_v(iv, NSIMDVL) trace_qw[iv] += q[ia][ib][iv]*w[ib][ia][iv];
	  for_simd_v(iv, NSIMDVL) d[ia][ib][iv]     = 0.5*(w[ia][ib][iv] + w[ib][ia][iv]);
	  for_simd_v(iv, NSIMDVL) omega[ia][ib][iv] = 0.5*(w[ia][ib][iv] - w[ib][ia][iv]);
	}
      }
	  
      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  for_simd_v(iv, NSIMDVL) {
	    s[ia][ib][iv] =
	      -2.0*be->param->xi*(q[ia][ib][iv] + r3*d_[ia][ib])*trace_qw[iv];
	  }
	  for (id = 0; id < 3; id++) {
	    for_simd_v(iv, NSIMDVL) {
	      s[ia][ib][iv] +=
		(be->param->xi*d[ia][id][iv] + omega[ia][id][iv])
		*(q[id][ib][iv] + r3*d_[id][ib])
		+ (q[ia][id][iv] + r3*d_[ia][id])
		*(be->param->xi*d[id][ib][iv] - omega[id][ib][iv]);
	    }
	  }
	}
      }
    }

    /* Fluctuating tensor order parameter */

    if (noise) {

      for_simd_v(iv, NSIMDVL) {
	
	noise_reap_n(noise, index+iv, NQAB, chi);
	
	for (id = 0; id < NQAB; id++) {
	  chi[id] = be->param->var*chi[id];
	}
	
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    chi_qab[ia][ib][iv] = 0.0;
	    for (id = 0; id < NQAB; id++) {
	      chi_qab[ia][ib][iv] += chi[id]*be->param->tmatrix[ia][ib][id];
	    }
	  }
	}
      }
    }

    /* Here's the full hydrodynamic update. */
    /* The divergence of advective fluxes involves (jc-1) and (kc-1)
     * which are masked out if not a valid kernel site */

    for_simd_v(iv, NSIMDVL) {
      indexj[iv] = lees_edw_index(be->le, ic[iv], jc[iv] - maskv[iv], kc[iv]);
    }
    for_simd_v(iv, NSIMDVL) {
      indexk[iv] = lees_edw_index(be->le, ic[iv], jc[iv], kc[iv] - maskv[iv]);
    }

    for_simd_v(iv, NSIMDVL) {
      if (maskv[iv]) {
      q[X][X][iv] += dt*
	(s[X][X][iv]
	 + chi_qab[X][X][iv]
	 + be->param->gamma*be->h[addr_rank1(be->nall, NQAB, index+iv, XX)]
	 - flux->fe[addr_rank1(flux->nsite,NQAB,index + iv,XX)]
	 + flux->fw[addr_rank1(flux->nsite,NQAB,index + iv,XX)]
	 - flux->fy[addr_rank1(flux->nsite,NQAB,index + iv,XX)]
	 + flux->fy[addr_rank1(flux->nsite,NQAB,indexj[iv],XX)]
	 - flux->fz[addr_rank1(flux->nsite,NQAB,index + iv,XX)]
	 + flux->fz[addr_rank1(flux->nsite,NQAB,indexk[iv],XX)]);
      }
    }

    for_simd_v(iv, NSIMDVL) {
      if (maskv[iv]) {
      q[X][Y][iv] += dt*
	(s[X][Y][iv]
	 + chi_qab[X][Y][iv]
	 + be->param->gamma*be->h[addr_rank1(be->nall, NQAB, index+iv, XY)]
	 - flux->fe[addr_rank1(flux->nsite,NQAB,index + iv,XY)]
	 + flux->fw[addr_rank1(flux->nsite,NQAB,index + iv,XY)]
	 - flux->fy[addr_rank1(flux->nsite,NQAB,index + iv,XY)]
	 + flux->fy[addr_rank1(flux->nsite,NQAB,indexj[iv],XY)]
	 - flux->fz[addr_rank1(flux->nsite,NQAB,index + iv,XY)]
	 + flux->fz[addr_rank1(flux->nsite,NQAB,indexk[iv],XY)]);
      }
    }
	
    for_simd_v(iv, NSIMDVL) {
      if (maskv[iv]) {
      q[X][Z][iv] += dt*
	(s[X][Z][iv]
	 + chi_qab[X][Z][iv]
	 + be->param->gamma*be->h[addr_rank1(be->nall, NQAB, index+iv, XZ)]
	 - flux->fe[addr_rank1(flux->nsite,NQAB,index + iv,XZ)]
	 + flux->fw[addr_rank1(flux->nsite,NQAB,index + iv,XZ)]
	 - flux->fy[addr_rank1(flux->nsite,NQAB,index + iv,XZ)]
	 + flux->fy[addr_rank1(flux->nsite,NQAB,indexj[iv],XZ)]
	 - flux->fz[addr_rank1(flux->nsite,NQAB,index + iv,XZ)]
	 + flux->fz[addr_rank1(flux->nsite,NQAB,indexk[iv],XZ)]);
      }
    }
	
    for_simd_v(iv, NSIMDVL) {
      if (maskv[iv]) {
      q[Y][Y][iv] += dt*
	(s[Y][Y][iv]
	 + chi_qab[Y][Y][iv]
	 + be->param->gamma*be->h[addr_rank1(be->nall, NQAB, index+iv, YY)]
	 - flux->fe[addr_rank1(flux->nsite,NQAB,index + iv,YY)]
	 + flux->fw[addr_rank1(flux->nsite,NQAB,index + iv,YY)]
	 - flux->fy[addr_rank1(flux->nsite,NQAB,index + iv,YY)]
	 + flux->fy[addr_rank1(flux->nsite,NQAB,indexj[iv],YY)]
	 - flux->fz[addr_rank1(flux->nsite,NQAB,index + iv,YY)]
	 + flux->fz[addr_rank1(flux->nsite,NQAB,indexk[iv],YY)]);
      }
    }
	
    for_simd_v(iv, NSIMDVL) {
      if (maskv[iv]) {
      q[Y][Z][iv] += dt*
	(s[Y][Z][iv]
	 + chi_qab[Y][Z][iv]
	 + be->param->gamma*be->h[addr_rank1(be->nall, NQAB, index+iv, YZ)]
	 - flux->fe[addr_rank1(flux->nsite,NQAB,index + iv,YZ)]
	 + flux->fw[addr_rank1(flux->nsite,NQAB,index + iv,YZ)]
	 - flux->fy[addr_rank1(flux->nsite,NQAB,index + iv,YZ)]
	 + flux->fy[addr_rank1(flux->nsite,NQAB,indexj[iv],YZ)]
	 - flux->fz[addr_rank1(flux->nsite,NQAB,index + iv,YZ)]
	 + flux->fz[addr_rank1(flux->nsite,NQAB,indexk[iv],YZ)]);
      }
    }

    for_simd_v(iv, NSIMDVL) fq->data[addr_rank1(fq->nsites,NQAB,index+iv,XX)] = q[X][X][iv];
    for_simd_v(iv, NSIMDVL) fq->data[addr_rank1(fq->nsites,NQAB,index+iv,XY)] = q[X][Y][iv];
    for_simd_v(iv, NSIMDVL) fq->data[addr_rank1(fq->nsites,NQAB,index+iv,XZ)] = q[X][Z][iv];
    for_simd_v(iv, NSIMDVL) fq->data[addr_rank1(fq->nsites,NQAB,index+iv,YY)] = q[Y][Y][iv];
    for_simd_v(iv, NSIMDVL) fq->data[addr_rank1(fq->nsites,NQAB,index+iv,YZ)] = q[Y][Z][iv];

    /* Next sites. */
  }

  return;
}

/*****************************************************************************
 *
 *  beris_edw_tmatrix
 *
 *  Sets the elements of the traceless, symmetric base matrices
 *  following Bhattacharjee et al. There are five:
 *
 *  T^0_ab = sqrt(3/2) [ z_a z_b ]
 *  T^1_ab = sqrt(1/2) ( x_a x_b - y_a y_b ) a simple dyadic product
 *  T^2_ab = sqrt(2)   [ x_a y_b ]
 *  T^3_ab = sqrt(2)   [ x_a z_b ]
 *  T^4_ab = sqrt(2)   [ y_a z_b ]
 *
 *  Where x, y, z, are unit vectors, and the square brackets should
 *  be interpreted as
 *     [t_ab] = (1/2) (t_ab + t_ba) - (1/3) Tr (t_ab) d_ab.
 *
 *  Note the contraction T^i_ab T^j_ab = d_ij.
 *
 *****************************************************************************/

__host__ __device__ int beris_edw_tmatrix(double t[3][3][NQAB]) {

  int ia, ib, id;
  const double r3 = (1.0/3.0);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (id = 0; id < NQAB; id++) {
      	t[ia][ib][id] = 0.0;
      }
    }
  }

  t[X][X][XX] = sqrt(3.0/2.0)*(0.0 - r3);
  t[Y][Y][XX] = sqrt(3.0/2.0)*(0.0 - r3);
  t[Z][Z][XX] = sqrt(3.0/2.0)*(1.0 - r3);

  t[X][X][XY] = sqrt(1.0/2.0)*(1.0 - 0.0);
  t[Y][Y][XY] = sqrt(1.0/2.0)*(0.0 - 1.0);

  t[X][Y][XZ] = sqrt(2.0)*(1.0/2.0);
  t[Y][X][XZ] = t[X][Y][XZ];

  t[X][Z][YY] = sqrt(2.0)*(1.0/2.0); 
  t[Z][X][YY] = t[X][Z][YY];

  t[Y][Z][YZ] = sqrt(2.0)*(1.0/2.0);
  t[Z][Y][YZ] = t[Y][Z][YZ];

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_h_driver
 *
 *****************************************************************************/

__host__ int beris_edw_h_driver(beris_edw_t * be, fe_t * fe) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  fe_t * fe_target = NULL;

  assert(be);
  assert(fe);

  cs_nlocal(be->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  TIMER_start(TIMER_BE_MOL_FIELD);

  kernel_ctxt_create(be->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  fe->func->target(fe, &fe_target);

  tdpLaunchKernel(beris_edw_h_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, be->target, fe_target);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  TIMER_stop(TIMER_BE_MOL_FIELD);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_h_kernel_v
 *
 *  Compute and store the relevant molecular field
 *
 *****************************************************************************/

__global__ void beris_edw_h_kernel_v(kernel_ctxt_t * ktx, beris_edw_t * be,
				     fe_t * fe) {

  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(be);
  assert(fe);
  assert(fe->func->htensor_v);

  kiter = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiter, NSIMDVL) {

    int index;
    int iv;

    double h[3][3][NSIMDVL];

    index  = kernel_baseindex(ktx, kindex);

    fe->func->htensor_v(fe, index, h);

    for_simd_v(iv, NSIMDVL) {
      be->h[addr_rank1(be->nall, NQAB, index + iv, XX)] = h[X][X][iv];
    }
    for_simd_v(iv, NSIMDVL) {
      be->h[addr_rank1(be->nall, NQAB, index + iv, XY)] = h[X][Y][iv];
    }
    for_simd_v(iv, NSIMDVL) {
      be->h[addr_rank1(be->nall, NQAB, index + iv, XZ)] = h[X][Z][iv];
    }
    for_simd_v(iv, NSIMDVL) {
      be->h[addr_rank1(be->nall, NQAB, index + iv, YY)] = h[Y][Y][iv];
    }
    for_simd_v(iv, NSIMDVL) {
      be->h[addr_rank1(be->nall, NQAB, index + iv, YZ)] = h[Y][Z][iv];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  beris_fix_swd
 *
 *  The velocity gradient tensor used in the Beris-Edwards equations
 *  requires some approximation to the velocity at solid lattice sites.
 *
 *  This makes an approximation only at solid sites (so cannot change
 *  advective fluxes).
 *
 *****************************************************************************/

__host__
int beris_edw_fix_swd(beris_edw_t * be, colloids_info_t * cinfo,
		      hydro_t * hydro, map_t * map) {

  int nlocal[3];
  int noffset[3];
  int nextra;
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(be);
  assert(cinfo);
  assert(map);

  if (hydro == NULL) return 0;

  lees_edw_nlocal(be->le, nlocal);
  lees_edw_nlocal_offset(be->le, noffset);

  nextra = 1;   /* Limits extend 1 point into halo to permit a gradient */
  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(be->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(beris_edw_fix_swd_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, cinfo->target, hydro->target, map->target,
		  noffset[X], noffset[Y], noffset[Z]);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_fix_swd_kernel
 *
 *  Device note: cinfo is coming from unified memory.
 *
 *  This routine is doing two things:
 *    solid walls u -> zero
 *    colloid     u -> solid body rotation v + Omega x r_b
 *
 *  (These could be separated, particularly if moving walls wanted.)
 *
 *****************************************************************************/

__global__
void beris_edw_fix_swd_kernel(kernel_ctxt_t * ktx, colloids_info_t * cinfo,
			      hydro_t * hydro, map_t * map, int noffsetx,
			      int noffsety, int noffsetz) {

  int kindex;
  __shared__ int kiterations;

  assert(ktx);
  assert(cinfo);
  assert(hydro);
  assert(map);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {
    
    colloid_t * pc = NULL;

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);

    /* To include stationary walls. */
    if (map->status[index] != MAP_FLUID) {
      double u0[3] = {0.0, 0.0, 0.0};
      hydro_u_set(hydro, index, u0);
    }

    /* Colloids */
    if (cinfo->map_new) pc = cinfo->map_new[index];
 
    if (pc) {
      /* Set the lattice velocity here to the solid body
       * rotational velocity: v + Omega x r_b */

      double u[3];
      double rb[3];

      double x = noffsetx + ic;
      double y = noffsety + jc;
      double z = noffsetz + kc;
	
      rb[X] = x - pc->s.r[X];
      rb[Y] = y - pc->s.r[Y];
      rb[Z] = z - pc->s.r[Z];
	
      u[X] = pc->s.w[Y]*rb[Z] - pc->s.w[Z]*rb[Y];
      u[Y] = pc->s.w[Z]*rb[X] - pc->s.w[X]*rb[Z];
      u[Z] = pc->s.w[X]*rb[Y] - pc->s.w[Y]*rb[X];

      u[X] += pc->s.v[X];
      u[Y] += pc->s.v[Y];
      u[Z] += pc->s.v[Z];

      hydro_u_set(hydro, index, u);
    }
  }

  return;
}
