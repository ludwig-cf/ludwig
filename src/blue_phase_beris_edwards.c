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
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Davide Marenduzzo supplied the inspiration.
 *
 *****************************************************************************/

#include <assert.h>
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
#include "free_energy_tensor.h"
#include "hydro_s.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "colloids_s.h"
#include "map_s.h"
#include "timer.h"

__host__ int beris_edw_update_driver(beris_edw_t * be, fe_t * fe, field_t * fq,
				     field_grad_t * fq_grad,
				     hydro_t * hydro,
				     map_t * map, noise_t * noise); 
__host__ int colloids_fix_swd(colloids_info_t * cinfo, hydro_t * hydro,
			      map_t * map);
__host__ int beris_edw_update_host(beris_edw_t * be, fe_t * fe, field_t * fq,
				   hydro_t * hydro, advflux_t * flux,
				   map_t * map, noise_t * noise);

__targetConst__ double tc_tmatrix[3][3][NQAB];

struct beris_edw_s {
  beris_edw_param_t * param;       /* Parameters */ 
  advflux_t * flux;                /* Advective fluxes */

  beris_edw_t * target;            /* Target memory */
};

static __constant__ beris_edw_param_t static_param;

/*****************************************************************************
 *
 *  beris_edw_create
 *
 *****************************************************************************/

__host__ int beris_edw_create(beris_edw_t ** pobj) {

  int ndevice;
  advflux_t * flx = NULL;
  beris_edw_t * obj = NULL;

  assert(pobj);

  obj = (beris_edw_t *) calloc(1, sizeof(beris_edw_t));
  if (obj == NULL) fatal("calloc(beris_edw) failed\n");

  obj->param = (beris_edw_param_t *) calloc(1, sizeof(beris_edw_param_t));
  if (obj->param == NULL) fatal("calloc(beris_edw_param_t) failed\n");

  advflux_create(NQAB, &flx);
  assert(flx);
  obj->flux = flx;

  /* Allocate a target copy, or alias */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    targetCalloc((void **) &obj->target, sizeof(beris_edw_t));
    targetConstAddress(&obj->target->param, static_param);
    assert(0); /* Awaiting a test */
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

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) targetFree(be->target);

  advflux_free(be->flux);
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

  assert(be);

  copyConstToTarget(&static_param, be->param, sizeof(beris_edw_param_t));

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_param_set
 *
 *****************************************************************************/

__host__ int beris_edw_param_set(beris_edw_t * be, beris_edw_param_t vals) {

  assert(be);

  *be->param = vals;

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
    colloids_fix_swd(cinfo, hydro, map);
    hydro_lees_edwards(hydro);
    advection_x(be->flux, hydro, fq);
    advection_bcs_no_normal_flux(nf, be->flux, map);
  }

  /* SHIT sort this out */
  /* beris_edw_update_driver(be, fe, fq, fq_grad, hydro, map, noise);*/
  beris_edw_update_host(be, fe, fq, hydro, be->flux, map, noise);

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
 *****************************************************************************/

__host__ int beris_edw_update_host(beris_edw_t * be, fe_t * fe, field_t * fq,
				   hydro_t * hydro, advflux_t * flux,
				   map_t * map, noise_t * noise) {
  int ic, jc, kc;
  int ia, ib, id;
  int index, indexj, indexk;
  int nlocal[3];
  int nsites;
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
  double tmatrix[3][3][NQAB];
  double kt, var = 0.0;

  const double dt = 1.0;

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
    physics_kt(&kt);
    beris_edw_tmatrix(tmatrix);
  }

  coords_nlocal(nlocal);
  nsites = le_nsites();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);

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
	      s[ia][ib] = -2.0*xi*(q[ia][ib] + r3_*d_[ia][ib])*trace_qw;
	      for (id = 0; id < 3; id++) {
		s[ia][ib] +=
		  (xi*d[ia][id] + omega[ia][id])*(q[id][ib] + r3_*d_[id][ib])
		+ (q[ia][id] + r3_*d_[ia][id])*(xi*d[id][ib] - omega[id][ib]);
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
	  
	indexj = le_site_index(ic, jc-1, kc);
	indexk = le_site_index(ic, jc, kc-1);

	q[X][X] += dt*(s[X][X] + gamma*h[X][X] + chi_qab[X][X]
		       - flux->fe[addr_rank1(nsites, NQAB, index, XX)]
		       + flux->fw[addr_rank1(nsites, NQAB, index, XX)]
		       - flux->fy[addr_rank1(nsites, NQAB, index, XX)]
		       + flux->fy[addr_rank1(nsites, NQAB, indexj, XX)]
		       - flux->fz[addr_rank1(nsites, NQAB, index, XX)]
		       + flux->fz[addr_rank1(nsites, NQAB, indexk, XX)]);

	q[X][Y] += dt*(s[X][Y] + gamma*h[X][Y] + chi_qab[X][Y]
		       - flux->fe[addr_rank1(nsites, NQAB, index, XY)]
		       + flux->fw[addr_rank1(nsites, NQAB, index, XY)]
		       - flux->fy[addr_rank1(nsites, NQAB, index, XY)]
		       + flux->fy[addr_rank1(nsites, NQAB, indexj, XY)]
		       - flux->fz[addr_rank1(nsites, NQAB, index,  XY)]
		       + flux->fz[addr_rank1(nsites, NQAB, indexk, XY)]);

	q[X][Z] += dt*(s[X][Z] + gamma*h[X][Z] + chi_qab[X][Z]
		       - flux->fe[addr_rank1(nsites, NQAB, index, XZ)]
		       + flux->fw[addr_rank1(nsites, NQAB, index, XZ)]
		       - flux->fy[addr_rank1(nsites, NQAB, index, XZ)]
		       + flux->fy[addr_rank1(nsites, NQAB, indexj, XZ)]
		       - flux->fz[addr_rank1(nsites, NQAB, index, XZ)]
		       + flux->fz[addr_rank1(nsites, NQAB, indexk, XZ)]);

	q[Y][Y] += dt*(s[Y][Y] + gamma*h[Y][Y] + chi_qab[Y][Y]
		       - flux->fe[addr_rank1(nsites, NQAB, index, YY)]
		       + flux->fw[addr_rank1(nsites, NQAB, index, YY)]
		       - flux->fy[addr_rank1(nsites, NQAB, index, YY)]
		       + flux->fy[addr_rank1(nsites, NQAB, indexj, YY)]
		       - flux->fz[addr_rank1(nsites, NQAB, index, YY)]
		       + flux->fz[addr_rank1(nsites, NQAB, indexk, YY)]);

	q[Y][Z] += dt*(s[Y][Z] + gamma*h[Y][Z] + chi_qab[Y][Z]
		       - flux->fe[addr_rank1(nsites, NQAB, index, YZ)]
		       + flux->fw[addr_rank1(nsites, NQAB, index, YZ)]
		       - flux->fy[addr_rank1(nsites, NQAB, index, YZ)]
		       + flux->fy[addr_rank1(nsites, NQAB, indexj, YZ)]
		       - flux->fz[addr_rank1(nsites, NQAB, index, YZ)]
		       + flux->fz[addr_rank1(nsites, NQAB, indexk, YZ)]);

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
				     fe_t * fe,
				     field_t * fq,
				     field_grad_t * fq_grad,
				     hydro_t * hydro,
				     map_t * map,
				     noise_t * noise) {
  int nlocal[3];
  int noise_on = 0;

  double tmatrix[3][3][NQAB];
  double kt, var = 0.0;
  hydro_t * hydrotarget = NULL;
  fe_t * fetarget = NULL;

  __targetEntry__
    void beris_edw_kernel(beris_edw_t * be, fe_t * fe, field_t * fq,
			  field_grad_t * fqgrad,
			  hydro_t * hydro,
			  advflux_t * flux,
			  map_t * map,
			  int noise_on,
			  noise_t * noise);

  assert(be);
  assert(fe);
  assert(fq);
  assert(map);

  coords_nlocal(nlocal);

  /* Get kBT, variance of noise and set basis of traceless,
   * symmetric matrices for contraction */

  if (noise) noise_present(noise, NOISE_QAB, &noise_on);
  if (noise_on) {
    physics_kt(&kt);
    var = sqrt(2.0*kt*be->param->gamma);
    beris_edw_tmatrix(tmatrix);
  }

  int nhalo;
  nhalo = coords_nhalo();


  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;


  int nSites;

  nSites =Nall[X]*Nall[Y]*Nall[Z];

  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));
  beris_edw_param_commit(be);

  fe->func->target(fe, &fetarget);
  if (hydro) hydrotarget = hydro->target;

  TIMER_start(BP_BE_UPDATE_KERNEL);

  beris_edw_kernel __targetLaunch__(nSites) (be->target,
					     fetarget,
					     fq->target,
					     fq_grad->tcopy,
					     hydrotarget,
					     be->flux->target,
					     map->target,
					     noise_on, noise);
  
  targetSynchronize();

  TIMER_stop(BP_BE_UPDATE_KERNEL);

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_kernel
 *
 *****************************************************************************/

__targetEntry__
void beris_edw_kernel(beris_edw_t * be, fe_t * fe,
		      field_t * fq,
		      field_grad_t * fqgrad,
		      hydro_t * hydro,
		      advflux_t * flux,
		      map_t * map,
		      int noise_on,
		      noise_t * noise) {

  int baseIndex;
  const double r3 = (1.0/3.0);

  assert(fe);
  assert(fe->func->htensor_v);
  assert(fq);
  assert(fqgrad);
  assert(flux);

  __targetTLP__(baseIndex,tc_nSites) {

    int iv=0;
    int i;

    int ia, ib, id;
    int indexj[VVL], indexk[VVL];
    int status;

    double q[3][3][VVL];
    double dq[3][3][3][VVL];
    double dsq[3][3][VVL];
    double w[3][3][VVL];
    double d[3][3][VVL];
    double h[3][3][VVL];
    double s[3][3][VVL];

    double omega[3][3][VVL];
    double trace_qw[VVL];
    double chi[NQAB], chi_qab[3][3][VVL];


    const double dt = 1.0;

    int coords[3];


    targetCoords3D(coords,tc_Nall,baseIndex);
  
    
#if VVL == 1    
    /*restrict operation to the interior lattice sites*/ 
    targetCoords3D(coords,tc_Nall,baseIndex); 
    if (coords[0] >= (tc_nhalo) && 
	coords[1] >= (tc_nhalo) && 
	coords[2] >= (tc_nhalo) &&
	coords[0] < tc_Nall[X]-(tc_nhalo) &&  
	coords[1] < tc_Nall[Y]-(tc_nhalo)  &&  
	coords[2] < tc_Nall[Z]-(tc_nhalo) )
#endif
      
      {
	
	/* work out which sites in this chunk should be included */

	int includeSite[VVL];
	__targetILP__(iv) includeSite[iv]=0;
	
	int coordschunk[3][VVL];
		
	__targetILP__(iv){
	  for(i=0;i<3;i++){
	    targetCoords3D(coords,tc_Nall,baseIndex+iv);
	    coordschunk[i][iv]=coords[i];
	  }
	}

	__targetILP__(iv){
	  
	  if ((coordschunk[0][iv] >= (tc_nhalo) &&
	       coordschunk[1][iv] >= (tc_nhalo) &&
	       coordschunk[2][iv] >= (tc_nhalo) &&
	       coordschunk[0][iv] < tc_Nall[X]-(tc_nhalo) &&
	       coordschunk[1][iv] < tc_Nall[Y]-(tc_nhalo)  &&
	       coordschunk[2][iv] < tc_Nall[Z]-(tc_nhalo)))
	    
	    includeSite[iv]=1;
	}
	

      
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    __targetILP__(iv) s[ia][ib][iv] = 0.0;
	    __targetILP__(iv) chi_qab[ia][ib][iv] = 0.0;
	  }
	}


#ifndef __NVCC__
#if VVL == 1
	map_status(map, baseIndex, &status);
	if (status != MAP_FLUID) continue;
#else
	/* SHIT: can we really ignore this? */
	status = MAP_FLUID; /* Add to prevent compiler warning */
#endif
#endif /* else just calc all sites (and discard non-fluid results)*/

	/* calculate molecular field	*/

	int ia, ib;

	__targetILP__(iv) q[X][X][iv] = fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)];
	__targetILP__(iv) q[X][Y][iv] = fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)];
	__targetILP__(iv) q[X][Z][iv] = fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)];
	__targetILP__(iv) q[Y][X][iv] = q[X][Y][iv];
	__targetILP__(iv) q[Y][Y][iv] = fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)];
	__targetILP__(iv) q[Y][Z][iv] = fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)];
	__targetILP__(iv) q[Z][X][iv] = q[X][Z][iv];
	__targetILP__(iv) q[Z][Y][iv] = q[Y][Z][iv];
	__targetILP__(iv) q[Z][Z][iv] = 0.0 - q[X][X][iv] - q[Y][Y][iv];


	for (ia = 0; ia < NVECTOR; ia++) {
	  __targetILP__(iv) dq[ia][X][X][iv] = fqgrad->grad[addr_rank2(tc_nSites,NQAB,NVECTOR,baseIndex+iv,XX,ia)];
	  __targetILP__(iv) dq[ia][X][Y][iv] = fqgrad->grad[addr_rank2(tc_nSites,NQAB,NVECTOR,baseIndex+iv,XY,ia)];
	  __targetILP__(iv) dq[ia][X][Z][iv] = fqgrad->grad[addr_rank2(tc_nSites,NQAB,NVECTOR,baseIndex+iv,XZ,ia)];
	  __targetILP__(iv) dq[ia][Y][X][iv] = dq[ia][X][Y][iv];
	  __targetILP__(iv) dq[ia][Y][Y][iv] = fqgrad->grad[addr_rank2(tc_nSites,NQAB,NVECTOR,baseIndex+iv,YY,ia)];
	  __targetILP__(iv) dq[ia][Y][Z][iv] = fqgrad->grad[addr_rank2(tc_nSites,NQAB,NVECTOR,baseIndex+iv,YZ,ia)];
	  __targetILP__(iv) dq[ia][Z][X][iv] = dq[ia][X][Z][iv];
	  __targetILP__(iv) dq[ia][Z][Y][iv] = dq[ia][Y][Z][iv];
	  __targetILP__(iv) dq[ia][Z][Z][iv] = 0.0 - dq[ia][X][X][iv] - dq[ia][Y][Y][iv];
	}
	
	__targetILP__(iv) dsq[X][X][iv] = fqgrad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)];
	__targetILP__(iv) dsq[X][Y][iv] = fqgrad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)];
	__targetILP__(iv) dsq[X][Z][iv] = fqgrad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)];
	__targetILP__(iv) dsq[Y][X][iv] = dsq[X][Y][iv];
	__targetILP__(iv) dsq[Y][Y][iv] = fqgrad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)];
	__targetILP__(iv) dsq[Y][Z][iv] = fqgrad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)];
	__targetILP__(iv) dsq[Z][X][iv] = dsq[X][Z][iv];
	__targetILP__(iv) dsq[Z][Y][iv] = dsq[Y][Z][iv];
	__targetILP__(iv) dsq[Z][Z][iv] = 0.0 - dsq[X][X][iv] - dsq[Y][Y][iv];

	/* Compute the molecular field. */

	fe->func->htensor_v(fe, q, dq, dsq, h);

	if (hydro) {
	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

	  /* hydro_u_gradient_tensor(hydro, ic, jc, kc, w);
	   * inline above function
	   * TODO add lees edwards support*/

	  int im1[VVL];
	  int ip1[VVL];
	  __targetILP__(iv)  im1[iv] = targetIndex3D(coordschunk[X][iv]-1,coordschunk[Y][iv],coordschunk[Z][iv],tc_Nall);
	  __targetILP__(iv)  ip1[iv] = targetIndex3D(coordschunk[X][iv]+1,coordschunk[Y][iv],coordschunk[Z][iv],tc_Nall);
	  

	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[X][X][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],X)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],X)]);
	    }
	  }
	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[Y][X][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],Y)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],Y)]);
	    }
	  }
	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[Z][X][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],Z)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],Z)]);
	    }
	  }

	  __targetILP__(iv) im1[iv] = targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv]-1,coordschunk[Z][iv],tc_Nall);
	  __targetILP__(iv) ip1[iv] = targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv]+1,coordschunk[Z][iv],tc_Nall);
	  
	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[X][Y][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],X)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],X)]);
	    }
	  }
	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[Y][Y][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],Y)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],Y)]);
	    }
	  }
	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[Z][Y][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],Z)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],Z)]);
	    }
	  }
	  
	  __targetILP__(iv) im1[iv] = targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv],coordschunk[Z][iv]-1,tc_Nall);
	  __targetILP__(iv) ip1[iv] = targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv],coordschunk[Z][iv]+1,tc_Nall);
	  
	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[X][Z][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],X)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],X)]);
	    }
	  }
	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[Y][Z][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],Y)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],Y)]);
	    }
	  }
	  __targetILP__(iv) { 
	    if (includeSite[iv]) {
	      w[Z][Z][iv] = 0.5*
		(hydro->u[addr_rank1(tc_nSites,3,ip1[iv],Z)] -
		 hydro->u[addr_rank1(tc_nSites,3,im1[iv],Z)]);
	    }
	  }

	  /* Enforce tracelessness */
	  
	  double tr[VVL];
	  __targetILP__(iv) tr[iv] = r3*(w[X][X][iv] + w[Y][Y][iv] + w[Z][Z][iv]);
	  __targetILP__(iv) w[X][X][iv] -= tr[iv];
	  __targetILP__(iv) w[Y][Y][iv] -= tr[iv];
	  __targetILP__(iv) w[Z][Z][iv] -= tr[iv];


	  __targetILP__(iv) trace_qw[iv] = 0.0;

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      __targetILP__(iv) trace_qw[iv] += q[ia][ib][iv]*w[ib][ia][iv];
	      __targetILP__(iv) d[ia][ib][iv]     = 0.5*(w[ia][ib][iv] + w[ib][ia][iv]);
	      __targetILP__(iv) omega[ia][ib][iv] = 0.5*(w[ia][ib][iv] - w[ib][ia][iv]);
	    }
	  }
	  
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      __targetILP__(iv) s[ia][ib][iv] = -2.0*be->param->xi*(q[ia][ib][iv] + r3*d_[ia][ib])*trace_qw[iv];
	      for (id = 0; id < 3; id++) {
		__targetILP__(iv) s[ia][ib][iv] +=
		  (be->param->xi*d[ia][id][iv] + omega[ia][id][iv])*(q[id][ib][iv] + r3*d_[id][ib])
		+ (q[ia][id][iv] + r3*d_[ia][id])*(be->param->xi*d[id][ib][iv] - omega[id][ib][iv]);
	      }
	    }
	  }
	}

	/* Fluctuating tensor order parameter */

	if (noise_on) {

	  __targetILP__(iv) {
	
	    noise_reap_n(noise, baseIndex, NQAB, chi);
	
	    for (id = 0; id < NQAB; id++) {
	      assert(0); /* set var? */
	      chi[id] = be->param->var*chi[id];
	    }
	
	    for (ia = 0; ia < 3; ia++) {
	      for (ib = 0; ib < 3; ib++) {
		chi_qab[ia][ib][iv] = 0.0;
		for (id = 0; id < NQAB; id++) {
		  chi_qab[ia][ib][iv] += chi[id]*tc_tmatrix[ia][ib][id];
		}
	      }
	    }
	  }
	}

	/* Here's the full hydrodynamic update. */
	  
	__targetILP__(iv) indexj[iv]=targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv]-1,coordschunk[Z][iv],tc_Nall);
	__targetILP__(iv) indexk[iv]=targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv],coordschunk[Z][iv]-1,tc_Nall);


	__targetILP__(iv) {
	  if (includeSite[iv]) {
	    q[X][X][iv] += dt*
	      (s[X][X][iv] + be->param->gamma*h[X][X][iv] + chi_qab[X][X][iv]
	       - flux->fe[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)]
	       + flux->fw[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)]
	       - flux->fy[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)]
	       + flux->fy[addr_rank1(tc_nSites,NQAB,indexj[iv],XX)]
	       - flux->fz[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)]
	       + flux->fz[addr_rank1(tc_nSites,NQAB,indexk[iv],XX)]);
      }
    }

    __targetILP__(iv) {
      if (includeSite[iv]) {
	q[X][Y][iv] += dt*
	  (s[X][Y][iv] + be->param->gamma*h[X][Y][iv] + chi_qab[X][Y][iv]
	   - flux->fe[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)]
	   + flux->fw[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)]
	   - flux->fy[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)]
	   + flux->fy[addr_rank1(tc_nSites,NQAB,indexj[iv],XY)]
	   - flux->fz[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)]
	   + flux->fz[addr_rank1(tc_nSites,NQAB,indexk[iv],XY)]);
      }
    }
	
    __targetILP__(iv) {
      if (includeSite[iv]) {
	q[X][Z][iv] += dt*
	  (s[X][Z][iv] + be->param->gamma*h[X][Z][iv] + chi_qab[X][Z][iv]
	   - flux->fe[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)]
	   + flux->fw[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)]
	   - flux->fy[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)]
	   + flux->fy[addr_rank1(tc_nSites,NQAB,indexj[iv],XZ)]
	   - flux->fz[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)]
	   + flux->fz[addr_rank1(tc_nSites,NQAB,indexk[iv],XZ)]);
      }
    }
	
    __targetILP__(iv) {
      if (includeSite[iv]) {
	q[Y][Y][iv] += dt*
	  (s[Y][Y][iv] + be->param->gamma*h[Y][Y][iv]+ chi_qab[Y][Y][iv]
	   - flux->fe[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)]
	   + flux->fw[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)]
	   - flux->fy[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)]
	   + flux->fy[addr_rank1(tc_nSites,NQAB,indexj[iv],YY)]
	   - flux->fz[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)]
	   + flux->fz[addr_rank1(tc_nSites,NQAB,indexk[iv],YY)]);
      }
    }
	
    __targetILP__(iv) {
      if (includeSite[iv]) {
	q[Y][Z][iv] += dt*
	  (s[Y][Z][iv] + be->param->gamma*h[Y][Z][iv] + chi_qab[Y][Z][iv]
	   - flux->fe[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)]
	   + flux->fw[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)]
	   - flux->fy[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)]
	   + flux->fy[addr_rank1(tc_nSites,NQAB,indexj[iv],YZ)]
	   - flux->fz[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)]
	   + flux->fz[addr_rank1(tc_nSites,NQAB,indexk[iv],YZ)]);
      }
    }

    __targetILP__(iv) fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)] = q[X][X][iv];
    __targetILP__(iv) fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)] = q[X][Y][iv];
    __targetILP__(iv) fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)] = q[X][Z][iv];
    __targetILP__(iv) fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)] = q[Y][Y][iv];
    __targetILP__(iv) fq->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)] = q[Y][Z][iv];

      }
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
 *  colloids_fix_swd
 *
 *  The velocity gradient tensor used in the Beris-Edwards equations
 *  requires some approximation to the velocity at lattice sites
 *  inside particles. Here we set the lattice velocity based on
 *  the solid body rotation u = v + Omega x rb
 *
 *****************************************************************************/
 
__host__ int colloids_fix_swd(colloids_info_t * cinfo, hydro_t * hydro, map_t * map) {

  int nlocal[3];
  int noffset[3];
  const int nextra = 1;
  __global__ void colloids_fix_swd_lattice(colloids_info_t * cinfo,
					   hydro_t * hydro,
					   map_t * map);

  assert(cinfo);
  assert(map);

  if (hydro == NULL) return 0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  int nhalo = coords_nhalo();

  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;


  int nSites;

  nSites =Nall[X]*Nall[Y]*Nall[Z];

  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(tc_noffset,noffset, 3*sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(&tc_nextra,&nextra, sizeof(int)); 

  colloids_fix_swd_lattice __targetLaunch__(nSites) (cinfo->tcopy,
						     hydro->target,
						     map->target);
  targetSynchronize();

  return 0;
}

/*****************************************************************************
 *
 *  colloids_fix_swd_kernel
 *
 *  Device note: cinfo is coming from unified memory.
 *
 *  Issues: this routines is doing towo things: solid and colloid.
 *  these could be separate.
 *
 *****************************************************************************/

__global__
void colloids_fix_swd_lattice(colloids_info_t * cinfo, hydro_t * hydro,
			      map_t * map) {

  int index;

  __targetTLPNoStride__(index, tc_nSites) {

    int coords[3];
    int ic, jc, kc, ia;
    
    double u[3];
    double rb[3];
    double x, y, z;
    
    colloid_t * p_c;
    
    targetCoords3D(coords,tc_Nall,index);
    
    // if not a halo site:
    if (coords[0] >= (tc_nhalo-tc_nextra) && 
	coords[1] >= (tc_nhalo-tc_nextra) && 
	coords[2] >= (tc_nhalo-tc_nextra) &&
	coords[0] < tc_Nall[X]-(tc_nhalo-tc_nextra) &&  
	coords[1] < tc_Nall[Y]-(tc_nhalo-tc_nextra)  &&  
	coords[2] < tc_Nall[Z]-(tc_nhalo-tc_nextra) ){ 
      
      
      int coords[3];
      
      targetCoords3D(coords,tc_Nall,index);
      
      ic=coords[0]-tc_nhalo+1;
      jc=coords[1]-tc_nhalo+1;
      kc=coords[2]-tc_nhalo+1;

      x = tc_noffset[X]+ic;
      y = tc_noffset[Y]+jc;
      z = tc_noffset[Z]+kc;
      
      
      if (map->status[index] != MAP_FLUID) {
	u[X] = 0.0;
	u[Y] = 0.0;
	u[Z] = 0.0;
	  
	for (ia = 0; ia < 3; ia++) {
	  hydro->u[addr_hydro(index, ia)] = u[ia];
	}  
      }
      
      p_c = NULL;
      if (cinfo->map_new) p_c = cinfo->map_new[index];
      
      if (p_c) {
	/* Set the lattice velocity here to the solid body
	 * rotational velocity */
	
	rb[X] = x - p_c->s.r[X];
	rb[Y] = y - p_c->s.r[Y];
	rb[Z] = z - p_c->s.r[Z];
	
	u[X] = p_c->s.w[Y]*rb[Z] - p_c->s.w[Z]*rb[Y];
	u[Y] = p_c->s.w[Z]*rb[X] - p_c->s.w[X]*rb[Z];
	u[Z] = p_c->s.w[X]*rb[Y] - p_c->s.w[Y]*rb[X];

	u[X] += p_c->s.v[X];
	u[Y] += p_c->s.v[Y];
	u[Z] += p_c->s.v[Z];

	for (ia = 0; ia < 3; ia++) {
	  hydro->u[addr_hydro(index, ia)] = u[ia];
	}
      }
    }
  }
  
  return;
}

