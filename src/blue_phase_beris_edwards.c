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
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
#include "colloids_Q_tensor.h"
#include "advection.h"
#include "advection_bcs.h"
#include "blue_phase.h"
#include "blue_phase_beris_edwards.h"
#include "advection_s.h"
#include "free_energy_tensor.h"
#include "hydro.h"
#include "hydro_s.h"
#include "field.h"
#include "field_s.h"
#include "field_grad.h"
#include "field_grad_s.h"
#include "map_s.h"

static int blue_phase_be_update(field_t * fq, field_grad_t * fq_grad, hydro_t * hydro, advflux_t * f,
				map_t * map, noise_t * noise);

/*****************************************************************************
 *
 *  blue_phase_beris_edwards
 *
 *  Driver routine for the update.
 *
 *  hydro is allowed to be NULL, in which case we only have relaxational
 *  dynamics.
 *
 *****************************************************************************/

__targetHost__ int blue_phase_beris_edwards(field_t * fq, field_grad_t * fq_grad, hydro_t * hydro, map_t * map,
			     noise_t * noise) {

  int nf;
  advflux_t * flux = NULL;

  assert(fq);
  assert(map);

  /* Set up advective fluxes (which default to zero),
   * work out the hydrodynmaic stuff if required, and do the update. */

  field_nf(fq, &nf);
  assert(nf == NQAB);

  advflux_create(nf, &flux);

  if (hydro) {

    hydro_lees_edwards(hydro);

    advection_x(flux, hydro, fq);

    advection_bcs_no_normal_flux(nf, flux, map);
  }


  blue_phase_be_update(fq, fq_grad, hydro, flux, map, noise);


  advflux_free(flux);


  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_be_update
 *
 *  Update q via Euler forward step. Note here we only update the
 *  5 independent elements of the Q tensor.
 *
 *  hydro is allowed to be NULL, in which case we only have relaxational
 *  dynamics.
 *
 *****************************************************************************/

__targetConst__ double tc_gamma;
__targetConst__ double tc_var;
__targetConst__ double tc_tmatrix[3][3][NQAB];


//perform update across the lattice on target
__targetEntry__ void blue_phase_be_update_lattice(field_t * t_q, field_grad_t * t_q_grad, 
						  hydro_t * hydro, advflux_t * flux, map_t * map,
						  int noise_on, noise_t * noise, void* pcon) {
  int ia, ib, id;
  int index, indexj, indexk;
  int nf;
  int status;

  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  double w[3][3];
  double d[3][3];
  double h[3][3];
  double s[3][3];

  double omega[3][3];
  double trace_qw;
  double chi[NQAB], chi_qab[3][3];


  const double dt = 1.0;

  bluePhaseKernelConstants_t* pbpc= (bluePhaseKernelConstants_t*) pcon;

  nf=t_q->nf;

  __targetTLPNoStride__(index,tc_nSites){
  
  int coords[3];
  targetCoords3D(coords,tc_Nall,index);
  
  // if not a halo site:
    if (coords[0] >= (tc_nhalo) &&
	coords[1] >= (tc_nhalo) &&
	coords[2] >= (tc_nhalo) &&
	coords[0] < tc_Nall[X]-(tc_nhalo) &&
	coords[1] < tc_Nall[Y]-(tc_nhalo)  &&
	coords[2] < tc_Nall[Z]-(tc_nhalo) ){

      
      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  s[ia][ib] = 0.0;
	  chi_qab[ia][ib] = 0.0;
	}
      }
      


#ifndef CUDA
      //on gpu we will just calc all sites (and discard non-fluid results)
      map_status(map, index, &status);
      if (status != MAP_FLUID) continue;
#endif

	//calculate molecular field	

	int ia, ib;
	
	q[X][X] = t_q->data[NQAB*index + XX];
	q[X][Y] = t_q->data[NQAB*index + XY];
	q[X][Z] = t_q->data[NQAB*index + XZ];
	q[Y][X] = q[X][Y];
	q[Y][Y] = t_q->data[NQAB*index + YY];
	q[Y][Z] = t_q->data[NQAB*index + YZ];
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];
	
	for (ia = 0; ia < NVECTOR; ia++) {
	  dq[ia][X][X] = t_q_grad->grad[NVECTOR*(NQAB*index + XX) + ia];
	  dq[ia][X][Y] = t_q_grad->grad[NVECTOR*(NQAB*index + XY) + ia];
	  dq[ia][X][Z] = t_q_grad->grad[NVECTOR*(NQAB*index + XZ) + ia];
	  dq[ia][Y][X] = dq[ia][X][Y];
	  dq[ia][Y][Y] = t_q_grad->grad[NVECTOR*(NQAB*index + YY) + ia];
	  dq[ia][Y][Z] = t_q_grad->grad[NVECTOR*(NQAB*index + YZ) + ia];
	  dq[ia][Z][X] = dq[ia][X][Z];
	  dq[ia][Z][Y] = dq[ia][Y][Z];
	  dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
	}
	
	
	dsq[X][X] = t_q_grad->delsq[NQAB*index + XX];
	dsq[X][Y] = t_q_grad->delsq[NQAB*index + XY];
	dsq[X][Z] = t_q_grad->delsq[NQAB*index + XZ];
	dsq[Y][X] = dsq[X][Y];
	dsq[Y][Y] = t_q_grad->delsq[NQAB*index + YY];
	dsq[Y][Z] = t_q_grad->delsq[NQAB*index + YZ];
	dsq[Z][X] = dsq[X][Z];
	dsq[Z][Y] = dsq[Y][Z];
	dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];
	
	
	blue_phase_compute_h(q, dq, dsq, h, pbpc);
	
	
	//end inline
	
	if (hydro) {

	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

	  //hydro_u_gradient_tensor(hydro, ic, jc, kc, w);
	  //inline above function
	  //TODO add lees edwards support

	  int im1 = targetIndex3D(coords[0]-1,coords[1],coords[2],tc_Nall);
	  int ip1 = targetIndex3D(coords[0]+1,coords[1],coords[2],tc_Nall);
	  
	  w[X][X] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,X)] - hydro->u[HYADR(tc_nSites,3,im1,X)]);
	  w[Y][X] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,Y)] - hydro->u[HYADR(tc_nSites,3,im1,Y)]);
	  w[Z][X] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,Z)] - hydro->u[HYADR(tc_nSites,3,im1,Z)]);
	  
	  im1 = targetIndex3D(coords[0],coords[1]-1,coords[2],tc_Nall);
	  ip1 = targetIndex3D(coords[0],coords[1]+1,coords[2],tc_Nall);
	  
	  w[X][Y] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,X)] - hydro->u[HYADR(tc_nSites,3,im1,X)]);
	  w[Y][Y] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,Y)] - hydro->u[HYADR(tc_nSites,3,im1,Y)]);
	  w[Z][Y] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,Z)] - hydro->u[HYADR(tc_nSites,3,im1,Z)]);
	  
	  im1 = targetIndex3D(coords[0],coords[1],coords[2]-1,tc_Nall);
	  ip1 = targetIndex3D(coords[0],coords[1],coords[2]+1,tc_Nall);
	  
	  w[X][Z] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,X)] - hydro->u[HYADR(tc_nSites,3,im1,X)]);
	  w[Y][Z] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,Y)] - hydro->u[HYADR(tc_nSites,3,im1,Y)]);
	  w[Z][Z] = 0.5*(hydro->u[HYADR(tc_nSites,3,ip1,Z)] - hydro->u[HYADR(tc_nSites,3,im1,Z)]);

	  /* Enforce tracelessness */
	  
	  double tr = pbpc->r3_*(w[X][X] + w[Y][Y] + w[Z][Z]);
	  w[X][X] -= tr;
	  w[Y][Y] -= tr;
	  w[Z][Z] -= tr;


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
	      s[ia][ib] = -2.0*pbpc->xi_*(q[ia][ib] + pbpc->r3_*pbpc->d_[ia][ib])*trace_qw;
	      for (id = 0; id < 3; id++) {
		s[ia][ib] +=
		  (pbpc->xi_*d[ia][id] + omega[ia][id])*(q[id][ib] + pbpc->r3_*pbpc->d_[id][ib])
		+ (q[ia][id] + pbpc->r3_*pbpc->d_[ia][id])*(pbpc->xi_*d[id][ib] - omega[id][ib]);
	      }
	    }
	  }
	}

	/* Fluctuating tensor order parameter */

    if (noise_on) {

#ifdef CUDA 
      printf("Error: noise is not yet supported for CUDA\n");
#else      
      noise_reap_n(noise, index, NQAB, chi);
      for (id = 0; id < NQAB; id++) {
	chi[id] = tc_var*chi[id];
      }
      
      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  chi_qab[ia][ib] = 0.0;
	  for (id = 0; id < NQAB; id++) {
	    chi_qab[ia][ib] += chi[id]*tc_tmatrix[ia][ib][id];
	  }
	}
      }
#endif
      
    }

	/* Here's the full hydrodynamic update. */
	  
	indexj=targetIndex3D(coords[0],coords[1]-1,coords[2],tc_Nall);
	indexk=targetIndex3D(coords[0],coords[1],coords[2]-1,tc_Nall);

	q[X][X] += dt*(s[X][X] + tc_gamma*h[X][X] + chi_qab[X][X]
		       - flux->fe[nf*index + XX] + flux->fw[nf*index  + XX]
		       - flux->fy[nf*index + XX] + flux->fy[nf*indexj + XX]
		       - flux->fz[nf*index + XX] + flux->fz[nf*indexk + XX]);

	q[X][Y] += dt*(s[X][Y] + tc_gamma*h[X][Y] + chi_qab[X][Y]
		       - flux->fe[nf*index + XY] + flux->fw[nf*index  + XY]
		       - flux->fy[nf*index + XY] + flux->fy[nf*indexj + XY]
		       - flux->fz[nf*index + XY] + flux->fz[nf*indexk + XY]);

	q[X][Z] += dt*(s[X][Z] + tc_gamma*h[X][Z] + chi_qab[X][Z]
		       - flux->fe[nf*index + XZ] + flux->fw[nf*index  + XZ]
		       - flux->fy[nf*index + XZ] + flux->fy[nf*indexj + XZ]
		       - flux->fz[nf*index + XZ] + flux->fz[nf*indexk + XZ]);

	q[Y][Y] += dt*(s[Y][Y] + tc_gamma*h[Y][Y] + chi_qab[Y][Y]
		       - flux->fe[nf*index + YY] + flux->fw[nf*index  + YY]
		       - flux->fy[nf*index + YY] + flux->fy[nf*indexj + YY]
		       - flux->fz[nf*index + YY] + flux->fz[nf*indexk + YY]);

	q[Y][Z] += dt*(s[Y][Z] + tc_gamma*h[Y][Z] + chi_qab[Y][Z]
		       - flux->fe[nf*index + YZ] + flux->fw[nf*index  + YZ]
		       - flux->fy[nf*index + YZ] + flux->fy[nf*indexj + YZ]
		       - flux->fz[nf*index + YZ] + flux->fz[nf*indexk + YZ]);


	t_q->data[NQAB*index + XX] = q[X][X];
	t_q->data[NQAB*index + XY] = q[X][Y];
	t_q->data[NQAB*index + XZ] = q[X][Z];
	t_q->data[NQAB*index + YY] = q[Y][Y];
	t_q->data[NQAB*index + YZ] = q[Y][Z];

    }
  }

  return;
}

static int blue_phase_be_update(field_t * fq, field_grad_t * fq_grad, hydro_t * hydro,
				advflux_t * flux, map_t * map,
				noise_t * noise) {
  int nlocal[3];
  int nf;
  int noise_on = 0;

  double gamma;

  double chi[NQAB], chi_qab[3][3];
  double tmatrix[3][3][NQAB];
  double kt, var = 0.0;

  void (* molecular_field)(const int index, double h[3][3]);

  assert(fq);
  assert(flux);
  assert(map);

  coords_nlocal(nlocal);
  field_nf(fq, &nf);
  assert(nf == NQAB);

  double xi = blue_phase_get_xi();
  physics_lc_gamma_rot(&gamma);

  /* Get kBT, variance of noise and set basis of traceless,
   * symmetric matrices for contraction */

  if (noise) noise_present(noise, NOISE_QAB, &noise_on);
  if (noise_on) {
    physics_kt(&kt);
    var = sqrt(2.0*kt*gamma);
    blue_phase_be_tmatrix_set(tmatrix);
  }

  molecular_field = fe_t_molecular_field();

  //make sure blue_phase_molecular_field is in use here because this is assumed in targetDP port
  if (molecular_field!=blue_phase_molecular_field)
    fatal("molecular field should be blue_phase_molecular_field\n");

  int nhalo;
  nhalo = coords_nhalo();


  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;


  int nSites=Nall[X]*Nall[Y]*Nall[Z];


  //set up constants on target
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));
  copyConstToTarget(&tc_gamma,&gamma, sizeof(double));  
  copyConstToTarget(&tc_var,&var, sizeof(double));  
  copyConstToTarget(tc_tmatrix,tmatrix, 3*3*NQAB*sizeof(double)); 

  // initialise kernel constants on both host and target
  blue_phase_set_kernel_constants();

  // get a pointer to target copy of stucture containing kernel constants
  void* pcon=NULL;
  blue_phase_target_constant_ptr(&pcon);


  field_t* t_q = fq->tcopy; //target copy of tensor order parameter field structure
  field_grad_t* t_q_grad = fq_grad->tcopy; //target copy of grad field structure


  hydro_t* t_hydro = NULL; //target copy of hydro structure

  if(hydro)
    t_hydro = hydro->tcopy; 

  advflux_t* t_flux = flux->tcopy; //target copy of flux structure

  double* tmpptr;

#ifndef KEEPFIELDONTARGET
  //populate target copies from host 
  copyFromTarget(&tmpptr,&(t_q->data),sizeof(double*)); 
  copyToTarget(tmpptr,fq->data,fq->nf*nSites*sizeof(double));
  
  copyFromTarget(&tmpptr,&(t_q_grad->grad),sizeof(double*)); 
  copyToTarget(tmpptr,fq_grad->grad,fq_grad->nf*NVECTOR*nSites*sizeof(double));
  
  copyFromTarget(&tmpptr,&(t_q_grad->delsq),sizeof(double*)); 
  copyToTarget(tmpptr,fq_grad->delsq,fq_grad->nf*nSites*sizeof(double));
#endif    



#ifndef KEEPHYDROONTARGET
  if(hydro){
    copyFromTarget(&tmpptr,&(t_hydro->u),sizeof(double*)); 
    copyToTarget(tmpptr,hydro->u,hydro->nf*nSites*sizeof(double));
  }
#endif

#ifndef KEEPFIELDONTARGET
  copyFromTarget(&tmpptr,&(t_flux->fe),sizeof(double*)); 
  copyToTarget(tmpptr,flux->fe,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fw),sizeof(double*)); 
  copyToTarget(tmpptr,flux->fw,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fy),sizeof(double*)); 
  copyToTarget(tmpptr,flux->fy,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fz),sizeof(double*)); 
  copyToTarget(tmpptr,flux->fz,nf*nSites*sizeof(double));
#endif

  //launch update across lattice on target
  blue_phase_be_update_lattice __targetLaunch__(nSites) (fq->tcopy, fq_grad->tcopy, 
  							 t_hydro,
  							 flux->tcopy, map,
  							 noise_on, noise, pcon);
  
  targetSynchronize();

#ifndef KEEPFIELDONTARGET
  //get result back from target
  copyFromTarget(&tmpptr,&(t_q->data),sizeof(double*)); 
  copyFromTarget(fq->data,tmpptr,fq->nf*nSites*sizeof(double));
#endif

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_be_tmatrix_set
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

__targetHost__ int blue_phase_be_tmatrix_set(double t[3][3][NQAB]) {

  int ia, ib, id;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (id = 0; id < NQAB; id++) {
      	t[ia][ib][id] = 0.0;
      }
    }
  }

  t[X][X][XX] = sqrt(3.0/2.0)*(0.0 - r3_);
  t[Y][Y][XX] = sqrt(3.0/2.0)*(0.0 - r3_);
  t[Z][Z][XX] = sqrt(3.0/2.0)*(1.0 - r3_);

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
