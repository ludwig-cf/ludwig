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
#include "colloids_Q_tensor.h"
#include "advection_bcs.h"
#include "blue_phase.h"
#include "blue_phase_beris_edwards.h"
#include "advection_s.h"
#include "free_energy_tensor.h"
#include "hydro_s.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "map_s.h"
#include "timer.h"

__host__ int beris_edw_update_driver(beris_edw_t * be, field_t * fq,
				     field_grad_t * fq_grad,
				     hydro_t * hydro,
				     map_t * map, noise_t * noise);

__targetConst__ double tc_gamma;
__targetConst__ double tc_var;
__targetConst__ double tc_tmatrix[3][3][NQAB];

struct beris_edw_s {
  beris_edw_param_t param;         /* Parameters */ 
  advflux_t * flux;                /* Advective fluxes */

  beris_edw_t * target;            /* Target memory */
};

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
  free(be);

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_memcpy
 *
 *****************************************************************************/

__host__ int beris_edw_memcpy(beris_edw_t * be, int flag) {

  int ndevice;

  assert(be);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(be->target = be);
  }
  else {
    switch (flag) {
    case cudaMemcpyHostToDevice:
      copyToTarget(&be->target->param, &be->param, sizeof(beris_edw_param_t));
      break;
    case cudaMemcpyDeviceToHost:
      /* no action */
      break;
    default:
      fatal("Bad flag in beris_edw_memcpy\n");
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  beris_edw_param_set
 *
 *****************************************************************************/

__host__ int beris_edw_param_set(beris_edw_t * be, beris_edw_param_t vals) {

  assert(be);

  be->param = vals;

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

__host__ int beris_edw_update(beris_edw_t * be, field_t * fq,
			      field_grad_t * fq_grad,
			      hydro_t * hydro, map_t * map,
			      noise_t * noise) {
  int nf;

  assert(be);
  assert(fq);
  assert(map);

  field_nf(fq, &nf);
  assert(nf == NQAB);

  if (hydro) {
    hydro_lees_edwards(hydro);
    advection_x(be->flux, hydro, fq);
    advection_bcs_no_normal_flux(nf, be->flux, map);
  }

  beris_edw_update_driver(be, fq, fq_grad, hydro, map, noise);

  return 0;
}

__targetHost__ __target__ void h_loop_unrolled_be(double sum[VVL], double dq[3][3][3][VVL],
				double dsq[3][3][VVL],
				double q[3][3][VVL],
				double h[3][3][VVL],
				double eq[VVL],
				bluePhaseKernelConstants_t* pbpc);


/*IMPORTANT NOTE*/

/* the below routine is a COPY of that in blue_phase.h */
/* required to be in scope here for performance reasons on GPU */
/* since otherwise the compiler places the emporary q[][][] etc arrays */
/* in regular off-chip memory rather than registers */
/* which has a huge impact on performance */
/* TO DO: place this in a header file to be included both here and blue_phase.c */
/* or work out how to get the compiler to inline it from the different source file */

__targetHost__ __target__
void blue_phase_compute_h_vec_inline(double q[3][3][VVL], 
				     double dq[3][3][3][VVL],
				     double dsq[3][3][VVL], 
				     double h[3][3][VVL],
				     bluePhaseKernelConstants_t* pbpc) {

  int iv=0;
  int ia, ib, ic;

  double q2[VVL];
  double e2[VVL];
  double eq[VVL];
  double sum[VVL];

  /* From the bulk terms in the free energy... */

  __targetILP__(iv) q2[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) q2[iv] += q[ia][ib][iv]*q[ia][ib][iv];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) sum[iv] = 0.0;
      for (ic = 0; ic < 3; ic++) {
	__targetILP__(iv) sum[iv] += q[ia][ic][iv]*q[ib][ic][iv];
      }
      __targetILP__(iv) h[ia][ib][iv] = -pbpc->a0_*(1.0 - pbpc->r3_*pbpc->gamma_)*q[ia][ib][iv]
	+ pbpc->a0_*pbpc->gamma_*(sum[iv] - pbpc->r3_*q2[iv]*pbpc->d_[ia][ib]) - pbpc->a0_*pbpc->gamma_*q2[iv]*q[ia][ib][iv];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  __targetILP__(iv) eq[iv] = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	__targetILP__(iv) eq[iv] += pbpc->e_[ib][ic][ia]*dq[ib][ic][ia][iv];
      }
    }
  }

  /* d_c Q_db written as d_c Q_bd etc */
  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */
  /*     __targetILP__(iv) sum[iv] = 0.0; */
  /*     for (ic = 0; ic < 3; ic++) { */
  /* 	for (id = 0; id < 3; id++) { */
  /* 	  __targetILP__(iv) sum[iv] += */
  /* 	    (pbpc->e_[ia][ic][id]*dq[ic][ib][id][iv] + pbpc->e_[ib][ic][id]*dq[ic][ia][id][iv]); */
  /* 	} */
  /*     } */
      
  /*     __targetILP__(iv) h[ia][ib][iv] += pbpc->kappa0*dsq[ia][ib][iv] */
  /* 	- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[ia][ib] */
  /* 	- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[ia][ib][iv]; */
  /*   } */
  /* } */
  //#include "h_loop.h"

  h_loop_unrolled_be(sum,dq,dsq,q,h,eq,pbpc);

  /* Electric field term */

  __targetILP__(iv) e2[iv] = 0.0;
  for (ia = 0; ia < 3; ia++) {
    __targetILP__(iv) e2[iv] += pbpc->e0[ia]*pbpc->e0[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) h[ia][ib][iv] +=  pbpc->epsilon_*(pbpc->e0[ia]*pbpc->e0[ib] - pbpc->r3_*pbpc->d_[ia][ib]*e2[iv]);
    }
  }

  return;
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

__host__ int beris_edw_update_driver(beris_edw_t * be, field_t * fq,
				     field_grad_t * fq_grad,
				     hydro_t * hydro,
				     map_t * map,
				     noise_t * noise) {
  int nlocal[3];
  int noise_on = 0;

  double tmatrix[3][3][NQAB];
  double kt, var = 0.0;

  void (* molecular_field)(const int index, double h[3][3]);
  __targetEntry__
    void beris_edw_kernel(field_t * fq, field_grad_t * fqgrad,
		      hydro_t * hydro,
		      advflux_t * flux,
		      map_t * map,
		      int noise_on,
		      noise_t * noise,
		      void* pcon,
			  void   (*molecular_field)(const int, double h[3][3]), int isBPMF);


  assert(be);
  assert(fq);
  assert(map);

  coords_nlocal(nlocal);

  /* Get kBT, variance of noise and set basis of traceless,
   * symmetric matrices for contraction */

  if (noise) noise_present(noise, NOISE_QAB, &noise_on);
  if (noise_on) {
    physics_kt(&kt);
    var = sqrt(2.0*kt*be->param.gamma);
    beris_edw_tmatrix_set(tmatrix);
  }

  molecular_field = fe_t_molecular_field();

  int isBPMF = (molecular_field==blue_phase_molecular_field);
#ifdef __NVCC__
  /* make sure blue_phase_molecular_field is in use here because
     this is assumed in targetDP port*/
  if (!isBPMF)
    fatal("only blue_phase_molecular_field is supported for CUDA\n");
#endif

  int nhalo;
  nhalo = coords_nhalo();


  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;


  int nSites=Nall[X]*Nall[Y]*Nall[Z];


  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));
  copyConstToTarget(&tc_gamma,&be->param.gamma, sizeof(double));  
  copyConstToTarget(&tc_var,&var, sizeof(double));  
  copyConstToTarget(tc_tmatrix,tmatrix, 3*3*NQAB*sizeof(double)); 

  /* initialise kernel constants on both host and target*/
  blue_phase_set_kernel_constants();

  /* get a pointer to target copy of stucture containing kernel constants*/

  void* pcon=NULL;
  blue_phase_target_constant_ptr(&pcon);


  TIMER_start(BP_BE_UPDATE_KERNEL);

  beris_edw_kernel __targetLaunch__(nSites) (fq->tcopy,
					     fq_grad->tcopy,
					     hydro->tcopy,
					     be->flux->target,
					     map->target,
					     noise_on, noise, pcon,
					     molecular_field, isBPMF);
  
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
void beris_edw_kernel(field_t * fq, field_grad_t * fqgrad,
		      hydro_t * hydro,
		      advflux_t * flux,
		      map_t * map,
		      int noise_on,
		      noise_t * noise,
		      void* pcon,
void   (*molecular_field)(const int, double h[3][3]), int isBPMF) {

  int baseIndex;

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

    bluePhaseKernelConstants_t* pbpc= (bluePhaseKernelConstants_t*) pcon;

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
	assert(0);
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

	if (isBPMF)
	  blue_phase_compute_h_vec_inline(q, dq, dsq, h, pbpc);
	else
	{
#ifndef __NVCC__
	    /*only BP supported for CUDA. This is caught earlier*/
	  __targetILP__(iv) {
	    double htmp[3][3];
	    molecular_field(baseIndex+iv, htmp);
	    for (ia = 0; ia < 3; ia++) 
	      for (ib = 0; ib < 3; ib++) 
		h[ia][ib][iv]=htmp[ia][ib];
	  }
#endif
	}
      
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
	  __targetILP__(iv) tr[iv] = pbpc->r3_*(w[X][X][iv] + w[Y][Y][iv] + w[Z][Z][iv]);
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
	      __targetILP__(iv) s[ia][ib][iv] = -2.0*pbpc->xi_*(q[ia][ib][iv] + pbpc->r3_*pbpc->d_[ia][ib])*trace_qw[iv];
	      for (id = 0; id < 3; id++) {
		__targetILP__(iv) s[ia][ib][iv] +=
		  (pbpc->xi_*d[ia][id][iv] + omega[ia][id][iv])*(q[id][ib][iv] + pbpc->r3_*pbpc->d_[id][ib])
		+ (q[ia][id][iv] + pbpc->r3_*pbpc->d_[ia][id])*(pbpc->xi_*d[id][ib][iv] - omega[id][ib][iv]);
	      }
	    }
	  }
	}

	/* Fluctuating tensor order parameter */

	if (noise_on) {
#ifdef __NVCC__
      printf("Error: noise is not yet supported for CUDA\n");
#else

      __targetILP__(iv) {
	
	noise_reap_n(noise, baseIndex, NQAB, chi);
	
	for (id = 0; id < NQAB; id++) {
	  chi[id] = tc_var*chi[id];
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
#endif
	}

	/* Here's the full hydrodynamic update. */
	  
	__targetILP__(iv) indexj[iv]=targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv]-1,coordschunk[Z][iv],tc_Nall);
	__targetILP__(iv) indexk[iv]=targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv],coordschunk[Z][iv]-1,tc_Nall);


	__targetILP__(iv) {
	  if (includeSite[iv]) {
	    q[X][X][iv] += dt*
	      (s[X][X][iv] + tc_gamma*h[X][X][iv] + chi_qab[X][X][iv]
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
	  (s[X][Y][iv] + tc_gamma*h[X][Y][iv] + chi_qab[X][Y][iv]
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
	  (s[X][Z][iv] + tc_gamma*h[X][Z][iv] + chi_qab[X][Z][iv]
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
	  (s[Y][Y][iv] + tc_gamma*h[Y][Y][iv]+ chi_qab[Y][Y][iv]
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
	  (s[Y][Z][iv] + tc_gamma*h[Y][Z][iv] + chi_qab[Y][Z][iv]
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
 *  beris_edw_tmatrix_set
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

__host__ __device__ int beris_edw_tmatrix_set(double t[3][3][NQAB]) {

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


/* Unrolled kernels: thes get much beter performance since he multiplications
 by 0 and repeated loading of duplicate coefficients have been eliminated */

__target__ void h_loop_unrolled_be(double sum[VVL], double dq[3][3][3][VVL],
				double dsq[3][3][VVL],
				double q[3][3][VVL],
				double h[3][3][VVL],
				double eq[VVL],
				bluePhaseKernelConstants_t* pbpc){

  int iv=0;

__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[1][0][2][iv] + dq[1][0][2][iv];
__targetILP__(iv) sum[iv] += -dq[2][0][1][iv] + -dq[2][0][1][iv];
__targetILP__(iv) h[0][0][iv] += pbpc->kappa0*dsq[0][0][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[0][0]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[0][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][0][2][iv];
__targetILP__(iv) sum[iv] += dq[1][1][2][iv] ;
__targetILP__(iv) sum[iv] += dq[2][0][0][iv];
__targetILP__(iv) sum[iv] += -dq[2][1][1][iv] ;
__targetILP__(iv) h[0][1][iv] += pbpc->kappa0*dsq[0][1][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[0][1]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[0][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][0][1][iv];
__targetILP__(iv) sum[iv] += -dq[1][0][0][iv];
__targetILP__(iv) sum[iv] += dq[1][2][2][iv] ;
__targetILP__(iv) sum[iv] += -dq[2][2][1][iv] ;
__targetILP__(iv) h[0][2][iv] += pbpc->kappa0*dsq[0][2][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[0][2]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[0][2][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][0][2][iv] ;
__targetILP__(iv) sum[iv] += dq[1][1][2][iv];
__targetILP__(iv) sum[iv] += dq[2][0][0][iv] ;
__targetILP__(iv) sum[iv] += -dq[2][1][1][iv];
__targetILP__(iv) h[1][0][iv] += pbpc->kappa0*dsq[1][0][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[1][0]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[1][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][1][2][iv] + -dq[0][1][2][iv];
__targetILP__(iv) sum[iv] += dq[2][1][0][iv] + dq[2][1][0][iv];
__targetILP__(iv) h[1][1][iv] += pbpc->kappa0*dsq[1][1][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[1][1]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[1][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][1][1][iv];
__targetILP__(iv) sum[iv] += -dq[0][2][2][iv] ;
__targetILP__(iv) sum[iv] += -dq[1][1][0][iv];
__targetILP__(iv) sum[iv] += dq[2][2][0][iv] ;
__targetILP__(iv) h[1][2][iv] += pbpc->kappa0*dsq[1][2][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[1][2]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[1][2][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][0][1][iv] ;
__targetILP__(iv) sum[iv] += -dq[1][0][0][iv] ;
__targetILP__(iv) sum[iv] += dq[1][2][2][iv];
__targetILP__(iv) sum[iv] += -dq[2][2][1][iv];
__targetILP__(iv) h[2][0][iv] += pbpc->kappa0*dsq[2][0][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[2][0]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[2][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][1][1][iv] ;
__targetILP__(iv) sum[iv] += -dq[0][2][2][iv];
__targetILP__(iv) sum[iv] += -dq[1][1][0][iv] ;
__targetILP__(iv) sum[iv] += dq[2][2][0][iv];
__targetILP__(iv) h[2][1][iv] += pbpc->kappa0*dsq[2][1][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[2][1]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[2][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][2][1][iv] + dq[0][2][1][iv];
__targetILP__(iv) sum[iv] += -dq[1][2][0][iv] + -dq[1][2][0][iv];
__targetILP__(iv) h[2][2][iv] += pbpc->kappa0*dsq[2][2][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[2][2]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[2][2][iv];

}
