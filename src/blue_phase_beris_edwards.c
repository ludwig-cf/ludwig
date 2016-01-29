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
 *  (c) 2009-2015 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
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
#include "timer.h"


static int blue_phase_be_update(field_t * fq, field_grad_t * fq_grad, hydro_t * hydro, advflux_t * f,
				map_t * map, noise_t * noise);

__targetConst__ double tc_gamma;
__targetConst__ double tc_var;
__targetConst__ double tc_tmatrix[3][3][NQAB];

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

__targetHost__ int blue_phase_beris_edwards(field_t * fq,
					    field_grad_t * fq_grad,
					    hydro_t * hydro, map_t * map,
					    noise_t * noise) {

  int nf;
  advflux_t * flux = NULL;

  assert(fq);
  assert(map);

  /* Set up advective fluxes (which default to zero),
   * work out the hydrodynmaic stuff if required, and do the update. */

  field_nf(fq, &nf);
  assert(nf == NQAB);

  TIMER_start(ADVECTION_BCS_MEM);
  advflux_create(nf, &flux);
  TIMER_stop(ADVECTION_BCS_MEM);

  if (hydro) {

    hydro_lees_edwards(hydro);

    advection_x(flux, hydro, fq);

    advection_bcs_no_normal_flux(nf, flux, map);
  }

  blue_phase_be_update(fq, fq_grad, hydro, flux, map, noise);

  TIMER_start(ADVECTION_BCS_MEM);
  advflux_free(flux);
  TIMER_stop(ADVECTION_BCS_MEM);

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
__targetHost__ __target__ void blue_phase_compute_h_vec_inline(double q[3][3][VVL], 
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
 *  blue_phase_be_update
 *
 *  Update q via Euler forward step. Note here we only update the
 *  5 independent elements of the Q tensor.
 *
 *  hydro is allowed to be NULL, in which case we only have relaxational
 *  dynamics.
 *
 *****************************************************************************/

__targetEntry__ void blue_phase_be_update_lattice(double* __restrict__ qdata,
						  const double* __restrict__ graddata, 
						  const double* __restrict__ graddelsq, 
						  const double* __restrict__ hydrou,
						  const double* __restrict__ fluxe,
						  const double* __restrict__ fluxw,
						  const double* __restrict__ fluxy,
						  const double* __restrict__ fluxz,
						  map_t * map,
						  int noise_on,
						  noise_t * noise,
						  void* pcon, int nf, int hydroOn,
void   (*molecular_field)(const int, double h[3][3]), int isBPMF) {

  int baseIndex;

  __targetTLP__(baseIndex,tc_nSites){

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
#endif
#endif /* else just calc all sites (and discard non-fluid results)*/

      /* calculate molecular field	*/

	int ia, ib;
	

	__targetILP__(iv) q[X][X][iv] = qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,XX)];
	__targetILP__(iv) q[X][Y][iv] = qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,XY)];
	__targetILP__(iv) q[X][Z][iv] = qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,XZ)];
	__targetILP__(iv) q[Y][X][iv] = q[X][Y][iv];
	__targetILP__(iv) q[Y][Y][iv] = qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,YY)];
	__targetILP__(iv) q[Y][Z][iv] = qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,YZ)];
	__targetILP__(iv) q[Z][X][iv] = q[X][Z][iv];
	__targetILP__(iv) q[Z][Y][iv] = q[Y][Z][iv];
	__targetILP__(iv) q[Z][Z][iv] = 0.0 - q[X][X][iv] - q[Y][Y][iv];
	
	for (ia = 0; ia < NVECTOR; ia++) {
	  __targetILP__(iv) dq[ia][X][X][iv] = graddata[FGRDADR(tc_nSites,NQAB,baseIndex+iv,XX,ia)];
	  __targetILP__(iv) dq[ia][X][Y][iv] = graddata[FGRDADR(tc_nSites,NQAB,baseIndex+iv,XY,ia)];
	  __targetILP__(iv) dq[ia][X][Z][iv] = graddata[FGRDADR(tc_nSites,NQAB,baseIndex+iv,XZ,ia)];
	  __targetILP__(iv) dq[ia][Y][X][iv] = dq[ia][X][Y][iv];
	  __targetILP__(iv) dq[ia][Y][Y][iv] = graddata[FGRDADR(tc_nSites,NQAB,baseIndex+iv,YY,ia)];
	  __targetILP__(iv) dq[ia][Y][Z][iv] = graddata[FGRDADR(tc_nSites,NQAB,baseIndex+iv,YZ,ia)];
	  __targetILP__(iv) dq[ia][Z][X][iv] = dq[ia][X][Z][iv];
	  __targetILP__(iv) dq[ia][Z][Y][iv] = dq[ia][Y][Z][iv];
	  __targetILP__(iv) dq[ia][Z][Z][iv] = 0.0 - dq[ia][X][X][iv] - dq[ia][Y][Y][iv];
	}
	
	
	__targetILP__(iv) dsq[X][X][iv] = graddelsq[FLDADR(tc_nSites,NQAB,baseIndex+iv,XX)];
	__targetILP__(iv) dsq[X][Y][iv] = graddelsq[FLDADR(tc_nSites,NQAB,baseIndex+iv,XY)];
	__targetILP__(iv) dsq[X][Z][iv] = graddelsq[FLDADR(tc_nSites,NQAB,baseIndex+iv,XZ)];
	__targetILP__(iv) dsq[Y][X][iv] = dsq[X][Y][iv];
	__targetILP__(iv) dsq[Y][Y][iv] = graddelsq[FLDADR(tc_nSites,NQAB,baseIndex+iv,YY)];
	__targetILP__(iv) dsq[Y][Z][iv] = graddelsq[FLDADR(tc_nSites,NQAB,baseIndex+iv,YZ)];
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
      
	  if (hydroOn) {

	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

	  /* hydro_u_gradient_tensor(hydro, ic, jc, kc, w);
	   * inline above function
	   * TODO add lees edwards support*/

	    int im1[VVL];
	    int ip1[VVL];
	  __targetILP__(iv)  im1[iv] = targetIndex3D(coordschunk[X][iv]-1,coordschunk[Y][iv],coordschunk[Z][iv],tc_Nall);
	  __targetILP__(iv)  ip1[iv] = targetIndex3D(coordschunk[X][iv]+1,coordschunk[Y][iv],coordschunk[Z][iv],tc_Nall);
	  
	  __targetILP__(iv) w[X][X][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],X)] - hydrou[HYADR(tc_nSites,3,im1[iv],X)]);
	  __targetILP__(iv) w[Y][X][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],Y)] - hydrou[HYADR(tc_nSites,3,im1[iv],Y)]);
	  __targetILP__(iv) w[Z][X][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],Z)] - hydrou[HYADR(tc_nSites,3,im1[iv],Z)]);
	  
	  __targetILP__(iv) im1[iv] = targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv]-1,coordschunk[Z][iv],tc_Nall);
	  __targetILP__(iv) ip1[iv] = targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv]+1,coordschunk[Z][iv],tc_Nall);
	  
	  __targetILP__(iv) w[X][Y][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],X)] - hydrou[HYADR(tc_nSites,3,im1[iv],X)]);
	  __targetILP__(iv) w[Y][Y][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],Y)] - hydrou[HYADR(tc_nSites,3,im1[iv],Y)]);
	  __targetILP__(iv) w[Z][Y][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],Z)] - hydrou[HYADR(tc_nSites,3,im1[iv],Z)]);
	  
	  __targetILP__(iv) im1[iv] = targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv],coordschunk[Z][iv]-1,tc_Nall);
	  __targetILP__(iv) ip1[iv] = targetIndex3D(coordschunk[X][iv],coordschunk[Y][iv],coordschunk[Z][iv]+1,tc_Nall);
	  
	  __targetILP__(iv) w[X][Z][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],X)] - hydrou[HYADR(tc_nSites,3,im1[iv],X)]);
	  __targetILP__(iv) w[Y][Z][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],Y)] - hydrou[HYADR(tc_nSites,3,im1[iv],Y)]);
	  __targetILP__(iv) w[Z][Z][iv] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1[iv],Z)] - hydrou[HYADR(tc_nSites,3,im1[iv],Z)]);

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

	__targetILP__(iv) q[X][X][iv] += dt*(s[X][X][iv] + tc_gamma*h[X][X][iv] + chi_qab[X][X][iv]
		       - fluxe[ADVADR(tc_nSites,nf,baseIndex+iv,XX)] + fluxw[ADVADR(tc_nSites,nf,baseIndex+iv,XX)]
		       - fluxy[ADVADR(tc_nSites,nf,baseIndex+iv,XX)] + fluxy[ADVADR(tc_nSites,nf,indexj[iv],XX)]
		       - fluxz[ADVADR(tc_nSites,nf,baseIndex+iv,XX)] + fluxz[ADVADR(tc_nSites,nf,indexk[iv],XX)]);


	__targetILP__(iv) q[X][Y][iv] += dt*(s[X][Y][iv] + tc_gamma*h[X][Y][iv] + chi_qab[X][Y][iv]
		       - fluxe[ADVADR(tc_nSites,nf,baseIndex+iv,XY)] + fluxw[ADVADR(tc_nSites,nf,baseIndex+iv,XY)]
		       - fluxy[ADVADR(tc_nSites,nf,baseIndex+iv,XY)] + fluxy[ADVADR(tc_nSites,nf,indexj[iv],XY)]
		       - fluxz[ADVADR(tc_nSites,nf,baseIndex+iv,XY)] + fluxz[ADVADR(tc_nSites,nf,indexk[iv],XY)]);

	__targetILP__(iv) q[X][Z][iv] += dt*(s[X][Z][iv] + tc_gamma*h[X][Z][iv] + chi_qab[X][Z][iv]
		       - fluxe[ADVADR(tc_nSites,nf,baseIndex+iv,XZ)] + fluxw[ADVADR(tc_nSites,nf,baseIndex+iv,XZ)]
		       - fluxy[ADVADR(tc_nSites,nf,baseIndex+iv,XZ)] + fluxy[ADVADR(tc_nSites,nf,indexj[iv],XZ)]
		       - fluxz[ADVADR(tc_nSites,nf,baseIndex+iv,XZ)] + fluxz[ADVADR(tc_nSites,nf,indexk[iv],XZ)]);

	__targetILP__(iv) q[Y][Y][iv] += dt*(s[Y][Y][iv] + tc_gamma*h[Y][Y][iv]+ chi_qab[Y][Y][iv]
		       - fluxe[ADVADR(tc_nSites,nf,baseIndex+iv,YY)] + fluxw[ADVADR(tc_nSites,nf,baseIndex+iv,YY)]
		       - fluxy[ADVADR(tc_nSites,nf,baseIndex+iv,YY)] + fluxy[ADVADR(tc_nSites,nf,indexj[iv],YY)]
		       - fluxz[ADVADR(tc_nSites,nf,baseIndex+iv,YY)] + fluxz[ADVADR(tc_nSites,nf,indexk[iv],YY)]);

	__targetILP__(iv) q[Y][Z][iv] += dt*(s[Y][Z][iv] + tc_gamma*h[Y][Z][iv] + chi_qab[Y][Z][iv]
		       - fluxe[ADVADR(tc_nSites,nf,baseIndex+iv,YZ)] + fluxw[ADVADR(tc_nSites,nf,baseIndex+iv,YZ)]
		       - fluxy[ADVADR(tc_nSites,nf,baseIndex+iv,YZ)] + fluxy[ADVADR(tc_nSites,nf,indexj[iv],YZ)]
		       - fluxz[ADVADR(tc_nSites,nf,baseIndex+iv,YZ)] + fluxz[ADVADR(tc_nSites,nf,indexk[iv],YZ)]);


	__targetILP__(iv) qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,XX)] = q[X][X][iv];
	__targetILP__(iv) qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,XY)] = q[X][Y][iv];
	__targetILP__(iv) qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,XZ)] = q[X][Z][iv];
	__targetILP__(iv) qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,YY)] = q[Y][Y][iv];
	__targetILP__(iv) qdata[FLDADR(tc_nSites,NQAB,baseIndex+iv,YZ)] = q[Y][Z][iv];

    }
  }

  return;
}

static int blue_phase_be_update(field_t * fq, field_grad_t * fq_grad,
				hydro_t * hydro,
				advflux_t * flux, map_t * map,
				noise_t * noise) {
  int nlocal[3];
  int nf;
  int noise_on = 0;

  double gamma;

  double tmatrix[3][3][NQAB];
  double kt, var = 0.0;

  void (* molecular_field)(const int index, double h[3][3]);

  assert(fq);
  assert(flux);
  assert(map);

  coords_nlocal(nlocal);
  field_nf(fq, &nf);
  assert(nf == NQAB);

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
  copyConstToTarget(&tc_gamma,&gamma, sizeof(double));  
  copyConstToTarget(&tc_var,&var, sizeof(double));  
  copyConstToTarget(tc_tmatrix,tmatrix, 3*3*NQAB*sizeof(double)); 

  /* initialise kernel constants on both host and target*/
  blue_phase_set_kernel_constants();

  /* get a pointer to target copy of stucture containing kernel constants*/
  void* pcon=NULL;
  blue_phase_target_constant_ptr(&pcon);

  /* target copy of tensor order parameter field structure
   * target copy of grad field structure
   * target copy of hydro structure */

  field_t* t_q = fq->tcopy;
  field_grad_t* t_q_grad = fq_grad->tcopy;
  hydro_t* t_hydro = NULL;

  if (hydro) {
    t_hydro = hydro->tcopy; 
  }

  /* target copy of flux structure*/
  advflux_t* t_flux = flux->tcopy;

  double* tmpptr;

#ifndef KEEPFIELDONTARGET
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

  int hydroOn=0;
  if (hydro)  hydroOn=1;

  /* launch update across lattice on target*/

  double* hydrou = NULL;
  double* qdata;
  double* graddata;
  double*  graddelsq;
  double*  fluxe;
  double*  fluxw;
  double*  fluxy;
  double*  fluxz;

  if (hydro) copyFromTarget(&hydrou,&(hydro->tcopy->u),sizeof(double*)); 
  copyFromTarget(&qdata,&(fq->tcopy->data),sizeof(double*)); 
  copyFromTarget(&graddata,&(fq_grad->tcopy->grad),sizeof(double*)); 
  copyFromTarget(&graddelsq,&(fq_grad->tcopy->delsq),sizeof(double*)); 
  copyFromTarget(&fluxe,&(flux->tcopy->fe),sizeof(double*)); 
  copyFromTarget(&fluxw,&(flux->tcopy->fw),sizeof(double*)); 
  copyFromTarget(&fluxy,&(flux->tcopy->fy),sizeof(double*)); 
  copyFromTarget(&fluxz,&(flux->tcopy->fz),sizeof(double*)); 


#ifdef VERBOSE_PERF_REPORT
  double t1, dataReadPerSite=0.,dataWrittenPerSite=0.;
  t1=-MPI_Wtime();
#endif

  TIMER_start(BP_BE_UPDATE_KERNEL);

  blue_phase_be_update_lattice __targetLaunch__(nSites) (qdata, graddata, graddelsq,
							 hydrou,
  							 fluxe,fluxw,fluxy,fluxz, map,
  							 noise_on, noise, pcon,
							  nf, hydroOn, molecular_field, isBPMF);
  
  targetSynchronize();
  TIMER_stop(BP_BE_UPDATE_KERNEL);

#ifdef VERBOSE_PERF_REPORT
  t1+=MPI_Wtime();

  //dynamically allocated arrays
  dataReadPerSite+=5*8;//fq
  dataWrittenPerSite+=5*8;//fq
  dataReadPerSite+=3*5*8;//grad 
  dataReadPerSite+=5*8;//delsq 
  dataReadPerSite+=4*5*8;//flux 
  dataReadPerSite+=3*8;//u:

  dataReadPerSite=dataReadPerSite/1024./1024./1024.; //convert to GB.
  dataWrittenPerSite=dataWrittenPerSite/1024./1024./1024.; //convert to GB.

  info("blue_phase_be_update: %1.3es, %3.3fGB/s read, %3.3fGB/s written, %3.3fGB/s total\n",t1,nSites*dataReadPerSite/t1,nSites*dataWrittenPerSite/t1,nSites*(dataReadPerSite+dataWrittenPerSite)/t1);
#endif

#ifndef KEEPFIELDONTARGET
  /* get result back from target*/
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
