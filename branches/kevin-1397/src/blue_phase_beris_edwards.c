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
#include "advection.h"
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

/*IMPORTANT NOTE*/

/* the below routine is a COPY of that in blue_phase.h */
/* required to be in scope here for performance reasons on GPU */
/* since otherwise the compiler places the emporary q[][][] etc arrays */
/* in regular off-chip memory rather than registers */
/* which has a huge impact on performance */
/* TO DO: place this in a header file to be included both here and blue_phase.c */
/* or work out how to get the compiler to inline it from the different source file */

__targetHost__ __target__ void blue_phase_compute_h_inline(double q[3][3], 
						    double dq[3][3][3],
						    double dsq[3][3], 
						    double h[3][3],
						    bluePhaseKernelConstants_t* pbpc) {
  int ia, ib, ic, id;

  double q2;
  double e2;
  double eq;
  double sum;

  /* From the bulk terms in the free energy... */

  q2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += q[ia][ib]*q[ia][ib];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
  	sum += q[ia][ic]*q[ib][ic];
      }
      h[ia][ib] = -pbpc->a0_*(1.0 - pbpc->r3_*pbpc->gamma_)*q[ia][ib]
  	+ pbpc->a0_*pbpc->gamma_*(sum - pbpc->r3_*q2*pbpc->d_[ia][ib]) - pbpc->a0_*pbpc->gamma_*q2*q[ia][ib];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  eq = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
  	eq += pbpc->e_[ib][ic][ia]*dq[ib][ic][ia];
      }
    }
  }

  /* d_c Q_db written as d_c Q_bd etc */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
       sum = 0.0;
       for (ic = 0; ic < 3; ic++) {
       for (id = 0; id < 3; id++) {
	   //sum += pbpc->e_[ia][ic][id]*dq[ic][ib][id] + pbpc->e_[ib][ic][id]*dq[ic][ia][id];
	 sum += pbpc->ec_[ia][ic][id]*dq[ic][ib][id] + pbpc->ec_[ib][ic][id]*dq[ic][ia][id];
       }
       }
       
       h[ia][ib] += pbpc->kappa0*dsq[ia][ib]
      - 2.0*pbpc->kappa1*pbpc->q0*sum + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq*pbpc->d_[ia][ib]
      - 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[ia][ib];
      
    }
  }

  /* Electric field term */

  e2 = 0.0;
  for (ia = 0; ia < 3; ia++) {
    e2 += pbpc->e0[ia]*pbpc->e0[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] +=  pbpc->epsilon_*(pbpc->e0[ia]*pbpc->e0[ib] - pbpc->r3_*pbpc->d_[ia][ib]*e2);
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

__targetEntry__
void blue_phase_be_update_lattice(double* __restrict__ qdata,
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

  int index;

  __targetTLPNoStride__(index,tc_nSites){


  int ia, ib, id;
  int indexj, indexk;
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

  int coords[3];
  targetCoords3D(coords,tc_Nall,index);
  
  /* if not a halo site:*/
    if (coords[0] >= (tc_nhalo) &&
	coords[1] >= (tc_nhalo) &&
	coords[2] >= (tc_nhalo) &&
	coords[0] < (tc_Nall[X]-tc_nhalo) &&
	coords[1] < (tc_Nall[Y]-tc_nhalo)  &&
	coords[2] < (tc_Nall[Z]-tc_nhalo) ){

      
      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  s[ia][ib] = 0.0;
	  chi_qab[ia][ib] = 0.0;
	}
      }
      


#ifndef __NVCC__
      /* on gpu we will just calc all sites (and discard non-fluid results)*/
      map_status(map, index, &status);
      if (status != MAP_FLUID) continue;
#endif

      /* calculate molecular field	*/

	int ia, ib;
#ifndef OLD_SHIT
	q[X][X] = qdata[addr_qab(tc_nSites, index, XX)];
	q[X][Y] = qdata[addr_qab(tc_nSites, index, XY)];
	q[X][Z] = qdata[addr_qab(tc_nSites, index, XZ)];
	q[Y][X] = q[X][Y];
	q[Y][Y] = qdata[addr_qab(tc_nSites, index, YY)];
	q[Y][Z] = qdata[addr_qab(tc_nSites, index, YZ)];
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];

	for (ia = 0; ia < NVECTOR; ia++) {
	  dq[ia][X][X] = graddata[addr_rank2(tc_nSites,NQAB,3, index,XX,ia)];
	  dq[ia][X][Y] = graddata[addr_rank2(tc_nSites,NQAB,3, index,XY,ia)];
	  dq[ia][X][Z] = graddata[addr_rank2(tc_nSites,NQAB,3, index,XZ,ia)];
	  dq[ia][Y][X] = dq[ia][X][Y];
	  dq[ia][Y][Y] = graddata[addr_rank2(tc_nSites,NQAB,3, index,YY,ia)];
	  dq[ia][Y][Z] = graddata[addr_rank2(tc_nSites,NQAB,3, index,YZ,ia)];
	  dq[ia][Z][X] = dq[ia][X][Z];
	  dq[ia][Z][Y] = dq[ia][Y][Z];
	  dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
	}
#else
	q[X][X] = qdata[FLDADR(tc_nSites,NQAB,index,XX)];
	q[X][Y] = qdata[FLDADR(tc_nSites,NQAB,index,XY)];
	q[X][Z] = qdata[FLDADR(tc_nSites,NQAB,index,XZ)];
	q[Y][X] = q[X][Y];
	q[Y][Y] = qdata[FLDADR(tc_nSites,NQAB,index,YY)];
	q[Y][Z] = qdata[FLDADR(tc_nSites,NQAB,index,YZ)];
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];

	for (ia = 0; ia < NVECTOR; ia++) {
	  dq[ia][X][X] = graddata[FGRDADR(tc_nSites,NQAB,index,XX,ia)];
	  dq[ia][X][Y] = graddata[FGRDADR(tc_nSites,NQAB,index,XY,ia)];
	  dq[ia][X][Z] = graddata[FGRDADR(tc_nSites,NQAB,index,XZ,ia)];
	  dq[ia][Y][X] = dq[ia][X][Y];
	  dq[ia][Y][Y] = graddata[FGRDADR(tc_nSites,NQAB,index,YY,ia)];
	  dq[ia][Y][Z] = graddata[FGRDADR(tc_nSites,NQAB,index,YZ,ia)];
	  dq[ia][Z][X] = dq[ia][X][Z];
	  dq[ia][Z][Y] = dq[ia][Y][Z];
	  dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
	}
#endif	
#ifndef OLD_SHIT
	dsq[X][X] = graddelsq[addr_rank1(tc_nSites, NQAB, index, XX)];
	dsq[X][Y] = graddelsq[addr_rank1(tc_nSites, NQAB, index, XY)];
	dsq[X][Z] = graddelsq[addr_rank1(tc_nSites, NQAB, index, XZ)];
	dsq[Y][X] = dsq[X][Y];
	dsq[Y][Y] = graddelsq[addr_rank1(tc_nSites, NQAB, index, YY)];
	dsq[Y][Z] = graddelsq[addr_rank1(tc_nSites, NQAB, index, YZ)];
	dsq[Z][X] = dsq[X][Z];
	dsq[Z][Y] = dsq[Y][Z];
	dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];
#else
	dsq[X][X] = graddelsq[FLDADR(tc_nSites,NQAB,index,XX)];
	dsq[X][Y] = graddelsq[FLDADR(tc_nSites,NQAB,index,XY)];
	dsq[X][Z] = graddelsq[FLDADR(tc_nSites,NQAB,index,XZ)];
	dsq[Y][X] = dsq[X][Y];
	dsq[Y][Y] = graddelsq[FLDADR(tc_nSites,NQAB,index,YY)];
	dsq[Y][Z] = graddelsq[FLDADR(tc_nSites,NQAB,index,YZ)];
	dsq[Z][X] = dsq[X][Z];
	dsq[Z][Y] = dsq[Y][Z];
	dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];
#endif
	


	if (isBPMF)
	  blue_phase_compute_h_inline(q, dq, dsq, h, pbpc);
	else
	{
#ifndef __NVCC__
	    /*only BP supported for CUDA. This is caught earlier*/
	  molecular_field(index, h);
#endif
		  }
	
	if (hydroOn) {
#ifndef OLD_SHIT
	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

	  /* hydro_u_gradient_tensor(hydro, ic, jc, kc, w);
	   * inline above function
	   * TODO add lees edwards support*/

	  int im1 = targetIndex3D(coords[X]-1,coords[Y],coords[Z],tc_Nall);
	  int ip1 = targetIndex3D(coords[X]+1,coords[Y],coords[Z],tc_Nall);
	  
	  w[X][X] = 0.5*(hydrou[addr_hydro(ip1, X)]
		       - hydrou[addr_hydro(im1, X)]);
	  w[Y][X] = 0.5*(hydrou[addr_hydro(ip1, Y)]
		       - hydrou[addr_hydro(im1, Y)]);
	  w[Z][X] = 0.5*(hydrou[addr_hydro(ip1, Z)]
		       - hydrou[addr_hydro(im1, Z)]);
	  
	  im1 = targetIndex3D(coords[X],coords[Y]-1,coords[Z],tc_Nall);
	  ip1 = targetIndex3D(coords[X],coords[Y]+1,coords[Z],tc_Nall);
	  
	  w[X][Y] = 0.5*(hydrou[addr_hydro(ip1, X)]
		       - hydrou[addr_hydro(im1, X)]);
	  w[Y][Y] = 0.5*(hydrou[addr_hydro(ip1, Y)]
		       - hydrou[addr_hydro(im1, Y)]);
	  w[Z][Y] = 0.5*(hydrou[addr_hydro(ip1, Z)]
		       - hydrou[addr_hydro(im1, Z)]);
	  
	  im1 = targetIndex3D(coords[X],coords[Y],coords[Z]-1,tc_Nall);
	  ip1 = targetIndex3D(coords[X],coords[Y],coords[Z]+1,tc_Nall);
	  
	  w[X][Z] = 0.5*(hydrou[addr_hydro(ip1, X)]
		       - hydrou[addr_hydro(im1, X)]);
	  w[Y][Z] = 0.5*(hydrou[addr_hydro(ip1, Y)]
		       - hydrou[addr_hydro(im1, Y)]);
	  w[Z][Z] = 0.5*(hydrou[addr_hydro(ip1, Z)]
		       - hydrou[addr_hydro(im1, Z)]);
#else
	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

	  /* hydro_u_gradient_tensor(hydro, ic, jc, kc, w);
	   * inline above function
	   * TODO add lees edwards support*/

	  int im1 = targetIndex3D(coords[X]-1,coords[Y],coords[Z],tc_Nall);
	  int ip1 = targetIndex3D(coords[X]+1,coords[Y],coords[Z],tc_Nall);
	  
	  w[X][X] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,X)] - hydrou[HYADR(tc_nSites,3,im1,X)]);
	  w[Y][X] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,Y)] - hydrou[HYADR(tc_nSites,3,im1,Y)]);
	  w[Z][X] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,Z)] - hydrou[HYADR(tc_nSites,3,im1,Z)]);
	  
	  im1 = targetIndex3D(coords[X],coords[Y]-1,coords[Z],tc_Nall);
	  ip1 = targetIndex3D(coords[X],coords[Y]+1,coords[Z],tc_Nall);
	  
	  w[X][Y] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,X)] - hydrou[HYADR(tc_nSites,3,im1,X)]);
	  w[Y][Y] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,Y)] - hydrou[HYADR(tc_nSites,3,im1,Y)]);
	  w[Z][Y] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,Z)] - hydrou[HYADR(tc_nSites,3,im1,Z)]);
	  
	  im1 = targetIndex3D(coords[X],coords[Y],coords[Z]-1,tc_Nall);
	  ip1 = targetIndex3D(coords[X],coords[Y],coords[Z]+1,tc_Nall);
	  
	  w[X][Z] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,X)] - hydrou[HYADR(tc_nSites,3,im1,X)]);
	  w[Y][Z] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,Y)] - hydrou[HYADR(tc_nSites,3,im1,Y)]);
	  w[Z][Z] = 0.5*(hydrou[HYADR(tc_nSites,3,ip1,Z)] - hydrou[HYADR(tc_nSites,3,im1,Z)]);
#endif
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

#ifdef __NVCC__
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
	  
	indexj=targetIndex3D(coords[X],coords[Y]-1,coords[Z],tc_Nall);
	indexk=targetIndex3D(coords[X],coords[Y],coords[Z]-1,tc_Nall);
#ifndef OLD_SHIT
	q[X][X] += dt*(s[X][X] + tc_gamma*h[X][X] + chi_qab[X][X]
		       - fluxe[addr_rank1(tc_nSites, nf, index, XX)]
		       + fluxw[addr_rank1(tc_nSites, nf, index, XX)]
		       - fluxy[addr_rank1(tc_nSites, nf, index, XX)]
		       + fluxy[addr_rank1(tc_nSites, nf, indexj,XX)]
		       - fluxz[addr_rank1(tc_nSites, nf, index, XX)]
		       + fluxz[addr_rank1(tc_nSites, nf, indexk,XX)]);

	q[X][Y] += dt*(s[X][Y] + tc_gamma*h[X][Y] + chi_qab[X][Y]
		       - fluxe[addr_rank1(tc_nSites, nf, index, XY)]
		       + fluxw[addr_rank1(tc_nSites, nf, index, XY)]
		       - fluxy[addr_rank1(tc_nSites, nf, index, XY)]
		       + fluxy[addr_rank1(tc_nSites, nf, indexj,XY)]
		       - fluxz[addr_rank1(tc_nSites, nf, index, XY)]
		       + fluxz[addr_rank1(tc_nSites, nf, indexk,XY)]);

	q[X][Z] += dt*(s[X][Z] + tc_gamma*h[X][Z] + chi_qab[X][Z]
		       - fluxe[addr_rank1(tc_nSites, nf, index, XZ)]
		       + fluxw[addr_rank1(tc_nSites, nf, index, XZ)]
		       - fluxy[addr_rank1(tc_nSites, nf, index, XZ)]
		       + fluxy[addr_rank1(tc_nSites, nf, indexj,XZ)]
		       - fluxz[addr_rank1(tc_nSites, nf, index, XZ)]
		       + fluxz[addr_rank1(tc_nSites, nf, indexk,XZ)]);

	q[Y][Y] += dt*(s[Y][Y] + tc_gamma*h[Y][Y] + chi_qab[Y][Y]
		       - fluxe[addr_rank1(tc_nSites, nf, index, YY)]
		       + fluxw[addr_rank1(tc_nSites, nf, index, YY)]
		       - fluxy[addr_rank1(tc_nSites, nf, index, YY)]
		       + fluxy[addr_rank1(tc_nSites, nf, indexj,YY)]
		       - fluxz[addr_rank1(tc_nSites, nf, index, YY)]
		       + fluxz[addr_rank1(tc_nSites, nf, indexk,YY)]);

	q[Y][Z] += dt*(s[Y][Z] + tc_gamma*h[Y][Z] + chi_qab[Y][Z]
		       - fluxe[addr_rank1(tc_nSites, nf, index, YZ)]
		       + fluxw[addr_rank1(tc_nSites, nf, index, YZ)]
		       - fluxy[addr_rank1(tc_nSites, nf, index, YZ)]
		       + fluxy[addr_rank1(tc_nSites, nf, indexj,YZ)]
		       - fluxz[addr_rank1(tc_nSites, nf, index, YZ)]
		       + fluxz[addr_rank1(tc_nSites, nf, indexk,YZ)]);
#else
	q[X][X] += dt*(s[X][X] + tc_gamma*h[X][X] + chi_qab[X][X]
		       - fluxe[ADVADR(tc_nSites,nf,index,XX)] + fluxw[ADVADR(tc_nSites,nf,index,XX)]
		       - fluxy[ADVADR(tc_nSites,nf,index,XX)] + fluxy[ADVADR(tc_nSites,nf,indexj,XX)]
		       - fluxz[ADVADR(tc_nSites,nf,index,XX)] + fluxz[ADVADR(tc_nSites,nf,indexk,XX)]);


	q[X][Y] += dt*(s[X][Y] + tc_gamma*h[X][Y] + chi_qab[X][Y]
		       - fluxe[ADVADR(tc_nSites,nf,index,XY)] + fluxw[ADVADR(tc_nSites,nf,index,XY)]
		       - fluxy[ADVADR(tc_nSites,nf,index,XY)] + fluxy[ADVADR(tc_nSites,nf,indexj,XY)]
		       - fluxz[ADVADR(tc_nSites,nf,index,XY)] + fluxz[ADVADR(tc_nSites,nf,indexk,XY)]);

	q[X][Z] += dt*(s[X][Z] + tc_gamma*h[X][Z] + chi_qab[X][Z]
		       - fluxe[ADVADR(tc_nSites,nf,index,XZ)] + fluxw[ADVADR(tc_nSites,nf,index,XZ)]
		       - fluxy[ADVADR(tc_nSites,nf,index,XZ)] + fluxy[ADVADR(tc_nSites,nf,indexj,XZ)]
		       - fluxz[ADVADR(tc_nSites,nf,index,XZ)] + fluxz[ADVADR(tc_nSites,nf,indexk,XZ)]);

	q[Y][Y] += dt*(s[Y][Y] + tc_gamma*h[Y][Y] + chi_qab[Y][Y]
		       - fluxe[ADVADR(tc_nSites,nf,index,YY)] + fluxw[ADVADR(tc_nSites,nf,index,YY)]
		       - fluxy[ADVADR(tc_nSites,nf,index,YY)] + fluxy[ADVADR(tc_nSites,nf,indexj,YY)]
		       - fluxz[ADVADR(tc_nSites,nf,index,YY)] + fluxz[ADVADR(tc_nSites,nf,indexk,YY)]);

	q[Y][Z] += dt*(s[Y][Z] + tc_gamma*h[Y][Z] + chi_qab[Y][Z]
		       - fluxe[ADVADR(tc_nSites,nf,index,YZ)] + fluxw[ADVADR(tc_nSites,nf,index,YZ)]
		       - fluxy[ADVADR(tc_nSites,nf,index,YZ)] + fluxy[ADVADR(tc_nSites,nf,indexj,YZ)]
		       - fluxz[ADVADR(tc_nSites,nf,index,YZ)] + fluxz[ADVADR(tc_nSites,nf,indexk,YZ)]);
#endif
#ifndef OLD_SHIT
	qdata[addr_qab(tc_nSites, index, XX)] = q[X][X];
	qdata[addr_qab(tc_nSites, index, XY)] = q[X][Y];
	qdata[addr_qab(tc_nSites, index, XZ)] = q[X][Z];
	qdata[addr_qab(tc_nSites, index, YY)] = q[Y][Y];
	qdata[addr_qab(tc_nSites, index, YZ)] = q[Y][Z];
#else
	qdata[FLDADR(tc_nSites,NQAB,index,XX)] = q[X][X];
	qdata[FLDADR(tc_nSites,NQAB,index,XY)] = q[X][Y];
	qdata[FLDADR(tc_nSites,NQAB,index,XZ)] = q[X][Z];
	qdata[FLDADR(tc_nSites,NQAB,index,YY)] = q[Y][Y];
	qdata[FLDADR(tc_nSites,NQAB,index,YZ)] = q[Y][Z];
#endif
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

  double* hydrou;
  double* qdata;
  double* graddata;
  double*  graddelsq;
  double*  fluxe;
  double*  fluxw;
  double*  fluxy;
  double*  fluxz;

  copyFromTarget(&hydrou,&(hydro->tcopy->u),sizeof(double*)); 
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
