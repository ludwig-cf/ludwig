/****************************************************************************
 *
 *  phi_lb_coupler.c
 *
 *  In cases where the order parameter is via "full LB", this couples
 *  the scalar order parameter field to the 'g' distribution.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2014 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "lb_model_s.h"
#include "field_s.h"
#include "phi_lb_coupler.h"

/*****************************************************************************
 *
 *  phi_lb_to_field
 *
 *****************************************************************************/

__target__ int phi_lb_to_field_site(double * phi, double * f, const int baseIndex) {

  int iv=0; 

  double phi0[VVL];

  __targetILP__(iv) phi0[iv]=0.;


#if VVL == 1
  /*restrict operation to the interiour lattice sites*/
  int coords[3];
  targetCoords3D(coords,tc_Nall,baseIndex);
  
  // if not a halo site:
  if (coords[0] >= tc_nhalo &&
      coords[1] >= tc_nhalo &&
      coords[2] >= tc_nhalo &&
      coords[0] < tc_Nall[X]-tc_nhalo &&
      coords[1] < tc_Nall[Y]-tc_nhalo &&
      coords[2] < tc_Nall[Z]-tc_nhalo)
#endif


    {
    
    
    int p;
     for (p = 0; p < NVEL; p++) {
      __targetILP__(iv) phi0[iv] += f[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex+iv, LB_PHI, p)];
    }
    
     __targetILP__(iv) phi[baseIndex+iv]=phi0[iv];
    
    
    }
  

  return 0;
}


__targetEntry__ void phi_lb_to_field_lattice(double * phi, lb_t * lb) {

  int baseIndex=0;

  __targetTLP__(baseIndex,tc_nSites){	  
    phi_lb_to_field_site(phi, lb->f, baseIndex);
  }


  return;
}



__targetHost__ int phi_lb_to_field(field_t * phi, lb_t  *lb) {

  int Nall[3];
  int nlocal[3];
  int nSites;
  int nhalo = coords_nhalo();

  assert(phi);
  assert(lb);

  coords_nlocal(nlocal);

  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;

  nSites = Nall[X]*Nall[Y]*Nall[Z];

  int nDist;
  copyFromTarget(&nDist,&(lb->ndist),sizeof(int)); 

  //start constant setup
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(&tc_ndist,&nDist, sizeof(int)); 
  //end constant setup

  phi_lb_to_field_lattice __targetLaunch__(nSites) (phi->t_data, lb);

#ifndef KEEPFIELDONTARGET   //temporary optimisation specific to GPU code for benchmarking
  copyFromTarget(phi->data,phi->t_data,nSites*sizeof(double)); 
#endif

  return 0;
}



/* Host-only version of the above */
__targetHost__ int phi_lb_to_field_host(field_t * phi, lb_t  *lb) {

  int ic, jc, kc, index;
  int nlocal[3];

  double phi0;

  assert(phi);
  assert(lb);
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	lb_0th_moment(lb, index, LB_PHI, &phi0);
	field_scalar_set(phi, index, phi0);

      }
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  phi_lb_from_field
 *
 *  Move the scalar order parameter into the non-propagating part
 *  of the distribution, and set other elements of distribution to
 *  zero.
 *
 *****************************************************************************/

__targetHost__ int phi_lb_from_field(field_t * phi, lb_t * lb) {

  int p;
  int ic, jc, kc, index;
  int nlocal[3];

  double phi0;

  assert(phi);
  assert(lb);
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	field_scalar(phi, index, &phi0);

	lb_f_set(lb, index, 0, 1, phi0);
	for (p = 1; p < NVEL; p++) {
	  lb_f_set(lb, index, p, 1, 0.0);
	}

      }
    }
  }

  return 0;
}
