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
 *  (c) 2010-2016 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "lb_model_s.h"
#include "field_s.h"
#include "phi_lb_coupler.h"

__global__ void phi_lb_to_field_kernel(field_t * phi, lb_t * lb);

/*****************************************************************************
 *
 *  phi_lb_to_field
 *
 *  Driver function: compute from the distribution the current
 *  values of phi and store.
 *
 *****************************************************************************/

__host__ int phi_lb_to_field(field_t * phi, lb_t  * lb) {

  int Nall[3];
  int nlocal[3];
  int nSites;
  int nhalo = coords_nhalo();

  assert(phi);
  assert(lb);

  coords_nlocal(nlocal);

  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;

  nSites = le_nsites();

  int nDist;
  copyFromTarget(&nDist,&(lb->ndist),sizeof(int)); 

  copyConstToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(&tc_ndist,&nDist, sizeof(int)); 

  phi_lb_to_field_kernel __targetLaunchNoStride__(nSites) (phi->target, lb->target);

  return 0;
}

/*****************************************************************************
 *
 *  phi_lb_to_field_kernel
 *
 *  Kernel: 1->nlocal[] each direction.
 *
 *****************************************************************************/

__global__ void phi_lb_to_field_kernel(field_t * phi, lb_t * lb) {

  int kindex;

  __targetTLPNoStride__(kindex, tc_nSites) {	  

    int p;
    int coords[3];
    double phi0;

    targetCoords3D(coords,tc_Nall, kindex);

    /* if not a halo site:*/
    if (coords[0] >= tc_nhalo &&
	coords[1] >= tc_nhalo &&
	coords[2] >= tc_nhalo &&
	coords[0] < tc_Nall[X]-tc_nhalo &&
	coords[1] < tc_Nall[Y]-tc_nhalo &&
	coords[2] < tc_Nall[Z]-tc_nhalo){
    
      phi0 = 0.0;
      for (p = 0; p < NVEL; p++) {
	phi0 += lb->f[LB_ADDR(tc_nSites, tc_ndist, NVEL, kindex, LB_PHI, p)];
      }

      phi->data[addr_rank0(tc_nSites, kindex)] = phi0;    
    }
  }

  return;
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
