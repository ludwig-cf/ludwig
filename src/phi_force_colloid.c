/*****************************************************************************
 *
 *  phi_force_colloid.c
 *
 *  The case of force from the thermodynamic sector on both fluid and
 *  colloid via the divergence of the chemical stress.
 *
 *  In the absence of solid, the force on the fluid is related
 *  to the divergence of the chemical stress
 *
 *  F_a = - d_b P_ab
 *
 *  Note that the stress is potentially antisymmetric, so this
 *  really is F_a = -d_b P_ab --- not F_a = -d_b P_ba.
 *
 *  The divergence is discretised as, e.g., in the x-direction,
 *  the difference between the interfacial values
 *
 *  d_x P_ab ~= P_ab(x+1/2) - P_ab(x-1/2)
 *
 *  and the interfacial values are based on linear interpolation
 *  P_ab(x+1/2) = 0.5 (P_ab(x) + P_ab(x+1))
 *
 *  etc (and likewise in the other directions).
 *
 *  At an interface, we are not able to interpolate, and we just
 *  use the value of P_ab from the fluid side.
 *
 *  The stress must be integrated over the colloid surface and the
 *  result added to the net force on a given colloid.
 *
 *  The procedure ensures total momentum is conserved, ie., that
 *  leaving the fluid enters the colloid and vice versa.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "wall.h"
#include "free_energy.h"
#include "phi_force.h"
#include "phi_force_stress.h"
#include "phi_force_colloid.h"
#include "hydro_s.h" 
#include "colloids_s.h"
#include "map_s.h"

static int phi_force_interpolation(colloids_info_t * cinfo, hydro_t * hydro,
				   map_t * map);


extern double * pth_;
extern double * t_pth_;

/*****************************************************************************
 *
 *  phi_force_colloid
 *
 *  Driver routine. For fractional timesteps, dt < 1.
 *  If no colloids, and no hydrodynamics, no action is required.
 *
 *****************************************************************************/

__targetHost__ int phi_force_colloid(colloids_info_t * cinfo, field_t* q, field_grad_t* q_grad, hydro_t * hydro, map_t * map) {

  int ncolloid;
  int required;

  phi_force_required(&required);
  colloids_info_ntotal(cinfo, &ncolloid);

  if (hydro == NULL && ncolloid == 0) required = 0;

  if (required) {

    phi_force_stress_allocate();

    phi_force_stress_compute(q, q_grad);
    phi_force_interpolation(cinfo, hydro, map);

    phi_force_stress_free();
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_interpolation
 *
 *  At solid interfaces use P^th from the adjacent fluid site.
 *
 *  hydro may be null, in which case fluid force is ignored;
 *  however, we still compute the colloid force.
 *
 *****************************************************************************/



/* For the current colloid implementation, the below atomicAdd function
 * is needed to update colloids when the lattice is parallelised 
 * across threads
 * TO DO: properly push this into targetDP, 
 * or replace with more performant implementation
 */

#ifdef __NVCC__

/* from https://devtalk.nvidia.com/default/topic/529341/speed-of-double-precision-cuda-atomic-operations-on-kepler-k20/ */

__device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#else
double atomicAdd(double* address, double val)
{
  
  double old=*address;

  /* TO DO: uncomment below pragma when OpenMP implemenation is active
   * #pragma omp atomic */
  *address+=val;

  return old;
}
#endif

__targetEntry__ void phi_force_interpolation_lattice(colloids_info_t * cinfo, hydro_t * hydro, map_t * map, double* t_pth) {



  int index;
  __targetTLPNoStride__(index,tc_nSites){

  int ia, ib, ic, jc, kc;
  int index1;
  int nlocal[3];
  int status;
  
  double pth0[3][3];
  double pth1[3][3];

  double force[3];                  /* Accumulated force on fluid */
  double fw[3];                     /* Accumulated force on wall */
  
  colloid_t * p_c;
  colloid_t * colloid_at_site_index(int);

  
  int coords[3];
  targetCoords3D(coords,tc_Nall,index);
  
  /*  if not a halo site:*/
    if (coords[0] >= (tc_nhalo) && 
	coords[1] >= (tc_nhalo) && 
	coords[2] >= (tc_nhalo) &&
	coords[0] < tc_Nall[X]-(tc_nhalo) &&  
	coords[1] < tc_Nall[Y]-(tc_nhalo)  &&  
	coords[2] < tc_Nall[Z]-(tc_nhalo) ){ 


    int coords[3];
    targetCoords3D(coords,tc_Nall,index);


    /* If this is solid, then there's no contribution here. */
    
    p_c = cinfo->map_new[index];

    if (!p_c){

      /* Compute pth at current point */
      for (ia = 0; ia < 3; ia++) 
	for (ib = 0; ib < 3; ib++) 
	  pth0[ia][ib]=t_pth[PTHADR(tc_nSites,index,ia,ib)];
      
      for (ia = 0; ia < 3; ia++) {
	fw[ia] = 0.0;
      }
      
      /* Compute differences */
      
      index1 = targetIndex3D(coords[0]+1,coords[1],coords[2],tc_Nall);	
      
      p_c = cinfo->map_new[index1];
      
      if (p_c) {
	/* Compute the fluxes at solid/fluid boundary */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -pth0[ia][X];
	  atomicAdd( &p_c->force[ia] , pth0[ia][X]);
	}
      }
      else {
	status=map->status[index1];	
	if (status == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -pth0[ia][X];
	    fw[ia] = pth0[ia][X];
	  }
	}
	else {
	  /* This flux is fluid-fluid */ 
	  for (ia = 0; ia < 3; ia++) 
	    for (ib = 0; ib < 3; ib++) 
	      pth1[ia][ib]=t_pth[PTHADR(tc_nSites,index1,ia,ib)];
	  
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	  }
	}
      }
      
      index1 = targetIndex3D(coords[0]-1,coords[1],coords[2],tc_Nall);	
      
      p_c = cinfo->map_new[index1];
      
      if (p_c) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][X];
	  atomicAdd( &p_c->force[ia] , -pth0[ia][X]);
	}
      }
      else {
	status=map->status[index1];	
	if (status == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][X];
	    fw[ia] -= pth0[ia][X];
	  }
	}
	else {
	  /* Fluid - fluid */
	  for (ia = 0; ia < 3; ia++) 
	    for (ib = 0; ib < 3; ib++) 
	      pth1[ia][ib]=t_pth[PTHADR(tc_nSites,index1,ia,ib)];
	  
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	  }
	}
      }
      
      index1 = targetIndex3D(coords[0],coords[1]+1,coords[2],tc_Nall);	

      p_c = cinfo->map_new[index1];
      
      if (p_c) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= pth0[ia][Y];
	  atomicAdd( &p_c->force[ia] , pth0[ia][Y]);
	}
      }
      else {
	status=map->status[index1];	
	if (status == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Y];
	    fw[ia] += pth0[ia][Y];
	  }
	}
	else {
	  /* Fluid-fluid */
	  for (ia = 0; ia < 3; ia++) 
	    for (ib = 0; ib < 3; ib++) 
	      pth1[ia][ib]=t_pth[PTHADR(tc_nSites,index1,ia,ib)];
	  
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	  }
	}
      }
      
      index1 = targetIndex3D(coords[0],coords[1]-1,coords[2],tc_Nall);	

      p_c = cinfo->map_new[index1];
      
      if (p_c) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][Y];
	  atomicAdd( &p_c->force[ia] , -pth0[ia][Y]);
	}
      }
      else {
	status=map->status[index1];	
	if (status == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Y];
	    fw[ia] -= pth0[ia][Y];
	  }
	}
	else {
	  /* Fluid-fluid */
	  for (ia = 0; ia < 3; ia++) 
	    for (ib = 0; ib < 3; ib++) 
	      pth1[ia][ib]=t_pth[PTHADR(tc_nSites,index1,ia,ib)];
	  
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	  }
	}
      }
      
      index1 = targetIndex3D(coords[0],coords[1],coords[2]+1,tc_Nall);	

      p_c = cinfo->map_new[index1];
      
      if (p_c) {
	/* Fluid-solid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= pth0[ia][Z];
	  atomicAdd( &p_c->force[ia] , pth0[ia][Z]);
	}
      }
      else {
	status=map->status[index1];	
	if (status == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Z];
	    fw[ia] += pth0[ia][Z];
	  }
	}
	else {
	  /* Fluid-fluid */
	  for (ia = 0; ia < 3; ia++) 
	    for (ib = 0; ib < 3; ib++) 
	      pth1[ia][ib]=t_pth[PTHADR(tc_nSites,index1,ia,ib)];
	  
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	  }
	}
      }
      
      index1 = targetIndex3D(coords[0],coords[1],coords[2]-1,tc_Nall);	

      p_c = cinfo->map_new[index1];
      
      if (p_c) {
	/* Fluid-solid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][Z];
	  atomicAdd( &p_c->force[ia] , -pth0[ia][Z]);
	}
      }
      else {
	status=map->status[index1];	
	if (status == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Z];
	    fw[ia] -= pth0[ia][Z];
	  }
	}
	else {
	  /* Fluid-fluid */
	  for (ia = 0; ia < 3; ia++) 
	    for (ib = 0; ib < 3; ib++) 
	      pth1[ia][ib]=t_pth[PTHADR(tc_nSites,index1,ia,ib)];
	  
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	  }
	}
      }
      
      /* Store the force on lattice */
      for (ia = 0; ia < 3; ia++) 
	hydro->f[HYADR(tc_nSites,hydro->nf,index,ia)] += force[ia];
      
      /*TO DO
       * from wall.c 
       * "This is for accounting purposes only.
       * There is no physical consequence." */

#ifndef __NVCC__
      wall_accumulate_force(fw);
#endif
      
    }
    /* Next site */
    }
  }
  
  
  return;
}


static int phi_force_interpolation(colloids_info_t * cinfo, hydro_t * hydro,
				   map_t * map) {
  int ia, ic, jc, kc;
  int index, index1;
  int nlocal[3];
  int status;

  double pth0[3][3];
  double pth1[3][3];
  double force[3];                  /* Accumulated force on fluid */
  double fw[3];                     /* Accumulated force on wall */

  colloid_t * p_c;
  colloid_t * colloid_at_site_index(int);

  void (* chemical_stress)(const int index, double s[3][3]);

  assert(cinfo);
  assert(map);

  coords_nlocal(nlocal);

  chemical_stress = phi_force_stress;

  int nhalo = coords_nhalo();
  int Nall[3];
  int nSites;
  Nall[X] = nlocal[X] + 2*nhalo;
  Nall[Y] = nlocal[Y] + 2*nhalo;
  Nall[Z] = nlocal[Z] + 2*nhalo;
  nSites  = Nall[X]*Nall[Y]*Nall[Z];


  /* set up constants on target */
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int)); 

  /*  copy stress to target */
#ifndef KEEPFIELDONTARGET    
  copyToTarget(t_pth_,pth_,3*3*nSites*sizeof(double));      
#endif

  hydro_t* t_hydro = hydro->tcopy; 
  double* tmpptr;
#ifndef KEEPHYDROONTARGET
  copyFromTarget(&tmpptr,&(t_hydro->f),sizeof(double*)); 
  copyToTarget(tmpptr,hydro->f,hydro->nf*nSites*sizeof(double));
#endif

  /* map_t* t_map = map->tcopy;
   * populate target copy of map from host 
   * copyFromTarget(&tmpptr,&(t_map->status),sizeof(char*)); 
   * copyToTarget(tmpptr,map->status,nSites*sizeof(char));

   *  set up colloids such that they can be accessed from target
   *  noting that each actual colloid structure stays resident on the host
   *   if (cinfo->map_new){
   * colloids_info_t* t_cinfo=cinfo->tcopy;
   * colloid_t* tmpcol;
   * copyFromTarget(&tmpcol,&(t_cinfo->map_new),sizeof(colloid_t**)); 
   * copyToTarget(tmpcol,cinfo->map_new,nSites*sizeof(colloid_t*));
   * }
   */

  /* launch operation across the lattice on target */

  phi_force_interpolation_lattice  __targetLaunchNoStride__(nSites) (cinfo->tcopy, hydro->tcopy,  map->tcopy, t_pth_);
  targetSynchronize();

#ifndef KEEPHYDROONTARGET
  copyFromTarget(&tmpptr,&(t_hydro->f),sizeof(double*)); 
  copyFromTarget(hydro->f,tmpptr,hydro->nf*nSites*sizeof(double));
#endif

  return 0;
}
