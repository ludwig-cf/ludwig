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
 *  (c) 2010-2016 The University of Edinburgh
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
#include "phi_force_colloid.h"
#include "hydro_s.h" 
#include "colloids_s.h"
#include "map_s.h"
#include "pth_s.h"
#include "timer.h"

static int phi_force_interpolation(pth_t * pth, colloids_info_t * cinfo,
				   hydro_t * hydro,
				   map_t * map);

/*****************************************************************************
 *
 *  phi_force_colloid
 *
 *  Driver routine. For fractional timesteps, dt < 1.
 *  If no colloids, and no hydrodynamics, no action is required.
 *
 *****************************************************************************/

__targetHost__ int phi_force_colloid(pth_t * pth, colloids_info_t * cinfo,
				     field_t * q, field_grad_t * q_grad,
				     hydro_t * hydro, map_t * map) {

  int ncolloid;

  assert(pth);

  colloids_info_ntotal(cinfo, &ncolloid);

  if (hydro == NULL && ncolloid == 0) return 0;

  if (pth->method == PTH_METHOD_DIVERGENCE) {
    phi_force_stress_compute(pth, q, q_grad);
    phi_force_interpolation(pth, cinfo, hydro, map);
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



__targetEntry__
void phi_force_interpolation_lattice(pth_t * pth, colloids_info_t * cinfo,
				     hydro_t * hydro,
				     map_t * map) {

  int index;

  __targetTLPNoStride__(index, tc_nSites) {

  int ia, ib;

  int index1;
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

      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  pth0[ia][ib] = pth->str[addr_rank2(tc_nSites,3,3,index,ia,ib)];
	}
      }

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

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(tc_nSites,3,3,index1,ia,ib)];
	    }
	  }

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

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(tc_nSites,3,3,index1,ia,ib)];
	    }
	  }

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

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(tc_nSites,3,3,index1,ia,ib)];
	    }
	  }

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

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(tc_nSites,3,3,index1,ia,ib)];
	    }
	  }

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

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(tc_nSites,3,3,index1,ia,ib)];
	    }
	  }

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

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(tc_nSites,3,3,index1,ia,ib)];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	  }
	}
      }
      
      /* Store the force on lattice */

      /* Can we re-encapsulate this? Only one instance */
      for (ia = 0; ia < 3; ia++) {
	hydro->f[addr_hydro(index, ia)] += force[ia];
      }

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



static int phi_force_interpolation(pth_t * pth, colloids_info_t * cinfo,
				   hydro_t * hydro,
				   map_t * map) {
  int ia;
  int nlocal[3];

  colloid_t * colloid_at_site_index(int);

  assert(cinfo);
  assert(map);

  coords_nlocal(nlocal);

  int nhalo = coords_nhalo();
  int Nall[3];
  int nSites;
  Nall[X] = nlocal[X] + 2*nhalo;
  Nall[Y] = nlocal[Y] + 2*nhalo;
  Nall[Z] = nlocal[Z] + 2*nhalo;
  nSites  = Nall[X]*Nall[Y]*Nall[Z];


  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int)); 


  TIMER_start(TIMER_PHI_FORCE_CALC);

  phi_force_interpolation_lattice  __targetLaunch__(nSites) (pth->target, cinfo->tcopy, hydro->tcopy, map->target);

/* note that ideally we would delay this sync to overlap 
   with below colloid force updates, but this is not working on current GPU 
   architectures because colloids are in unified memory */
  targetSynchronize(); 



#ifdef __NVCC__

  /* update colloid-affected lattice sites from target*/
  int ncolsite=colloids_number_sites(cinfo);


  /* allocate space */
  int* colloidSiteList =  (int*) malloc(ncolsite*sizeof(int));

  /* populate list with all site indices */
  colloids_list_sites(colloidSiteList,cinfo);


  /* get fluid data from this subset of sites */
  copyFromTargetSubset(pth,t_pth_,colloidSiteList,ncolsite,nSites,9);

  free(colloidSiteList);

#endif

  colloid_t * pc;
  colloid_link_t * p_link;

  /* All colloids, including halo */
  colloids_info_all_head(cinfo, &pc);
 
  for ( ; pc; pc = pc->nextall) {

    p_link = pc->lnk;

    for (; p_link; p_link = p_link->next) {

      if (p_link->status == LINK_UNUSED) continue;

      int coordsOutside[3];
      coords_index_to_ijk(p_link->i,coordsOutside);

      int coordsInside[3];
      coords_index_to_ijk(p_link->j,coordsInside);

      int hopsApart=0;

      /* only include those inside-outside pairs that fall
	 on th 7-point stencil */

      for (ia = 0; ia < 3; ia++) 
	hopsApart+=abs(coordsOutside[ia]-coordsInside[ia]);

      if (hopsApart>1) continue;


      /* work out which X,Y or Z direction they are neighbours in
	 and the +ve or -ve direction */
      int idir=0,fac=0;
      if ((coordsOutside[0]-coordsInside[0])==1){
	idir=0;fac=-1;
      }
      if ((coordsOutside[0]-coordsInside[0])==-1){
	idir=0;fac=1;
      }
      if ((coordsOutside[1]-coordsInside[1])==1){
	idir=1;fac=-1;
      }
      if ((coordsOutside[1]-coordsInside[1])==-1){
	idir=1;fac=1;
      }
      if ((coordsOutside[2]-coordsInside[2])==1){
	idir=2;fac=-1;
      }
      if ((coordsOutside[2]-coordsInside[2])==-1){
	idir=2;fac=1;
      }

      /* update the colloid force using the relavent part of the potential */
	for (ia = 0; ia < 3; ia++) {
	  pc->force[ia] += fac*pth->str[addr_rank2(nSites,3,3,p_link->i,ia,idir)];
	}

    }
  }


  TIMER_stop(TIMER_PHI_FORCE_CALC);

  return 0;
}
