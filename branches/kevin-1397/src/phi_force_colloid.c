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
#include "kernel.h"
#include "wall.h"
#include "hydro_s.h" 
#include "colloids_s.h"
#include "map_s.h"
#include "pth_s.h"
#include "phi_force_colloid.h"
#include "timer.h"

int phi_force_driver(pth_t * pth, colloids_info_t * cinfo,
		     hydro_t * hydro, map_t * map, wall_t * wall);

__global__
void phi_force_kernel(kernel_ctxt_t * ktx, pth_t * pth,
		      colloids_info_t * cinfo, hydro_t * hydro, map_t * map,
		      wall_t * wall);


/*****************************************************************************
 *
 *  phi_force_colloid
 *
 *  Driver routine. For fractional timesteps, dt < 1.
 *  If no colloids, and no hydrodynamics, no action is required.
 *
 *****************************************************************************/

__host__ int phi_force_colloid(pth_t * pth, fe_t * fe, colloids_info_t * cinfo,
			       hydro_t * hydro, map_t * map, wall_t * wall) {

  int ncolloid;

  assert(pth);

  colloids_info_ntotal(cinfo, &ncolloid);

  if (hydro == NULL && ncolloid == 0) return 0;

  if (pth->method == PTH_METHOD_DIVERGENCE) {
    pth_stress_compute(pth, fe);
    phi_force_driver(pth, cinfo, hydro, map, wall);
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_driver
 *
 *  At solid interfaces use P^th from the adjacent fluid site.
 *
 *  hydro may be null, in which case fluid force is ignored;
 *  however, we still compute the colloid force.
 *
 *****************************************************************************/

__host__ int phi_force_driver(pth_t * pth, colloids_info_t * cinfo,
			      hydro_t * hydro, map_t * map, wall_t * wall) {
  int ia;
  int nlocal[3];
  dim3 nblk, ntpb;
  wall_t * wallt = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(pth);
  assert(cinfo);
  assert(hydro);
  assert(map);

  coords_nlocal(nlocal);
  wall_target(wall, &wallt);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(TIMER_PHI_FORCE_CALC);

  __host_launch(phi_force_kernel, nblk, ntpb, ctxt->target,
		pth->target, cinfo->tcopy, hydro->target, map->target,
		wallt);

  /* note that ideally we would delay this sync to overlap 
   with below colloid force updates, but this is not working on current GPU 
   architectures because colloids are in unified memory */

  targetSynchronize(); 

  kernel_ctxt_free(ctxt);


  /* A separate kernel is requred to allow reduction of the
   * force on each particle. A truly parallel version is
   pending... */

#ifdef __OLD_SHIT__

  /* update colloid-affected lattice sites from target*/
  int ncolsite=colloids_number_sites(cinfo);


  /* allocate space */
  int* colloidSiteList =  (int*) malloc(ncolsite*sizeof(int));

  colloids_list_sites(colloidSiteList,cinfo);


  /* get fluid data from this subset of sites */
  copyFromTargetSubset(pth->str,pth->target->str,colloidSiteList,ncolsite,pth->nsites,9);

  free(colloidSiteList);
#else
  /* Again, pending true device solution */
  pth_memcpy(pth, cudaMemcpyDeviceToHost);
#endif

  colloid_t * pc;
  colloid_link_t * p_link;

  /* All colloids, including halo */
  colloids_info_all_head(cinfo, &pc);
 
  for ( ; pc; pc = pc->nextall) {

    p_link = pc->lnk;

    for (; p_link; p_link = p_link->next) {

      if (p_link->status == LINK_FLUID) {
	int id, p;
	int cmod;

	p = p_link->p;
	cmod = cv[p][X]*cv[p][X] + cv[p][Y]*cv[p][Y] + cv[p][Z]*cv[p][Z];

	if (cmod != 1) continue;
	if (cv[p][X]) id = X;
	if (cv[p][Y]) id = Y;
	if (cv[p][Z]) id = Z;

	for (ia = 0; ia < 3; ia++) {
	  pc->force[ia] += 1.0*cv[p][id]
	    *pth->str[addr_rank2(pth->nsites, 3, 3, p_link->i, ia, id)];
	}
      }
    }
  }

  TIMER_stop(TIMER_PHI_FORCE_CALC);

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_kernel
 *
 *  This computes the force on the fluid, but not the colloids.
 *
 *  cinfo is currently through unified memory.
 *
 *  TODO: The accumulation for the wall momentum is unsafe in shared
 *  memory.
 *
 *****************************************************************************/

__global__
void phi_force_kernel(kernel_ctxt_t * ktx, pth_t * pth,
		      colloids_info_t * cinfo, hydro_t * hydro, map_t * map,
		      wall_t * wall) {

  int kindex;
  __shared__ int kiterations;

  assert(ktx);
  assert(pth);
  assert(cinfo);
  assert(hydro);
  assert(map);

  kiterations = kernel_iterations(ktx);

  __target_simt_parallel_for(kindex, kiterations, 1) {

    int ic, jc, kc;
    int ia, ib;
    int index, index1;

    double pth0[3][3];
    double pth1[3][3];
  
    double force[3];                  /* Accumulated force on fluid */
    double fw[3];                     /* Accumulated force on wall */

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

    /* If this is solid, then there's no contribution here. */

    if (cinfo->map_new[index] == NULL) {

      /* Compute pth at current point */

      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  pth0[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)];
	}
      }

      for (ia = 0; ia < 3; ia++) {
	fw[ia] = 0.0;
      }
      
      /* Compute differences */
      
      index1 = kernel_coords_index(ktx, ic+1, jc, kc);
      
      if (cinfo->map_new[index1]) {
	/* Compute the fluxes at solid/fluid boundary */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -pth0[ia][X];
	}
      }
      else {
	if (map->status[index1] == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -pth0[ia][X];
	    fw[ia] = pth0[ia][X];
	  }
	}
	else {
	  /* This flux is fluid-fluid */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	  }
	}
      }
      
      index1 = kernel_coords_index(ktx, ic-1, jc, kc);
      
      if (cinfo->map_new[index1]) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][X];
	}
      }
      else {
	if (map->status[index1] == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][X];
	    fw[ia] -= pth0[ia][X];
	  }
	}
	else {
	  /* Fluid - fluid */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	  }
	}
      }
      
      index1 = kernel_coords_index(ktx, ic, jc+1, kc);
      
      if (cinfo->map_new[index1]) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= pth0[ia][Y];
	}
      }
      else {
	if (map->status[index1] == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Y];
	    fw[ia] += pth0[ia][Y];
	  }
	}
	else {
	  /* Fluid-fluid */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	  }
	}
      }
      
      index1 = kernel_coords_index(ktx, ic, jc-1, kc);
      
      if (cinfo->map_new[index1]) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][Y];
	}
      }
      else {
	if (map->status[index1] == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Y];
	    fw[ia] -= pth0[ia][Y];
	  }
	}
	else {
	  /* Fluid-fluid */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	  }
	}
      }
      
      index1 = kernel_coords_index(ktx, ic, jc, kc+1);
      
      if (cinfo->map_new[index1]) {
	/* Fluid-solid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= pth0[ia][Z];
	}
      }
      else {
	if (map->status[index1] == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Z];
	    fw[ia] += pth0[ia][Z];
	  }
	}
	else {
	  /* Fluid-fluid */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	  }
	}
      }
      
      index1 = kernel_coords_index(ktx, ic, jc, kc-1);
      
      if (cinfo->map_new[index1]) {
	/* Fluid-solid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][Z];
	}
      }
      else {
	if (map->status[index1] == MAP_BOUNDARY) {
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Z];
	    fw[ia] -= pth0[ia][Z];
	  }
	}
	else {
	  /* Fluid-fluid */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	  }
	}
      }
      
      /* Store the force on lattice */

      for (ia = 0; ia < 3; ia++) {
	hydro->f[addr_rank1(hydro->nsite, NHDIM, index, ia)] += force[ia];
      }

      wall_momentum_add(wall, fw);
    }
    /* Next site */
  }

  return;
}
