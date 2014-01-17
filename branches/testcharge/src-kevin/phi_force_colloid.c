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
#include "colloids.h"
#include "wall.h"
#include "free_energy.h"
#include "phi_force.h"
#include "phi_force_stress.h"
#include "phi_force_colloid.h"


#ifdef OLD_ONLY

static int phi_force_interpolation(hydro_t * hydro, map_t * map, double dt);

/*****************************************************************************
 *
 *  phi_force_colloid
 *
 *  Driver routine.
 *
 *****************************************************************************/

int phi_force_colloid(hydro_t * hydro, map_t * map, double dt) {

  int required;

  phi_force_required(&required);

  if (required) {

    phi_force_stress_allocate();

    phi_force_stress_compute();
    phi_force_interpolation(hydro, map, dt);

    phi_force_stress_free();
  }

  return 0;
}
#else

static int phi_force_interpolation(colloids_info_t * cinfo, hydro_t * hydro,
				   map_t * map);


int phi_force_colloid(colloids_info_t * cinfo, hydro_t * hydro, map_t * map) {

  int required;

  phi_force_required(&required);

  if (required) {

    phi_force_stress_allocate();

    phi_force_stress_compute();
    phi_force_interpolation(cinfo, hydro, map);

    phi_force_stress_free();
  }

  return 0;
}
#endif

/*****************************************************************************
 *
 *  phi_force_interpolation
 *
 *  At solid interfaces use P^th from the adjacent fluid site.
 *
 *****************************************************************************/
#ifdef OLD_ONLY
static int phi_force_interpolation(hydro_t * hydro, map_t * map, double dt) {

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

  assert(hydro);
  assert(map);

  coords_nlocal(nlocal);

  chemical_stress = phi_force_stress;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	/* If this is solid, then there's no contribution here. */
	p_c = colloid_at_site_index(index);
	if (p_c) continue;

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fw[ia] = 0.0;
	}

	/* Compute differences */
	
	index1 = coords_index(ic+1, jc, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Compute the fluxes at solid/fluid boundary */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -pth0[ia][X]*dt;
	    p_c->force[ia] += pth0[ia][X]*dt;
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] = -pth0[ia][X]*dt;
	      fw[ia] = pth0[ia][X]*dt;
	    }
	  }
	  else {
	    /* This flux is fluid-fluid */ 
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X])*dt;
	    }
	  }
	}

	index1 = coords_index(ic-1, jc, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][X]*dt;
	    p_c->force[ia] -= pth0[ia][X]*dt;
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += pth0[ia][X]*dt;
	      fw[ia] -= pth0[ia][X]*dt;
	    }
	  }
	  else {
	    /* Fluid - fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X])*dt;
	    }
	  }
	}

	index1 = coords_index(ic, jc+1, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Y]*dt;
	    p_c->force[ia] += pth0[ia][Y]*dt;
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= pth0[ia][Y]*dt;
	      fw[ia] += pth0[ia][Y]*dt;
	    }
	  }
	  else {
	    /* Fluid-fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y])*dt;
	    }
	  }
	}

	index1 = coords_index(ic, jc-1, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Y]*dt;
	    p_c->force[ia] -= pth0[ia][Y]*dt;
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += pth0[ia][Y]*dt;
	      fw[ia] -= pth0[ia][Y]*dt;
	    }
	  }
	  else {
	    /* Fluid-fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y])*dt;
	    }
	  }
	}
	
	index1 = coords_index(ic, jc, kc+1);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Fluid-solid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Z]*dt;
	    p_c->force[ia] += pth0[ia][Z]*dt;
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= pth0[ia][Z]*dt;
	      fw[ia] += pth0[ia][Z]*dt;
	    }
	  }
	  else {
	    /* Fluid-fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z])*dt;
	    }
	  }
	}

	index1 = coords_index(ic, jc, kc-1);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Fluid-solid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Z]*dt;
	    p_c->force[ia] -= pth0[ia][Z]*dt;
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += pth0[ia][Z]*dt;
	      fw[ia] -= pth0[ia][Z]*dt;
	    }
	  }
	  else {
	    /* Fluid-fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z])*dt;
	    }
	  }
	}

	/* Store the force on lattice */

	hydro_f_local_add(hydro, index, force);
	wall_accumulate_force(fw);

	/* Next site */
      }
    }
  }

  return 0;
}
#else
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
  assert(hydro);
  assert(map);

  coords_nlocal(nlocal);

  chemical_stress = phi_force_stress;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	/* If this is solid, then there's no contribution here. */

	colloids_info_map(cinfo, index, &p_c);
	if (p_c) continue;

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fw[ia] = 0.0;
	}

	/* Compute differences */
	
	index1 = coords_index(ic+1, jc, kc);
	colloids_info_map(cinfo, index1, &p_c);

	if (p_c) {
	  /* Compute the fluxes at solid/fluid boundary */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -pth0[ia][X];
	    p_c->force[ia] += pth0[ia][X];
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] = -pth0[ia][X];
	      fw[ia] = pth0[ia][X];
	    }
	  }
	  else {
	    /* This flux is fluid-fluid */ 
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	    }
	  }
	}

	index1 = coords_index(ic-1, jc, kc);
	colloids_info_map(cinfo, index1, &p_c);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][X];
	    p_c->force[ia] -= pth0[ia][X];
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += pth0[ia][X];
	      fw[ia] -= pth0[ia][X];
	    }
	  }
	  else {
	    /* Fluid - fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	    }
	  }
	}

	index1 = coords_index(ic, jc+1, kc);
	colloids_info_map(cinfo, index1, &p_c);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Y];
	    p_c->force[ia] += pth0[ia][Y];
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= pth0[ia][Y];
	      fw[ia] += pth0[ia][Y];
	    }
	  }
	  else {
	    /* Fluid-fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	    }
	  }
	}

	index1 = coords_index(ic, jc-1, kc);
	colloids_info_map(cinfo, index1, &p_c);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Y];
	    p_c->force[ia] -= pth0[ia][Y];
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += pth0[ia][Y];
	      fw[ia] -= pth0[ia][Y];
	    }
	  }
	  else {
	    /* Fluid-fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	    }
	  }
	}
	
	index1 = coords_index(ic, jc, kc+1);
	colloids_info_map(cinfo, index1, &p_c);

	if (p_c) {
	  /* Fluid-solid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Z];
	    p_c->force[ia] += pth0[ia][Z];
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= pth0[ia][Z];
	      fw[ia] += pth0[ia][Z];
	    }
	  }
	  else {
	    /* Fluid-fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	    }
	  }
	}

	index1 = coords_index(ic, jc, kc-1);
	colloids_info_map(cinfo, index1, &p_c);

	if (p_c) {
	  /* Fluid-solid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Z];
	    p_c->force[ia] -= pth0[ia][Z];
	  }
	}
	else {
	  map_status(map, index1, &status);
	  if (status == MAP_BOUNDARY) {
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += pth0[ia][Z];
	      fw[ia] -= pth0[ia][Z];
	    }
	  }
	  else {
	    /* Fluid-fluid */
	    chemical_stress(index1, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	    }
	  }
	}

	/* Store the force on lattice */

	hydro_f_local_add(hydro, index, force);
	wall_accumulate_force(fw);

	/* Next site */
      }
    }
  }

  return 0;
}
#endif
