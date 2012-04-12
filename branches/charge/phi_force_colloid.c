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
 *  The stress must be integrated over the colloid surface and the
 *  result added to the net force on a given colloid. Here the linear
 *  interpolation of the stress to the solid/fluid interface is again
 *  used. There are two options:
 *
 *  1) If the interior stress is available, use the value;
 *  2) If no interior stress is available, interpolate using zero.
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
#include "lattice.h"
#include "colloids.h"
#include "site_map.h"
#include "wall.h"
#include "colloids_Q_tensor.h"
#include "free_energy.h"

static double * pth_; 

static void phi_force_interpolation1(void);
static void phi_force_interpolation2(void);
static void phi_force_stress(const int index, double p[3][3]);
static void phi_force_stress_set(const int index, double p[3][3]);
static void phi_force_stress_compute(void);
static void phi_force_fast(void);

/*****************************************************************************
 *
 *  phi_force_colloid
 *
 *  Driver routine.
 *
 *****************************************************************************/

void phi_force_colloid(void) {

  phi_force_fast();

  return;
}

/*****************************************************************************
 *
 *  phi_force_fast
 *
 *****************************************************************************/

static void phi_force_fast(void) {

  int n;

  assert(coords_nhalo() >= 2);

  n = coords_nsites();

  pth_ = (double *) malloc(9*n*sizeof(double));
  if (pth_ == NULL) fatal("malloc(pth_) failed\n");

  phi_force_stress_compute();

  if (colloids_q_anchoring_method() == ANCHORING_METHOD_ONE) {
    phi_force_interpolation1();
  }
  else {
    phi_force_interpolation2();
  }

  free(pth_);

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_compute
 *
 *  Compute the stress everywhere and store.
 *
 *****************************************************************************/

static void phi_force_stress_compute(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nextra = 1;

  double pth_local[3][3];
  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  chemical_stress = fe_chemical_stress_function();

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);

	chemical_stress(index, pth_local);
	phi_force_stress_set(index, pth_local);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_set
 *
 *****************************************************************************/

static void phi_force_stress_set(const int index, double p[3][3]) {

  int ia, ib, n;

  assert(pth_);

  n = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      pth_[9*index + n++] = p[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress
 *
 *****************************************************************************/

static void phi_force_stress(const int index, double p[3][3]) {

  int ia, ib, n;

  assert(pth_);

  n = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = pth_[9*index + n++];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_interpolation1
 *
 *  We assume values of the stress inside the particle are
 *  available.
 *
 *****************************************************************************/

static void phi_force_interpolation1(void) {

  int ia, ic, jc, kc;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];
  double force[3];

  colloid_t * p_c;
  colloid_t * colloid_at_site_index(int);

  void (* chemical_stress)(const int index, double s[3][3]);

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

	/* Compute differences */
	
	index1 = coords_index(ic+1, jc, kc);
	chemical_stress(index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Compute the fluxes at solid/fluid boundary */

	  for (ia = 0; ia < 3; ia++) {
	    p_c->force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	  }
	}

	index1 = coords_index(ic-1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  for (ia = 0; ia < 3; ia++) {
	    p_c->force[ia] -= 0.5*(pth1[ia][X] + pth0[ia][X]);
	  }
	}



	index1 = coords_index(ic, jc+1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}

	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  for (ia = 0; ia < 3; ia++) {
	    p_c->force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	  }
	}

	index1 = coords_index(ic, jc-1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}

	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  for (ia = 0; ia < 3; ia++) {
	    p_c->force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	  }
	}


	index1 = coords_index(ic, jc, kc+1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  for (ia = 0; ia < 3; ia++) {
	    p_c->force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	  }
	}

	index1 = coords_index(ic, jc, kc-1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  for (ia = 0; ia < 3; ia++) {
	    p_c->force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	  }
	}

	/* Store the force on lattice */

	hydrodynamics_add_force_local(index, force);

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_interpolation2
 *
 *  General version which deals with solid interfaces by using P^th from
 *  the adjacent fluid site at the interface.
 *
 *****************************************************************************/

static void phi_force_interpolation2(void) {

  int ia, ic, jc, kc;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];
  double force[3];                  /* Accumulated force on fluid */
  double fw[3];                     /* Accumulated force on wall */

  colloid_t * p_c;
  colloid_t * colloid_at_site_index(int);

  void (* chemical_stress)(const int index, double s[3][3]);

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
	    force[ia] = -pth0[ia][X];
	    p_c->force[ia] += pth0[ia][X];
	  }
	}
	else if (site_map_get_status_index(index1) == BOUNDARY) {
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

	index1 = coords_index(ic-1, jc, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][X];
	    p_c->force[ia] -= pth0[ia][X];
	  }
	}
	else if (site_map_get_status_index(index1) == BOUNDARY) {
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

	index1 = coords_index(ic, jc+1, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Y];
	    p_c->force[ia] += pth0[ia][Y];
	  }
	}
	else if (site_map_get_status_index(index1) == BOUNDARY) {
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

	index1 = coords_index(ic, jc-1, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Y];
	    p_c->force[ia] -= pth0[ia][Y];
	  }
	}
	else if (site_map_get_status_index(index1) == BOUNDARY) {
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
	
	index1 = coords_index(ic, jc, kc+1);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Fluid-solid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Z];
	    p_c->force[ia] += pth0[ia][Z];
	  }
	}
	else if (site_map_get_status_index(index1) == BOUNDARY) {
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

	index1 = coords_index(ic, jc, kc-1);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Fluid-solid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Z];
	    p_c->force[ia] -= pth0[ia][Z];
	  }
	}
	else if (site_map_get_status_index(index1) == BOUNDARY) {
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

	/* Store the force on lattice */

	hydrodynamics_add_force_local(index, force);
	wall_accumulate_force(fw);

	/* Next site */
      }
    }
  }

  return;
}
