/*****************************************************************************
 *
 *  phi_force_colloid.c
 *
 *  The special case of force from the thermodynamic sector on
 *  both fluid and colloid via the divergence of the chemical
 *  stress.
 *
 *  The stress must effectively be integrated over the colloid
 *  surface and the result added to the net force on a given
 *  colloid.
 *
 *  This is isolated in this file owing to the dependency on the
 *  colloid structure.
 *
 *  $Id: phi_force_colloid.c,v 1.1.2.1 2009-11-13 14:49:28 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

#include "pe.h"
#include "coords.h"
#include "lattice.h"
#include "colloids.h"
#include "free_energy.h"

/*****************************************************************************
 *
 *  phi_force_colloid
 *
 *  In the abscence of solid, the force on the fluid is related
 *  to the divergence of the chemical stress
 *
 *  F_a = - d_b P_ab
 *
 *  The divergence is discretised as, e.g., in the x-direction,
 *  the difference between the interfacial values
 *
 *  d_x P_xb ~= P_xb(x+1/2) - P_xb(x-1/2)
 *
 *  and the interfacial values are based on linear interpolation
 *  P_xb(x+1/2) = 0.5 (P_xb(x) + P_xb(x+1))
 *
 *  etc (and likewise in the other directions).
 *
 *  In the presence of solid, P_th is assumed to be zero, and the
 *  calculation procedes as above, but allowing the momentum flux
 *  at solid-fluid faces to be transfered as a force on the colloid.
 *
 *  This is equivalent to integrating the surface stress around the
 *  discrete particle.
 *
 *  In this way, total momentum in the system (solid+fluid) is conserved.
 *
 *****************************************************************************/

void phi_force_colloid(void) {

  int ia, ic, jc, kc;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];
  double force[3];

  Colloid * p_c;
  Colloid * colloid_at_site_index(int);

  void (* chemical_stress)(const int index, double s[3][3]);

  get_N_local(nlocal);
  assert(nhalo_ >= 2);

  chemical_stess = fe_chemical_stress_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	/* If this is solid, then there's no contribution here. */
	p_c = colloid_at_site_index(index);
	if (p_c) continue;

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	/* Compute differences */
	
	index1 = get_site_index(ic+1, jc, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Compute the fluxes at solid/fluid boundary */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -0.5*pth0[X][ia];
	  }

	  p_c->force.x += 0.5*pth0[X][X];
	  p_c->force.y += 0.5*pth0[X][Y];
	  p_c->force.z += 0.5*pth0[X][Z];
	}
	else {
	  /* This flux is fluid-fluid */ 
	  chemical_stress(index1, pth1);
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -0.5*(pth1[X][ia] + pth0[X][ia]);
	  }
	}

	index1 = get_site_index(ic-1, jc, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*pth0[X][ia];
	  }

	  p_c->force.x -= 0.5*pth0[X][X];
	  p_c->force.y -= 0.5*pth0[X][Y];
	  p_c->force.z -= 0.5*pth0[X][Z];
	}
	else {
	  /* Fluid - fluid */
	  chemical_stress(index1, pth1);
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[X][ia] + pth0[X][ia]);
	  }
	}

	index1 = get_site_index(ic, jc+1, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= 0.5*pth0[Y][ia];
	  }

	  p_c->force.x += 0.5*pth0[Y][X];
	  p_c->force.y += 0.5*pth0[Y][Y];
	  p_c->force.z += 0.5*pth0[Y][Z];
	}
	else {
	  /* Fluid-fluid */
	  chemical_stress(index1, pth1);
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= 0.5*(pth1[Y][ia] + pth0[Y][ia]);
	  }
	}

	index1 = get_site_index(ic, jc-1, kc);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*pth0[Y][ia];
	  }

	  p_c->force.x -= 0.5*pth0[Y][X];
	  p_c->force.y -= 0.5*pth0[Y][Y];
	  p_c->force.z -= 0.5*pth0[Y][Z];
	}
	else {
	  /* Fluid-fluid */
	  chemical_stress(index1, pth1);
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[Y][ia] + pth0[Y][ia]);
	  }
	}
	
	index1 = get_site_index(ic, jc, kc+1);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Fluid-solid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= 0.5*pth0[Z][ia];
	  }

	  p_c->force.x += 0.5*pth0[Z][X];
	  p_c->force.y += 0.5*pth0[Z][Y];
	  p_c->force.z += 0.5*pth0[Z][Z];
	}
	else {
	  /* Fluid-fluid */
	  chemical_stress(index1, pth1);
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= 0.5*(pth1[Z][ia] + pth0[Z][ia]);
	  }
	}

	index1 = get_site_index(ic, jc, kc-1);
	p_c = colloid_at_site_index(index1);

	if (p_c) {
	  /* Fluid-solid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*pth0[Z][ia];
	  }

	  p_c->force.x -= 0.5*pth0[Z][X];
	  p_c->force.y -= 0.5*pth0[Z][Y];
	  p_c->force.z -= 0.5*pth0[Z][Z];
	}
	else {
	  /* Fluid-fluid */
	  chemical_stress(index1, pth1);
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += 0.5*(pth1[Z][ia] + pth0[Z][ia]);
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
