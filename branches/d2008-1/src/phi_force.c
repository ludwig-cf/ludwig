/*****************************************************************************
 *
 *  phi_force.c
 *
 *  Computes the force on the fluid from the thermodynamic sector
 *  via the divergence of the chemical stress. Its calculation as
 *  a divergence ensures momentum is conserved.
 *
 *  $Id: phi_force.c,v 1.1.2.2 2008-02-26 17:11:09 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "phi.h"
#include "free_energy.h"

/*****************************************************************************
 *
 *  phi_force_calculation_fluid
 *
 *  Compute force from thermodynamic sector via
 *    F_alpha = nalba_beta Pth_alphabeta
 *
 *****************************************************************************/

void phi_force_calculation_fluid_six() {

  int ia, ic, jc, kc;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];
  double force[3];

  void add_force_at_site(const int, double *);

  get_N_local(nlocal);
  assert(nhalo_ >= 2);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	/* Compute pth at current point */
	free_energy_get_chemical_stress(index, pth0);

	/* Compute differences */
	
	index1 = get_site_index(ic+1, jc, kc);
	free_energy_get_chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[X][ia] + pth0[X][ia]);
	}
	index1 = get_site_index(ic-1, jc, kc);
	free_energy_get_chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[X][ia] + pth0[X][ia]);
	}

	
	index1 = get_site_index(ic, jc+1, kc);
	free_energy_get_chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[Y][ia] + pth0[Y][ia]);
	}
	index1 = get_site_index(ic, jc-1, kc);
	free_energy_get_chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[Y][ia] + pth0[Y][ia]);
	}
	
	index1 = get_site_index(ic, jc, kc+1);
	free_energy_get_chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[Z][ia] + pth0[Z][ia]);
	}
	index1 = get_site_index(ic, jc, kc-1);
	free_energy_get_chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[Z][ia] + pth0[Z][ia]);
	}

	/* Store the force on lattice */

	add_force_at_site(index, force);

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_calculation_fluid
 *
 *  Compute force from thermodynamic sector via
 *    F_alpha = nalba_beta Pth_alphabeta
 *
 *****************************************************************************/

void phi_force_calculation_fluid() {

  int p, ia, ib, ic, jc, kc, ic1, jc1, kc1;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];
  double pdiffs[NVEL][3][3];
  double gradpth[3][3];
  double force[3];
  double r10 = 0.1;

  void add_force_at_site(const int, double *);

  get_N_local(nlocal);
  assert(nhalo_ >= 2);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	/* Compute pth at current point */
	free_energy_get_chemical_stress(index, pth0);

	/* Compute differences */

	for (p = 1; p < NVEL; p++) {

	  /* Compute pth1 at target point */

	  ic1 = ic + cv[p][X];
	  jc1 = jc + cv[p][Y];
	  kc1 = kc + cv[p][Z];
	  index1 = get_site_index(ic1, jc1, kc1);
	  free_energy_get_chemical_stress(index1, pth1);

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      pdiffs[p][ia][ib] = pth1[ia][ib] - pth0[ia][ib];
	    }
	  }
	}

	/* Accumulate the differences */

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    gradpth[ia][ib] = 0.0;
	  }
	}

	for (p = 1; p < NVEL; p++) {
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      gradpth[ia][ib] += cv[p][ib]*pdiffs[p][ia][ib];
	    }
	  }
	}

	/* Compute the force */
	
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = 0.0;
	  for (ib = 0; ib < 3; ib++) {
	    force[ia] -= r10*gradpth[ia][ib];
	  }
	}

	/* Store the force on lattice */

	add_force_at_site(index, force);

	/* Next site */
      }
    }
  }

  return;
}
