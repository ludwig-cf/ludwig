/*****************************************************************************
*
*  psi_init.c
*
*  Various initial states for electrokinetics.
*
*  $Id$
*
*  Edinburgh Soft Matter and Statistical Physics Group and
*  Edinburgh Parallel Computing Centre
*
*  Oliver Henrich (o.henrich@ucl.ac.uk) wrote most of these.
*  (c) 2012 The University of Edinburgh
*
*****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
 
#include "pe.h"
#include "coords.h"
#include "site_map.h"
#include "psi.h"
#include "psi_init.h"

/*****************************************************************************
 *
 *  psi_init_uniform
 *
 *  Set the charge density for all species to be rho_el everywhere.
 *  The potential is initialised to zero.
 *
 *****************************************************************************/

int psi_init_uniform(psi_t * obj, double rho_el) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nk;

  assert(obj);
  assert(rho_el >= 0.0);

  coords_nlocal(nlocal);
  psi_nk(obj, &nk);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(obj, index, 0.0);

	for (n = 0; n < nk; n++) {
	  psi_rho_set(obj, index, n, rho_el);
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * psi_init_gouy_chapman_set
 *
 *  Set rho(x = 1)  = + (1/2NyNz)
 *      rho(x = Lx) = + (1/2NyNz)
 *      rho         = - 1/(NyNz*(Nx-2)) + electrolyte
 *
 *  This sets up the system for Gouy-Chapman.
 *
 *  rho_el is the electrolyte (background) charge density.
 *  sigma is the sufrace charge density at the wall.
 *
 *****************************************************************************/

int psi_init_gouy_chapman_set(psi_t * obj, double rho_el, double sigma) {

  int ic, jc, kc, index;
  int nlocal[3];
  double rho_w, rho_i;

  assert(obj);

  coords_nlocal(nlocal);

  /* wall surface charge density */
  rho_w = sigma;

  /* counter charge density */
  rho_i = rho_w * 2.0 *L(Y)*L(Z) / (L(Y)*L(Z)*(L(X) - 2.0));

  /* apply counter charges & electrolyte */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(obj, index, 0.0);
	psi_rho_set(obj, index, 0, rho_el);
	psi_rho_set(obj, index, 1, rho_el + rho_i);

      }
    }
  }

  /* apply wall charges */
  if (cart_coords(X) == 0) {
    ic = 1;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	site_map_set_status(ic,jc,kc,BOUNDARY);

	psi_rho_set(obj, index, 0, rho_w);
	psi_rho_set(obj, index, 1, 0.0);

      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {
    ic = nlocal[X];
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	site_map_set_status(ic,jc,kc,BOUNDARY);

	psi_rho_set(obj, index, 0, rho_w);
	psi_rho_set(obj, index, 1, 0.0);

      }
    }
  }

  site_map_halo();

  return 0;
}

/*****************************************************************************
 *
 *  psi_init_liquid_junction_set
 *
 *  Set rho(1 <= x <= Lx/2)    = 1.01 * electrolyte
 *      rho(Lx/2+1 <= x <= Lx) = 0.99 * electrolyte
 *
 *  This sets up the system for liquid junction potential.
 *
 *  rho_el is the average electrolyte concentration.
 *  delta_el is the relative difference of the concentrations.
 *
 *****************************************************************************/

int psi_init_liquid_junction_set(psi_t * obj, double rho_el, double delta_el) {

  int ic, jc, kc, index;
  int nlocal[3], noff[3];

  assert(obj);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noff);

  /* Set electrolyte densities */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(obj, index, 0.0);

	if ((1 <= noff[0] + ic) && (noff[0] + ic < N_total(X)/2)) {
	  psi_rho_set(obj, index, 0, rho_el * (1.0 + delta_el));
	  psi_rho_set(obj, index, 1, rho_el * (1.0 + delta_el));
	}
	else{
	  psi_rho_set(obj, index, 0, rho_el * (1.0 - delta_el));
	  psi_rho_set(obj, index, 1, rho_el * (1.0 - delta_el));
	}
      }
    }
  }

  return 0;
}
