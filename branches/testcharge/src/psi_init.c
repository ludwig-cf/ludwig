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
#include "runtime.h"
#include "psi.h"
#include "psi_s.h"
#include "psi_init.h"

static int gouy_chapman_set(void);
static int liquid_junction_set(void);

int psi_init_charges(void){

  liquid_junction_set();
//  gouy_chapman_set();

  return 0;
}

/*****************************************************************************
 *
 * gouy_chapman_set
 *
 *  Set rho(z = 1)  = + (1/2NxNy)
 *      rho(z = Lz) = + (1/2NxNy)
 *      rho         = - 1/(NxNy*(Nz-2)) + electrolyte
 *
 *  This sets up the system for Gouy-Chapman.
 *
 *****************************************************************************/

static int gouy_chapman_set(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  double rho_w, rho_i, rho_el;

  coords_nlocal(nlocal);

  /* wall charge density */
  rho_w = 1.e+0 / (2.0*L(X)*L(Y));
  rho_el = 1.e-2;

  /* counter charge density */
  rho_i = rho_w * (2.0*L(X)*L(Y)) / (L(X)*L(Y)*(L(Z) - 2.0));

  /* apply counter charges & electrolyte */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(psi_, index, 0.0);
	psi_rho_set(psi_, index, 0, rho_el);
	psi_rho_set(psi_, index, 1, rho_el + rho_i);

      }
    }
  }

  /* apply wall charges */
  if (cart_coords(Z) == 0) {
    kc = 1;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);

	site_map_set_status(ic,jc,kc,BOUNDARY);

	psi_rho_set(psi_, index, 0, rho_w);
	psi_rho_set(psi_, index, 1, 0.0);

      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {
    kc = nlocal[Z];
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);

	site_map_set_status(ic,jc,kc,BOUNDARY);

	psi_rho_set(psi_, index, 0, rho_w);
	psi_rho_set(psi_, index, 1, 0.0);

      }
    }
  }

  site_map_halo();

  return 0;
}

/*****************************************************************************
 *
 * liquid_junction_set
 *
 *  Set rho(1 <= x <= Lx/2)    = 1.01 * electrolyte
 *      rho(Lx/2+1 <= x <= Lx) = 0.99 * electrolyte
 *
 *  This sets up the system for liquid junction potential.
 *
 *****************************************************************************/

static int liquid_junction_set(void) {

  int ic, jc, kc, index;
  int nlocal[3], noff[3];
  double rho_el = 1.e-2, delta_el = 0.001;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noff);

  /* Set electrolyte densities */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(psi_, index, 0.0);

	if ((1 <= noff[0] + ic) && (noff[0] + ic < L(X)/2)) {
	  psi_rho_set(psi_, index, 0, rho_el * (1.0 + delta_el));
	  psi_rho_set(psi_, index, 1, rho_el * (1.0 + delta_el));
	}
	else{
	  psi_rho_set(psi_, index, 0, rho_el * (1.0 - delta_el));
	  psi_rho_set(psi_, index, 1, rho_el * (1.0 - delta_el));
	}
      }
    }
  }


  return 0;
}
