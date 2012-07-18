/****************************************************************************
 *
 *  advection_bcs.c
 *
 *  Advection boundary conditions.
 *
 *  $Id: advection_bcs.c,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "wall.h"
#include "coords.h"
#include "site_map.h"
#include "phi.h"

/*****************************************************************************
 *
 *  advection_bcs_no_normal_fluxes
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

void advection_bcs_no_normal_flux(double * fluxe, double * fluxw,
				  double * fluxy, double * fluxz) {

  int nlocal[3];
  int ic, jc, kc, index, n;
  int nop;

  double mask, maskw, maske, masky, maskz;

  assert(fluxe);
  assert(fluxw);
  assert(fluxy);
  assert(fluxz);

  coords_nlocal(nlocal);
  nop = phi_nop();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	mask  = (site_map_get_status_index(index)  == FLUID);
	maske = (site_map_get_status(ic+1, jc, kc) == FLUID);
	maskw = (site_map_get_status(ic-1, jc, kc) == FLUID);
	masky = (site_map_get_status(ic, jc+1, kc) == FLUID);
	maskz = (site_map_get_status(ic, jc, kc+1) == FLUID);

	for (n = 0;  n < nop; n++) {
	  fluxw[nop*index + n] *= mask*maskw;
	  fluxe[nop*index + n] *= mask*maske;
	  fluxy[nop*index + n] *= mask*masky;
	  fluxz[nop*index + n] *= mask*maskz;
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  advection_bcs_wall
 *
 *  For the case of flat walls, we kludge the order parameter advection
 *  by borrowing the adjacent fluid value.
 *
 *  The official explanation is this may be viewed as a no gradient
 *  condition on the order parameter near the wall.
 *
 *  This will be effective for fluxes up to fourth order.
 *
 ****************************************************************************/

void advection_bcs_wall(void) {

  int ic, jc, kc, index, index1;
  int nlocal[3];
  int n, nop;

  extern double * phi_site;

  if (wall_at_edge(X) == 0) return;

  coords_nlocal(nlocal);
  nop = phi_nop();

  if (cart_coords(X) == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index  = coords_index(ic, jc, kc);
	index1 = coords_index(ic-1, jc, kc);

	for (n = 0; n < nop; n++) {
	  phi_site[nop*index1 + n] = phi_site[nop*index + n];
	}
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {

    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	index1 = coords_index(ic+1, jc, kc);

	for (n = 0; n < nop; n++) {
	  phi_site[nop*index1 + n] = phi_site[nop*index + n];
	}
      }
    }
  }

  return;
}
