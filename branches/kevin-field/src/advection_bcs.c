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

#ifdef OLD_PHI
#include "phi.h"
#else
#include "field.h"
#endif

/*****************************************************************************
 *
 *  advection_bcs_no_normal_fluxes
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

void advection_bcs_no_normal_flux(int nf, double * fluxe, double * fluxw,
				  double * fluxy, double * fluxz) {

  int nlocal[3];
  int ic, jc, kc, index, n;

  double mask, maskw, maske, masky, maskz;

  assert(fluxe);
  assert(fluxw);
  assert(fluxy);
  assert(fluxz);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	mask  = (site_map_get_status_index(index)  == FLUID);
	maske = (site_map_get_status(ic+1, jc, kc) == FLUID);
	maskw = (site_map_get_status(ic-1, jc, kc) == FLUID);
	masky = (site_map_get_status(ic, jc+1, kc) == FLUID);
	maskz = (site_map_get_status(ic, jc, kc+1) == FLUID);

	for (n = 0;  n < nf; n++) {
	  fluxw[nf*index + n] *= mask*maskw;
	  fluxe[nf*index + n] *= mask*maske;
	  fluxy[nf*index + n] *= mask*masky;
	  fluxz[nf*index + n] *= mask*maskz;
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  advective_bcs_no_flux
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

int advective_bcs_no_flux(int nf, double * fx, double * fy, double * fz) {

  int nlocal[3];
  int ic, jc, kc, index, n;

  double mask, maskx, masky, maskz;

  assert(nf > 0);
  assert(fx);
  assert(fy);
  assert(fz);

  coords_nlocal(nlocal);

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	mask  = (site_map_get_status_index(index)  == FLUID);
	maskx = (site_map_get_status(ic+1, jc, kc) == FLUID);
	masky = (site_map_get_status(ic, jc+1, kc) == FLUID);
	maskz = (site_map_get_status(ic, jc, kc+1) == FLUID);

	for (n = 0;  n < nf; n++) {
	  fx[nf*index + n] *= mask*maskx;
	  fy[nf*index + n] *= mask*masky;
	  fz[nf*index + n] *= mask*maskz;
	}

      }
    }
  }

  return 0;
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
  int n, nf;

#ifdef OLD_PHI
  extern double * phi_site;

  if (wall_at_edge(X) == 0) return;

  coords_nlocal(nlocal);
  nf = phi_nop();
#else
  field_t * test_object = NULL;

  if (wall_at_edge(X) == 0) return;

  field_nf(test_object, &nf);
  coords_nlocal(nlocal);
#endif

  if (cart_coords(X) == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index  = coords_index(ic, jc, kc);
	index1 = coords_index(ic-1, jc, kc);

	for (n = 0; n < nf; n++) {
#ifdef OLD_PHI
	  phi_site[nf*index1 + n] = phi_site[nf*index + n];
#else
	  assert(0);
	  /* Here we could use a function to return all as flattened array */ 
#endif

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

	for (n = 0; n < nf; n++) {
#ifdef OLD_PHI
	  phi_site[nf*index1 + n] = phi_site[nf*index + n];
#else
	  assert(0);
	  /* Here we could use a function to return all as flattened array */ 
#endif
	}
      }
    }
  }

  return;
}
