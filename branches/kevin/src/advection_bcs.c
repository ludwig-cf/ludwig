/****************************************************************************
 *
 *  advection_bcs.c
 *
 *  Advection boundary conditions.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "wall.h"
#include "coords.h"
#include "coords_field.h"
#include "advection_s.h"
#include "advection_bcs.h"

/*****************************************************************************
 *
 *  advection_bcs_no_normal_fluxes
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

/* PENDING Lees Edwards only */ 

int advection_bcs_no_normal_flux(int nf, advflux_t * flux, map_t * map) {

  int n;
  int nlocal[3];
  int ic, jc, kc, index;
  int status;

  double mask, maskw, maske, masky, maskz;

  assert(nf > 0);
  assert(flux);
  assert(map);

  le_nlocal(flux->le, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = le_site_index(flux->le, ic-1, jc, kc);
	map_status(map, index, &status);
	maskw = (status == MAP_FLUID);

	index = le_site_index(flux->le, ic+1, jc, kc);
	map_status(map, index, &status);
	maske = (status == MAP_FLUID);

	index = le_site_index(flux->le, ic, jc+1, kc);
	map_status(map, index, &status);
	masky = (status == MAP_FLUID);

	index = le_site_index(flux->le, ic, jc, kc+1);
	map_status(map, index, &status);
	maskz = (status == MAP_FLUID);

	index = le_site_index(flux->le, ic, jc, kc);
	map_status(map, index, &status);
	mask = (status == MAP_FLUID);

	for (n = 0;  n < nf; n++) {
	  flux->fw[nf*index + n] *= mask*maskw;
	  flux->fe[nf*index + n] *= mask*maske;
	  flux->fy[nf*index + n] *= mask*masky;
	  flux->fz[nf*index + n] *= mask*maskz;
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
 *  This allows third and fourth order (x-direction) advective fluxes
 *  to be computed at interface one cell away from wall. Fluxes at
 *  the wall will always be zero.
 *
 ****************************************************************************/

/* PENDING wall (allowed with LE) */

#include "field_s.h"

int advection_bcs_wall(advflux_t * flux, wall_t * wall, field_t * fphi) {

  int ic, jc, kc, index, index1;
  int nlocal[3];
  int nf;
  int iswall[3];
  int cartsz[3];
  int cartcoords[3];

  double q[NQAB];

  assert(flux);
  assert(fphi);
  assert(nf <= NQAB);

  if (wall == NULL) return 0;
  wall_present(wall, iswall);
  if (iswall[X] == 0) return 0;

  assert(iswall[Y] == 0);
  assert(iswall[Z] == 0);

  coords_cartsz(fphi->cs, cartsz);
  coords_cart_coords(fphi->cs, cartcoords);

  field_nf(fphi, &nf);
  coords_nlocal(fphi->cs, nlocal);

  if (cartcoords[X] == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index  = le_site_index(flux->le, ic, jc, kc);
	index1 = le_site_index(flux->le, ic-1, jc, kc);

	field_scalar_array(fphi, index, q);
	field_scalar_array_set(fphi, index1, q);
      }
    }
  }

  if (cartcoords[X] == cartsz[X] - 1) {

    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(flux->le, ic, jc, kc);
	index1 = le_site_index(flux->le, ic+1, jc, kc);

	field_scalar_array(fphi, index, q);
	field_scalar_array_set(fphi, index1, q);

      }
    }
  }

  return 0;
}
