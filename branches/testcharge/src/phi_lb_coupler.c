/****************************************************************************
 *
 *  phi_lb_coupler.c
 *
 *  In cases where the order parameter is via "full LB", this couples
 *  the scalar order parameter phi_site[] to the distributions.
 *
 *  $Id: phi_lb_coupler.c,v 1.3 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "site_map.h"
#include "phi_lb_coupler.h"

/*****************************************************************************
 *
 *  phi_lb_to_field
 *
 *****************************************************************************/

int phi_lb_to_field(field_t * phi) {

  int ic, jc, kc, index;
  int nlocal[3];

  double phi0;

  assert(phi);
  assert(distribution_ndist() == 2);
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	phi0 = distribution_zeroth_moment(index, 1);
	field_scalar_set(phi, index, phi0);

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_lb_from_field
 *
 *  Move the scalar order parameter into the non-propagating part
 *  of the distribution, and set other elements of distribution to
 *  zero.
 *
 *****************************************************************************/

int phi_lb_from_field(field_t * phi) {

  int p;
  int ic, jc, kc, index;
  int nlocal[3];

  double phi0;

  assert(phi);
  assert(distribution_ndist() == 2);
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	field_scalar(phi, index, &phi0);

	distribution_f_set(index, 0, 1, phi0);
	for (p = 1; p < NVEL; p++) {
	  distribution_f_set(index, p, 1, 0.0);
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_set_mean_phi
 *
 *  Compute the current mean phi in the system and remove the excess
 *  so that the mean phi is phi_global (allowing for presence of any
 *  particles or, for that matter, other solids).
 *
 *  The value of phi_global is generally (but not necessilarily) zero.
 *
 *****************************************************************************/

void phi_set_mean_phi(double phi_global) {

  int      index, ic, jc, kc;
  int      nlocal[3];
  double   phi_local = 0.0, phi_total, phi_correction;
  double   vlocal = 0.0, vtotal;
  MPI_Comm comm = cart_comm();

  assert(0); /* This should take a field_t argument */
  assert(distribution_ndist() == 2);

  coords_nlocal(nlocal);

  /* Compute the mean phi in the domain proper */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = coords_index(ic, jc, kc);
	phi_local += distribution_zeroth_moment(index, 1);
	vlocal += 1.0;
      }
    }
  }

  /* All processes need the total phi, and number of fluid sites
   * to compute the mean */

  MPI_Allreduce(&phi_local, &phi_total, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&vlocal, &vtotal,   1, MPI_DOUBLE,    MPI_SUM, comm);

  /* The correction requied at each fluid site is then ... */
  phi_correction = phi_global - phi_total / vtotal;

  /* The correction is added to the rest distribution g[0],
   * which should be good approximation to where it should
   * all end up if there were a full reprojection. */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) == FLUID) {
	  index = coords_index(ic, jc, kc);
	  phi_local = distribution_f(index, 0, 1) + phi_correction;
	  distribution_f_set(index, 0, 1, phi_local);
	}
      }
    }
  }

  return;
}
