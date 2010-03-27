/****************************************************************************
 *
 *  phi_lb_coupler.c
 *
 *  In cases where the order parameter is via "full LB", this couples
 *  the scalar order parameter phi_site[] to the distributions.
 *
 *  $Id: phi_lb_coupler.c,v 1.2.4.3 2010-03-27 06:17:33 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "physics.h"
#include "site_map.h"
#include "phi.h"
#include "phi_lb_coupler.h"
#include "utilities.h"

extern double * phi_site;

/*****************************************************************************
 *
 *  phi_lb_coupler_phi_set
 *
 *  This is to mediate between order parameter by LB and order parameter
 *  via finite difference.
 *
 *****************************************************************************/

void phi_lb_coupler_phi_set(const int index, const double phi) {

  if (phi_is_finite_difference()) {
    phi_op_set_phi_site(index, 0, phi);
  }
  else {
    assert(distribution_ndist() == 2);
    distribution_zeroth_moment_set_equilibrium(index, 1, phi);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_compute_phi_site
 *
 *  Recompute the value of the order parameter at all the current
 *  fluid sites (domain proper).
 *
 *  This couples the scalar order parameter phi to the LB distribution
 *  in the case of binary LB.
 *
 *****************************************************************************/

void phi_compute_phi_site() {

  int ic, jc, kc, index;
  int nlocal[3];
  int nop;

  if (phi_is_finite_difference()) return;

  assert(distribution_ndist() == 2);

  coords_nlocal(nlocal);
  nop = phi_nop();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = coords_index(ic, jc, kc);
	phi_site[nop*index] = distribution_zeroth_moment(index, 1);
      }
    }
  }

  return;
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

/*****************************************************************************
 *
 *  phi_lb_init_drop
 *
 *  Initialise a drop of radius r and interfacial width xi0 in the
 *  centre of the system.
 *
 *****************************************************************************/

void phi_lb_init_drop(double radius, double xi0) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc, p;

  double position[3];
  double centre[3];
  double phi, r, rxi0;

  assert(distribution_ndist() == 2);

  coords_nlocal(nlocal);
  get_N_offset(noffset);

  rxi0 = 1.0/xi0;

  centre[X] = 0.5*L(X);
  centre[Y] = 0.5*L(Y);
  centre[Z] = 0.5*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
        position[X] = 1.0*(noffset[X] + ic) - centre[X];
        position[Y] = 1.0*(noffset[Y] + jc) - centre[Y];
        position[Z] = 1.0*(noffset[Z] + kc) - centre[Z];

        r = sqrt(dot_product(position, position));

        phi = tanh(rxi0*(r - radius));

	distribution_zeroth_moment_set_equilibrium(index, 0, get_rho0());

        distribution_f_set(index, 0, 1, phi);
        for (p = 1; p < NVEL; p++) {
          distribution_f_set(index, p, 1, 0.0);
        }

      }
    }
  }

  return;
}
