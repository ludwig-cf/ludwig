/****************************************************************************
 *
 *  phi_lb_coupler.c
 *
 *  In cases where the order parameter is via "full LB", this couples
 *  the scalar order parameter phi_site[] to the distributions.
 *
 *  $Id: phi_lb_coupler.c,v 1.1 2009-08-20 16:23:42 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 ****************************************************************************/

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "model.h"
#include "site_map.h"
#include "phi.h"
#include "phi_lb_coupler.h"

extern double * phi_site;

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

  if (phi_is_finite_difference()) return;

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = le_site_index(ic, jc, kc);
	phi_site[nop_*index] = get_phi_at_site(index);
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

  get_N_local(nlocal);

  /* Compute the mean phi in the domain proper */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = get_site_index(ic, jc, kc);
	phi_local += get_phi_at_site(index);
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
	  index = get_site_index(ic, jc, kc);
	  phi_local = get_g_at_site(index, 0) + phi_correction;
	  set_g_at_site(index, 0,  phi_local);
	}
      }
    }
  }

  return;
}
