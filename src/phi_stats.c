/*****************************************************************************
 *
 *  phi_stats.c
 *
 *  Order parameter statistics.
 *
 *  $Id: phi_stats.c,v 1.2 2008-08-24 16:58:10 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "phi.h"
#include "phi_stats.h"

/* Set phi function should move to 'coupler' */
#include "site_map.h"
#include "model.h"

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

/*****************************************************************************
 *
 *  phi_stats_print_stats
 *
 *  Return: the total, the mean, the variance, the maximum, the minimum
 *  of the order parameter.
 *
 *****************************************************************************/

void phi_stats_print_stats() {

  int      index, ic, jc, kc;
  int      nlocal[3];
  double   phi0, phi1, phi_local[5], phi_total[5];
  MPI_Comm comm = cart_comm();

  get_N_local(nlocal);

  phi_local[0] = 0.0;
  phi_local[1] = 0.0;
  phi_local[2] = 0.0;
  phi_local[3] = -DBL_MAX;
  phi_local[4] = +DBL_MAX;

  /* Compute the mean phi in the domain proper */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);
	phi0 = phi_get_phi_site(index);

	phi_local[0] += 1.0;
	phi_local[1] += phi0;
	phi_local[2] += phi0*phi0;
	phi_local[3] = dmax(phi0, phi_local[3]);
	phi_local[4] = dmin(phi0, phi_local[4]);
      }
    }
  }

  MPI_Reduce(phi_local,   phi_total,   3, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(phi_local+3, phi_total+3, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(phi_local+4, phi_total+4, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

  /* Mean and variance */

  phi0 = phi_total[1]/phi_total[0];
  phi1 = phi_total[2]/phi_total[1] - phi0*phi0;

  info("[phi][%.8g, %.8g, %.8g, %.8g, %.8g]\n", phi_total[1], phi0, phi1,
       phi_total[3], phi_total[4]);

  return;
}

/*****************************************************************************
 *
 *  phi_init_block
 *
 *  Initialise two blocks with interfaces at z = Lz/4 and z = 3Lz/4.
 *
 *****************************************************************************/

void phi_init_block() {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  int k1, k2;

  get_N_local(nlocal);
  get_N_offset(noffset);

  k1 = 0.25*L(Z);
  k2 = 0.75*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	if (noffset[Z] + kc > k1 && noffset[Z] + kc <= k2) {
	  phi_set_phi_site(index, -1.0);
	}
	else {
	  phi_set_phi_site(index, +1.0);
	}
      }
    }
  }

  return;
}
