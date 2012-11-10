/****************************************************************************
 *
 *  stats_velocity.c
 *
 *  Basic statistics for the velocity field.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "site_map.h"
#include "lattice.h"
#include "util.h"
#include "stats_velocity.h"

/****************************************************************************
 *
 *  stats_velocity_minmax
 *
 *  The volume flux of is of interest for porous media calculations
 *  of permeability. Note that with the body force density f, the
 *  volume flux is the same as the total momemtum  plus 0.5*f per
 *  lattice site. So for complex porous media, the total momentum
 *  can actually look quite wrong (e.g., have the opposite sign to
 *  the flow).
 *
 ****************************************************************************/

void stats_velocity_minmax(void) {

  int ic, jc, kc, ia, index;
  int nlocal[3];
  double umin[3];
  double umax[3];
  double utmp[3];
  double usum_local[3], usum[3];

  MPI_Comm comm;

  coords_nlocal(nlocal);
  comm = pe_comm();

  for (ia = 0; ia < 3; ia++) {
    umin[ia] = FLT_MAX;
    umax[ia] = FLT_MIN;
    usum_local[ia] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

	if (site_map_get_status_index(index) == FLUID) {
	  hydrodynamics_get_velocity(index, utmp);

	  for (ia = 0; ia < 3; ia++) {
	    umin[ia] = dmin(umin[ia], utmp[ia]);
	    umax[ia] = dmax(umax[ia], utmp[ia]);
	    usum_local[ia] += utmp[ia];
	  }
	}
      }
    }
  }

  MPI_Reduce(umin, utmp, 3, MPI_DOUBLE, MPI_MIN, 0, comm);

  for (ia = 0; ia < 3; ia++) {
    umin[ia] = utmp[ia];
  }

  MPI_Reduce(umax, utmp, 3, MPI_DOUBLE, MPI_MAX, 0, comm);

  for (ia = 0; ia < 3; ia++) {
    umax[ia] = utmp[ia];
  }

  MPI_Reduce(usum_local, usum, 3, MPI_DOUBLE, MPI_SUM, 0, comm);

  info("\n");
  info("Velocity - x y z\n");
  info("[minimum ] %14.7e %14.7e %14.7e\n", umin[X], umin[Y], umin[Z]);
  info("[maximum ] %14.7e %14.7e %14.7e\n", umax[X], umax[Y], umax[Z]);
  info("[vol flux] %14.7e %14.7e %14.7e\n", usum[X], usum[Y], usum[Z]);

  return;
}
