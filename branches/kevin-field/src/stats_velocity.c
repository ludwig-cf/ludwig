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
#include "util.h"
#include "stats_velocity.h"

/****************************************************************************
 *
 *  stats_velocity_minmax
 *
 ****************************************************************************/

int stats_velocity_minmax(hydro_t * hydro) {

  int ic, jc, kc, ia, index;
  int nlocal[3];
  double umin[3];
  double umax[3];
  double utmp[3];

  MPI_Comm comm;

  coords_nlocal(nlocal);
  comm = pe_comm();

  for (ia = 0; ia < 3; ia++) {
    umin[ia] = FLT_MAX;
    umax[ia] = FLT_MIN;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

	if (site_map_get_status_index(index) == FLUID) {
	  hydro_u(hydro, index, utmp);

	  for (ia = 0; ia < 3; ia++) {
	    umin[ia] = dmin(umin[ia], utmp[ia]);
	    umax[ia] = dmax(umax[ia], utmp[ia]);
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

  info("\n");
  info("Velocity - x y z\n");
  info("[minimum ] %14.7e %14.7e %14.7e\n", umin[X], umin[Y], umin[Z]);
  info("[maximum ] %14.7e %14.7e %14.7e\n", umax[X], umax[Y], umax[Z]);

  return 0;
}
