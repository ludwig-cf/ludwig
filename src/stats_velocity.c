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
 *  (c) 2011-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "util.h"
#include "hydro_s.h"
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

int stats_velocity_minmax(hydro_t * hydro, map_t * map, int print_vol_flux) {

  int ic, jc, kc, ia, index;
  int nlocal[3];
  int status;

  double umin[3];
  double umax[3];
  double utmp[3];
  double usum_local[3], usum[3];

  MPI_Comm comm;

  assert(hydro);
  assert(map);

  cs_nlocal(hydro->cs, nlocal);
  pe_mpi_comm(hydro->pe, &comm);

  for (ia = 0; ia < 3; ia++) {
    umin[ia] = FLT_MAX;
    umax[ia] = FLT_MIN;
    usum_local[ia] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(hydro->cs, ic, jc, kc);
	map_status(map, index, &status);

	if (status == MAP_FLUID) {

	  hydro_u(hydro, index, utmp);

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

  pe_info(hydro->pe, "\n");
  pe_info(hydro->pe, "Velocity - x y z\n");
  pe_info(hydro->pe, "[minimum ] %14.7e %14.7e %14.7e\n", umin[X], umin[Y], umin[Z]);
  pe_info(hydro->pe, "[maximum ] %14.7e %14.7e %14.7e\n", umax[X], umax[Y], umax[Z]);

  if (print_vol_flux) {
    pe_info(hydro->pe, "[vol flux] %14.7e %14.7e %14.7e\n", usum[X], usum[Y], usum[Z]);
  }

  return 0;
}
