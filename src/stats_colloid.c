/*****************************************************************************
 *
 *  stats_colloid.c
 *
 *  Some useful quantities concerning colloids.
 *
 *  $Id: stats_colloid.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "colloids_s.h"
#include "util.h"
#include "stats_colloid.h"

/*****************************************************************************
 *
 *  stats_colloid_momentum
 *
 *  Return net colloid momentum as g[3].
 *
 *  The final reduction to rank 0 in pe_comm is for the
 *  purpose of output statistics via info().
 *
 *****************************************************************************/

int stats_colloid_momentum(colloids_info_t * cinfo, double g[3]) {

  int ic, jc, kc;
  int ntotal;
  int ncell[3];

  double glocal[3] = {0.0, 0.0, 0.0};
  double rho0;
  double mass;
  PI_DOUBLE(pi);

  colloid_t * pc = NULL;
  MPI_Comm comm;

  assert(cinfo);

  colloids_info_ntotal(cinfo, &ntotal);
  if (ntotal == 0) return 0;

  colloids_info_ncell(cinfo, ncell);
  colloids_info_rho0(cinfo, &rho0);
  pe_mpi_comm(cinfo->pe, &comm);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {
	  mass = 4.0*pi*pow(pc->s.a0, 3)*rho0/3.0;
	  if (pc->s.type == COLLOID_TYPE_SUBGRID) mass = 0.0; /* No inertia */

	  glocal[X] += mass*pc->s.v[X];
	  glocal[Y] += mass*pc->s.v[Y];
	  glocal[Z] += mass*pc->s.v[Z];

	  /* Next colloid */
	  pc = pc->next;
	}

	/* Next cell */
      }
    }
  }

  MPI_Reduce(glocal, g, 3, MPI_DOUBLE, MPI_SUM, 0, comm);

  return 0;
}

/****************************************************************************
 *
 *  stats_colloid_velocity_minmax
 *
 *  Report stats on particle speeds. Accumulate min(-v) for maximum.
 *
 ****************************************************************************/ 

int stats_colloid_velocity_minmax(colloids_info_t * cinfo) {

  int ia;
  int ic, jc, kc;
  int ncell[3];
  double vmin[6];
  double vminlocal[6];
  colloid_t * pc;
  MPI_Comm comm;

  for (ia = 0; ia < 6; ia++) {
    vminlocal[ia] = FLT_MAX;
  }

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);
  pe_mpi_comm(cinfo->pe, &comm);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {
	  for (ia = 0; ia < 3; ia++) {
	    vminlocal[ia] = dmin(vminlocal[ia], pc->s.v[ia]);
	    vminlocal[3+ia] = dmin(vminlocal[3+ia], -pc->s.v[ia]);
	  }
	  pc = pc->next;
	}
      }
    }
  }

  MPI_Reduce(vminlocal, vmin, 6, MPI_DOUBLE, MPI_MIN, 0, comm);

  pe_info(cinfo->pe, "Colloid velocities - x y z\n");
  pe_info(cinfo->pe, "[minimum ] %14.7e %14.7e %14.7e\n",
	             vmin[X], vmin[Y], vmin[Z]);
  pe_info(cinfo->pe, "[maximum ] %14.7e %14.7e %14.7e\n",
	             -vmin[3+X],-vmin[3+Y],-vmin[3+Z]);

  return 0;
}
