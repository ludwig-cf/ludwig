/*****************************************************************************
 *
 *  stats_colloid.c
 *
 *  Some useful quantities concerning colloids.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
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
#include "colloids.h"
#include "util.h"
#include "util_ellipsoid.h"
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

  int ntotal = 0;
  double glocal[3] = {0.0, 0.0, 0.0};
  double rho0;

  colloid_t * pc = NULL;
  MPI_Comm comm = MPI_COMM_NULL;

  assert(cinfo);

  colloids_info_ntotal(cinfo, &ntotal);
  if (ntotal == 0) return 0;

  colloids_info_rho0(cinfo, &rho0);
  pe_mpi_comm(cinfo->pe, &comm);

  colloids_info_local_head(cinfo, &pc);

  for (; pc; pc = pc->nextlocal) {

    double mass = 0.0;

    colloid_state_mass(&pc->s, rho0, &mass);
    if (pc->s.bc == COLLOID_BC_SUBGRID) mass = 0.0; /* No inertia */

    glocal[X] += mass*pc->s.v[X];
    glocal[Y] += mass*pc->s.v[Y];
    glocal[Z] += mass*pc->s.v[Z];
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

/*****************************************************************************
 *
 *  stats_colloid_write_velocities
 *
 *****************************************************************************/

int stats_colloid_write_velocities(pe_t * pe, colloids_info_t * info) {

  colloid_t * pc = NULL;

  colloids_info_all_head(info, &pc);

  pe_info(pe, "Colloid velocities\n");

  for ( ; pc; pc = pc->nextlocal) {
    printf("%22.15e %22.15e %22.15e %22.15e %22.15e %22.15e\n",
       pc->s.v[X], pc->s.v[Y], pc->s.v[Z], pc->s.w[X], pc->s.w[Y], pc->s.w[Z]);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_colloid_write_orientations
 *
 *****************************************************************************/

int stats_colloid_write_info(pe_t * pe, colloids_info_t * info, double const t) {

  colloid_t * pc = NULL;
  double phi,theta,psi;
  double r;
  double force = -0.01;
  double mu = 0.1;
  double U[2];

  colloids_info_all_head(info, &pc);

  PI_DOUBLE(pi);

  pe_info(pe, "Colloid positions, orientations and velocities\n");

  for ( ; pc; pc = pc->nextlocal) {

    printf("%22.15e %22.15e %22.15e position \n", pc->s.r[X], pc->s.r[Y], pc->s.r[Z]);
    util_q4_to_euler_angles(pc->s.quater, &phi, &theta, &psi);
    printf("%22.15e, %22.15e, %22.15e\n",phi*180.0/pi,theta*180.0/pi,psi*180.0/pi);
    r = pc->s.elabc[0]/pc->s.elabc[1];
    //Jeffery_omega_predicted(r,pc->s.quater, gammadot,opred,angpred);
    settling_velocity_prolate(r, force, mu, pc->s.elabc[0], U);
    //printf("%22.15e,\t%22.15e, Predicted U\n",U[0],U[1]);
    //elc=(sqrt(pc->s.elabc[0]*pc->s.elabc[0]-pc->s.elabc[1]*pc->s.elabc[1]))/pc->s.elabc[0];
    //tau0=1.0/elc;
    //sqU=tau0*(tau0-(tau0*tau0-1.0)*atanh(elc));
    //printf("%22.15e, Predicted U of squirmer\n",pc->s.b1*sqU);
    //ellipsoid_nearwall_predicted(pc->s.elabc,pc->s.r[1],pc->s.quater, Upred,opred);
    //printf("%22.15e,\t%22.15e,\t%22.15e Predicted U\n",Upred[0],Upred[1],Upred[2]);
    printf("%22.15e %22.15e %22.15e U \n", pc->s.v[X], pc->s.v[Y], pc->s.v[Z]);
    //printf("%22.15e,\t%22.15e,\t%22.15e Predicted O\n",opred[0],opred[1],opred[2]);
    printf("%22.15e,\t%22.15e,\t%22.15e Calculated O\n",pc->s.w[X], pc->s.w[Y], pc->s.w[Z]);
  }

  return 0;
}
