/****************************************************************************
 *
 *  stats_free_energy.c
 *
 *  Statistics for free energy density.
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

#include "pe.h"
#include "coords.h"
#include "control.h"
#include "colloids.h"
#include "colloids_Q_tensor.h"
#include "phi.h"
#include "site_map.h"
#include "wall.h"
#include "free_energy.h"
#include "stats_free_energy.h"

static void stats_free_energy_wall(double * fs);
static void stats_free_energy_colloid(double * fs);

/****************************************************************************
 *
 *  stats_free_energy_density
 *
 *  Tots up the free energy density. The loop here totals the fluid,
 *  and there is an additional calculation for different types of
 *  solid surface.
 *
 ****************************************************************************/

void stats_free_energy_density(void) {

  int ic, jc, kc, index;
  int nlocal[3];

  double fed;
  double fe_local[5];
  double fe_total[5];
  double rv;

  double (* free_energy_density)(const int index);

  coords_nlocal(nlocal);
  free_energy_density = fe_density_function();

  fe_local[0] = 0.0; /* Total free energy (fluid all sites) */
  fe_local[1] = 0.0; /* Fluid only free energy */
  fe_local[2] = 0.0; /* Volume of fluid */
  fe_local[3] = 0.0; /* surface free energy */
  fe_local[4] = 0.0; /* other wall free energy (walls only) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	fed = free_energy_density(index);
	fe_local[0] += fed;
	if (site_map_get_status_index(index) == FLUID) {
	    fe_local[1] += fed;
	    fe_local[2] += 1.0;
	}
      }
    }
  }

  /* A robust mechanism is required to get the surface free energy */

  if (wall_present()) {

    stats_free_energy_wall(fe_local + 3);

    MPI_Reduce(fe_local, fe_total, 5, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

    info("\nFree energies - timestep f v f/v f_s1 fs_s2 \n");
    info("[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	 get_step(), fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	 fe_total[3], fe_total[4]);
  }
  else if (colloid_ntotal() > 0) {

    stats_free_energy_colloid(fe_local + 3);

    MPI_Reduce(fe_local, fe_total, 5, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

    info("\nFree energies - timestep f v f/v f_s a f_s/a\n");

    if (fe_total[4] > 0.0) {
      /* Area > 0 means the free energy is available */
      info("[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	   get_step(), fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	   fe_total[3], fe_total[4], fe_total[3]/fe_total[4]);
    }
    else {
      info("[fe] %14d %17.10e %17.10e %17.10e %17.10e\n",
	   get_step(), fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	   fe_total[3]);
    }
  }
  else {
    MPI_Reduce(fe_local, fe_total, 3, MPI_DOUBLE, MPI_SUM, 0, pe_comm());
    rv = 1.0/(L(X)*L(Y)*L(Z));

    info("\nFree energy density - timestep total fluid\n");
    info("[fed] %14d %17.10e %17.10e\n", get_step(), rv*fe_total[0],
	 fe_total[1]/fe_total[2]);
  }

  return;
}

/*****************************************************************************
 *
 *  stats_free_energy_wall
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

static void stats_free_energy_wall(double * fs) {

  int ic, jc, kc, index;
  int ia, ib;
  int nlocal[3];

  double w;
  double dn[3];
  double qs[3][3], q0[3][3];

  fs[0] = 0.0;
  fs[1] = 0.0;

  if (colloids_q_anchoring_method() != ANCHORING_METHOD_TWO) return;

  if (wall_at_edge(Y)) fatal("No y wall free energy yet\n");
  if (wall_at_edge(Z)) fatal("No z wall free energy yet\n");

  coords_nlocal(nlocal);
  w = colloids_q_tensor_w();

  assert(phi_nop() == 5);

  dn[Y] = 0.0;
  dn[Z] = 0.0;

  if (cart_coords(X) == 0) {

    ic = 1;
    dn[X] = +1.0;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	phi_get_q_tensor(index, qs);
	colloids_q_boundary(dn, qs, q0);
	
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    fs[0] += 0.5*w*(qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
	  }
	}

      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {

    ic = nlocal[X];
    dn[X] = -1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	phi_get_q_tensor(index, qs);
	colloids_q_boundary(dn, qs, q0);
	
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    fs[1] += 0.5*w*(qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
	  }
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  stats_free_energy_colloid
 *
 *  Return f_s for the local colloid surface (and an area).
 *
 *     fs[0] = free energy (integrated over area)
 *     fs[1] = discrete surface area
 *
 *****************************************************************************/

static void stats_free_energy_colloid(double * fs) {

  int ic, jc, kc, index;
  int ia, ib;
  int nhat[3];
  int nlocal[3];

  double dn[3];
  double q0[3][3], qs[3][3];
  double w;

  coords_nlocal(nlocal);
  w = colloids_q_tensor_w();

  fs[0] = 0.0;
  fs[1] = 0.0;

  if (colloids_q_anchoring_method() != ANCHORING_METHOD_TWO) return;

  assert(phi_nop() == 5);
  assert(w >= 0.0);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
        if (site_map_get_status_index(index) != FLUID) continue;

	phi_get_q_tensor(index, qs);

        nhat[Y] = 0;
        nhat[Z] = 0;

        if (site_map_get_status(ic+1, jc, kc) == COLLOID) {
          nhat[X] = -1;
          colloids_q_boundary_normal(index, nhat, dn);
	  colloids_q_boundary(dn, qs, q0);
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      fs[0] += 0.5*w*
		(qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
	    }
	  }
	  fs[1] += 1.0;
        }

        if (site_map_get_status(ic-1, jc, kc) == COLLOID) {
          nhat[X] = +1;
          colloids_q_boundary_normal(index, nhat, dn);
	  colloids_q_boundary(dn, qs, q0);
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      fs[0] += 0.5*w*
		(qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
	    }
	  }
	  fs[1] += 1.0;
        }

	nhat[X] = 0;
	nhat[Z] = 0;

        if (site_map_get_status(ic, jc+1, kc) == COLLOID) {
          nhat[Y] = -1;
          colloids_q_boundary_normal(index, nhat, dn);
	  colloids_q_boundary(dn, qs, q0);
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      fs[0] += 0.5*w*
		(qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
	    }
	  }
	  fs[1] += 1.0;
        }

        if (site_map_get_status(ic, jc-1, kc) == COLLOID) {
          nhat[Y] = +1;
          colloids_q_boundary_normal(index, nhat, dn);
	  colloids_q_boundary(dn, qs, q0);
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      fs[0] += 0.5*w*
		(qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
	    }
	  }
	  fs[1] += 1.0;
        }

	nhat[X] = 0;
	nhat[Y] = 0;

        if (site_map_get_status(ic, jc, kc+1) == COLLOID) {
          nhat[Z] = -1;
          colloids_q_boundary_normal(index, nhat, dn);
	  colloids_q_boundary(dn, qs, q0);
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      fs[0] += 0.5*w*
		(qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
	    }
	  }
	  fs[1] += 1.0;
        }

        if (site_map_get_status(ic, jc, kc-1) == COLLOID) {
          nhat[Z] = +1;
          colloids_q_boundary_normal(index, nhat, dn);
	  colloids_q_boundary(dn, qs, q0);
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      fs[0] += 0.5*w*
		(qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
	    }
	  }
	  fs[1] += 1.0;
        }
	
      }
    }
  }

  return;
}
