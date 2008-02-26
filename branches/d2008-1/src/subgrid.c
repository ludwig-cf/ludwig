/*****************************************************************************
 *
 *  subgrid.c
 *
 *  Routines for point-like particles.
 *
 *  See Nash et al. (2007).
 *
 *  $Id: subgrid.c,v 1.3.4.1 2008-02-26 09:41:08 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>
#include <stddef.h>

#include "pe.h"
#include "coords.h"
#include "timer.h"
#include "physics.h"
#include "lattice.h"
#include "colloids.h"
#include "ccomms.h"
#include "subgrid.h"

static double d_peskin(double);
static void   subgrid_interpolation(void);
static double drange_ = 1.0; /* Max. range of interpolation - 1 */

/*****************************************************************************
 *
 *  subgrid_force_from_particles()
 *
 *  For each particle, accumulate the force on the relevant surrounding
 *  lattice nodes. Only nodes in the local domain are involved.
 *
 *****************************************************************************/

void subgrid_force_from_particles() {

  int ic, jc, kc;
  int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max;
  int index;
  int N[3], offset[3];

  double r[3], r0[3], force[3], g[3];
  double dr;
  Colloid * p_colloid;

  TIMER_start(TIMER_FREE1);

  get_N_local(N);
  get_N_offset(offset);

  get_gravity(g);

  /* Loop through all cells (including the halo cells) */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

        p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {

          /* Need to translate the colloid position to "local"
           * coordinates, so that the correct range of lattice
           * nodes is found */

          r0[X] = p_colloid->r.x - (double) offset[X];
          r0[Y] = p_colloid->r.y - (double) offset[Y];
          r0[Z] = p_colloid->r.z - (double) offset[Z];

	  /* Work out which local lattice sites are involved
	   * and loop around */

          i_min = imax(1,    (int) floor(r0[X] - drange_));
          i_max = imin(N[X], (int) ceil (r0[X] + drange_));
          j_min = imax(1,    (int) floor(r0[Y] - drange_));
          j_max = imin(N[Y], (int) ceil (r0[Y] + drange_));
          k_min = imax(1,    (int) floor(r0[Z] - drange_));
          k_max = imin(N[Z], (int) ceil (r0[Z] + drange_));

          for (i = i_min; i <= i_max; i++) {
            for (j = j_min; j <= j_max; j++) {
	      for (k = k_min; k <= k_max; k++) {

		index = get_site_index(i, j, k);

                /* Separation between r0 and the coordinate position of
		 * this site */

		r[X] = r0[X] - (double) i;
		r[Y] = r0[Y] - (double) j;
		r[Z] = r0[Z] - (double) k;

		dr = d_peskin(r[X])*d_peskin(r[Y])*d_peskin(r[Z]);

		force[X] = g[X]*dr;
		force[Y] = g[Y]*dr;
		force[Z] = g[Z]*dr;
		add_force_at_site(index, force);
	      }
	    }
	  }

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }
    }
  }

  TIMER_stop(TIMER_FREE1);

  return;
}

/*****************************************************************************
 *
 *  subgrid_update
 *
 *  This function is responsible for update of position for
 *  sub-gridscale particles. It takes the place of BBL for
 *  fully resolved particles.
 *
 *****************************************************************************/

void subgrid_update() {

  int ic, jc, kc;
  double drag, eta;
  double g[3];
  Colloid * p_colloid;

  TIMER_start(TIMER_FREE1);

  subgrid_interpolation();
  CCOM_halo_sum(CHALO_TYPE7);

  /* Loop through all cells (including the halo cells) */

  eta = get_eta_shear();
  get_gravity(g);

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

        p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {

	  drag = (1.0/(6.0*PI*eta))*(1.0/p_colloid->a0 - 1.0/p_colloid->ah);

	  p_colloid->r.x += (p_colloid->f0.x + drag*g[X]);
	  p_colloid->r.y += (p_colloid->f0.y + drag*g[Y]);
	  p_colloid->r.z += (p_colloid->f0.z + drag*g[Z]);

	  /* Store the effective velocity of the particle
	   * (don't use the p->v as this shows up in the momentum) */

	  p_colloid->stats.x = p_colloid->f0.x + drag*g[X];
	  p_colloid->stats.y = p_colloid->f0.y + drag*g[Y];
	  p_colloid->stats.z = p_colloid->f0.z + drag*g[Z];

	  p_colloid = p_colloid->next;
	}

      }
    }
  }

  TIMER_stop(TIMER_FREE1);

  return;
}

/*****************************************************************************
 *
 *  subgrid_interpolation
 *
 *  Interpolate (delta function method) the lattice velocity field
 *  to the position of the particles.
 *
 *****************************************************************************/

static void subgrid_interpolation() {

  int ic, jc, kc;
  int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max;
  int index;
  int N[3], offset[3];

  double r0[3], r[3], u[3];
  double dr;
  Colloid * p_colloid;

  get_N_local(N);
  get_N_offset(offset);

  /* Loop through all cells (including the halo cells) and set
   * the velocity at each particle to zero for this step. */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

        p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {
	  p_colloid->f0.x = 0.0;
	  p_colloid->f0.y = 0.0;
	  p_colloid->f0.z = 0.0;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  /* And add up the contributions to the velocity from the lattice. */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

        p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {

          /* Need to translate the colloid position to "local"
           * coordinates, so that the correct range of lattice
           * nodes is found */

          r0[X] = p_colloid->r.x - (double) offset[X];
          r0[Y] = p_colloid->r.y - (double) offset[Y];
          r0[Z] = p_colloid->r.z - (double) offset[Z];

	  /* Work out which local lattice sites are involved
	   * and loop around */

          i_min = imax(1,    (int) floor(r0[X] - drange_));
          i_max = imin(N[X], (int) ceil (r0[X] + drange_));
          j_min = imax(1,    (int) floor(r0[Y] - drange_));
          j_max = imin(N[Y], (int) ceil (r0[Y] + drange_));
          k_min = imax(1,    (int) floor(r0[Z] - drange_));
          k_max = imin(N[Z], (int) ceil (r0[Z] + drange_));

          for (i = i_min; i <= i_max; i++) {
            for (j = j_min; j <= j_max; j++) {
	      for (k = k_min; k <= k_max; k++) {

		index = get_site_index(i, j, k);

                /* Separation between r0 and the coordinate position of
		 * this site */

		r[X] = r0[X] - (double) i;
		r[Y] = r0[Y] - (double) j;
		r[Z] = r0[Z] - (double) k;

		dr = d_peskin(r[X])*d_peskin(r[Y])*d_peskin(r[Z]);
		get_velocity_at_lattice(index, u);

		p_colloid->f0.x += u[X]*dr;
		p_colloid->f0.y += u[Y]*dr;
		p_colloid->f0.z += u[Z]*dr;
	      }
	    }
	  }

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  d_peskin
 *
 *  Approximation to \delta(r) according to Peskin.
 *
 *****************************************************************************/

static double d_peskin(double r) {

  double rmod;
  double delta = 0.0;

  rmod = fabs(r);

  if (rmod <= 1.0) {
    delta = 0.125*(3.0 - 2.0*rmod + sqrt(1.0 + 4.0*rmod - 4.0*rmod*rmod));
  }
  else if (rmod <= 2.0) {
    delta = 0.125*(5.0 - 2.0*rmod - sqrt(-7.0 + 12.0*rmod  - 4.0*rmod*rmod));
  }

  return delta;
}
