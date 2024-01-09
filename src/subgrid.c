/*****************************************************************************
 *
 *  subgrid.c
 *
 *  Routines for point-like particles.
 *
 *  See Nash et al. (2007).
 *
 *  Overview.
 *
 *  Two-way coupling between sub-grid particles and the fluid is implemented
 *  in roughly two phases.
 *
 *  (1) Force on each particle from the immediately surrounding fluid.
 *  (2) Influence of the particles on local fluid nodes;
 *
 *  (1) subgrid_update() is responsible for the particle update and setting
 *      any position increment dr. Schematically:
 *
 *      -> subgrid_interpolation()
 *         accumulates contributions to the force on the particle from
 *         local fluid nodes to local particle copies (fsub[3]);
 *      -> COLLOID_SUM_SUBGRID
 *         ensures all copies agree on the net force per particle fsub[3].
 *      -> all copies update v and dr = v.dt from fsub[3] and must agree.
 *      -> Actual position updates must be deferred until the start of
 *         the next time step and solloids_info_position_update().
 *
 *  (2) subgrid_force_from_particle() is responsible for computing
 *      the force on local fluid nodes from particles. Schematically;
 *
 *      -> On entry, fex[3] may contain pair interaction and other
 *         "external" forces on the particle;
 *      -> subgrid_wall_lubrication()
 *          detect and compute particle/wall lubrication forces,
 *          and accumulate to fex[3] once per particle (i.e. local copies
 *          only involved);
 *      -> COLLOID_SUM_FORCE_EXT_ONLY
 *         => all copies agree on fext[3], text[3]
 *
 *      -> All particle copies contribute \delta(r - R_i) fext[3] to local
 *         fluid nodes only via hydro_f_local_add() at position r.
 *      -> This force may then enter the fluid collision stage.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "colloids.h"
#include "colloid_sums.h"
#include "util.h"
#include "subgrid.h"

static double d_peskin(double);
static int subgrid_interpolation(colloids_info_t * cinfo, hydro_t * hydro);
static const double drange_ = 1.0; /* Max. range of interpolation - 1 */

/*****************************************************************************
 *
 *  subgrid_force_from_particles()
 *
 *  For each particle, accumulate the force on the relevant surrounding
 *  lattice nodes. Only nodes in the local domain are involved.
 *
 *  If there are no subgrid particles, hydro is allowed to be NULL.
 *
 *****************************************************************************/


int subgrid_force_from_particles(colloids_info_t * cinfo, hydro_t * hydro,
				 wall_t * wall) {

  int ic, jc, kc;
  int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max;
  int index;
  int nlocal[3], offset[3];
  int ncell[3];

  double r[3], r0[3], force[3];
  double dr;
  colloid_t * p_colloid = NULL;  /* Subgrid colloid */
  colloid_t * presolved = NULL;  /* Resolved colloid occupuing node */

  assert(cinfo);
  assert(wall);

  if (cinfo->nsubgrid == 0) return 0;

  cs_nlocal(cinfo->cs, nlocal);
  cs_nlocal_offset(cinfo->cs, offset);
  colloids_info_ncell(cinfo, ncell);

  /* Add any wall lubrication corrections before communication to
   * find total external force on each particle */

  subgrid_wall_lubrication(cinfo, wall);
  colloid_sums_halo(cinfo, COLLOID_SUM_FORCE_EXT_ONLY);

  /* While there is no device implementation, must copy back-and forth
   * the force. */

  assert(hydro);
  hydro_memcpy(hydro, tdpMemcpyDeviceToHost);

  /* Loop through all cells (including the halo cells) */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	for ( ; p_colloid; p_colloid = p_colloid->next) {

	  if (p_colloid->s.bc != COLLOID_BC_SUBGRID) continue;

	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */

	  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
	  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
	  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

	  /* Work out which local lattice sites are involved
	   * and loop around */

	  i_min = imax(1,         (int) floor(r0[X] - drange_));
	  i_max = imin(nlocal[X], (int) ceil (r0[X] + drange_));
	  j_min = imax(1,         (int) floor(r0[Y] - drange_));
	  j_max = imin(nlocal[Y], (int) ceil (r0[Y] + drange_));
	  k_min = imax(1,         (int) floor(r0[Z] - drange_));
	  k_max = imin(nlocal[Z], (int) ceil (r0[Z] + drange_));

	  for (i = i_min; i <= i_max; i++) {
	    for (j = j_min; j <= j_max; j++) {
	      for (k = k_min; k <= k_max; k++) {

		index = cs_index(cinfo->cs, i, j, k);

		/* Separation between r0 and the coordinate position of
		 * this site */

		r[X] = r0[X] - 1.0*i;
		r[Y] = r0[Y] - 1.0*j;
		r[Z] = r0[Z] - 1.0*k;

		dr = d_peskin(r[X])*d_peskin(r[Y])*d_peskin(r[Z]);

		force[X] = p_colloid->fex[X]*dr;
		force[Y] = p_colloid->fex[Y]*dr;
		force[Z] = p_colloid->fex[Z]*dr;

		colloids_info_map(cinfo, index, &presolved);

		if (presolved == NULL) {
		  hydro_f_local_add(hydro, index, force);
		}
		else {
		  double rd[3] = {0};
		  double torque[3] = {0};
		  presolved->force[X] += force[X];
		  presolved->force[Y] += force[Y];
		  presolved->force[Z] += force[Z];
		  rd[X] = 1.0*i - (presolved->s.r[X] - 1.0*offset[X]);
		  rd[Y] = 1.0*j - (presolved->s.r[Y] - 1.0*offset[Y]);
		  rd[Z] = 1.0*k - (presolved->s.r[Z] - 1.0*offset[Z]);
		  cross_product(rd, force, torque);
		  presolved->torque[X] += torque[X];
		  presolved->torque[Y] += torque[Y];
		  presolved->torque[Z] += torque[Z];
                }

	      }
	    }
	  }

	  /* Next colloid */
	}

	/* Next cell */
      }
    }
  }

  hydro_memcpy(hydro, tdpMemcpyHostToDevice);

  return 0;
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

int subgrid_update(colloids_info_t * cinfo, hydro_t * hydro, int noise_flag) {

  int ia;
  int ic, jc, kc;
  int ncell[3];
  double drag, reta;
  double eta;
  PI_DOUBLE(pi);
  colloid_t * p_colloid;
  physics_t * phys = NULL;

  double ran[2];    /* Random numbers for fluctuation dissipation correction */
  double frand[3];  /* Random force */
  double kt;        /* Temperature */

  assert(cinfo);
  assert(hydro);

  if (cinfo->nsubgrid == 0) return 0;

  colloids_info_ncell(cinfo, ncell);

  subgrid_interpolation(cinfo, hydro);
  colloid_sums_halo(cinfo, COLLOID_SUM_SUBGRID);

  /* Loop through all cells (including the halo cells) */

  physics_ref(&phys);
  physics_eta_shear(phys, &eta);
  physics_kt(phys, &kt);
  reta = 1.0/(6.0*pi*eta);

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	for ( ; p_colloid; p_colloid = p_colloid->next) {

	  if (p_colloid->s.bc != COLLOID_BC_SUBGRID) continue;

	  drag = reta*(1.0/p_colloid->s.ah - 1.0/p_colloid->s.al);

	  if (noise_flag == 0) {
	    frand[X] = 0.0; frand[Y] = 0.0; frand[Z] = 0.0;
	  }
	  else {
	    for (ia = 0; ia < 3; ia++) {
	      while (1) {
		/* To keep the random correction smaller than 3 sigma.
		 * Otherwise, a large thermal fluctuation may cause a
		 * numerical problem. */
		util_ranlcg_reap_gaussian(&p_colloid->s.rng, ran);
		if (fabs(ran[0]) < 3.0) {
		  frand[ia] = sqrt(2.0*kt*drag)*ran[0];
		  break;
		}
		if (fabs(ran[1]) < 3.0) {
		  frand[ia] = sqrt(2.0*kt*drag)*ran[1];
		  break;
		}
	      }
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    p_colloid->s.v[ia] = p_colloid->fsub[ia] + drag*p_colloid->fex[ia]
                               + frand[ia];
	    p_colloid->s.dr[ia] = p_colloid->s.v[ia];
	  }
	}
	/* Next cell */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  subgrid_interpolation
 *
 *  Interpolate (delta function method) the lattice velocity field
 *  to the position of the particles.
 *
 *****************************************************************************/

static int subgrid_interpolation(colloids_info_t * cinfo, hydro_t * hydro) {

  int ic, jc, kc;
  int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max;
  int index;
  int nlocal[3], offset[3];
  int ncell[3];

  double r0[3], r[3], u[3];
  double dr;
  colloid_t * p_colloid;

  assert(cinfo);
  assert(hydro);

  cs_nlocal(cinfo->cs, nlocal);
  cs_nlocal_offset(cinfo->cs, offset);
  colloids_info_ncell(cinfo, ncell);

  /* While there is no subgrid device implementation,
     need to recover the current velocity. */

  hydro_memcpy(hydro, tdpMemcpyDeviceToHost);

  /* Loop through all cells (including the halo cells) and set
   * the velocity at each particle to zero for this step. */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	for ( ; p_colloid; p_colloid = p_colloid->next) {

	  if (p_colloid->s.bc != COLLOID_BC_SUBGRID) continue;

	  p_colloid->fsub[X] = 0.0;
	  p_colloid->fsub[Y] = 0.0;
	  p_colloid->fsub[Z] = 0.0;
	}
      }
    }
  }

  /* And add up the contributions to the velocity from the lattice. */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	for ( ; p_colloid; p_colloid = p_colloid->next) {

	  if (p_colloid->s.bc != COLLOID_BC_SUBGRID) continue;

	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */

	  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
	  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
	  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

	  /* Work out which local lattice sites are involved
	   * and loop around */

	  i_min = imax(1,         (int) floor(r0[X] - drange_));
	  i_max = imin(nlocal[X], (int) ceil (r0[X] + drange_));
	  j_min = imax(1,         (int) floor(r0[Y] - drange_));
	  j_max = imin(nlocal[Y], (int) ceil (r0[Y] + drange_));
	  k_min = imax(1,         (int) floor(r0[Z] - drange_));
	  k_max = imin(nlocal[Z], (int) ceil (r0[Z] + drange_));

	  for (i = i_min; i <= i_max; i++) {
	    for (j = j_min; j <= j_max; j++) {
	      for (k = k_min; k <= k_max; k++) {

		index = cs_index(cinfo->cs, i, j, k);

		/* Separation between r0 and the coordinate position of
		 * this site */

		r[X] = r0[X] - 1.0*i;
		r[Y] = r0[Y] - 1.0*j;
		r[Z] = r0[Z] - 1.0*k;

		dr = d_peskin(r[X])*d_peskin(r[Y])*d_peskin(r[Z]);
		hydro_u(hydro, index, u);

		p_colloid->fsub[X] += u[X]*dr;
		p_colloid->fsub[Y] += u[Y]*dr;
		p_colloid->fsub[Z] += u[Z]*dr;
	      }
	    }
	  }

	  /* Next colloid */
	}

	/* Next cell */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  subgrid_wall_lubrication
 *
 *  Accumulate lubrication corrections to the external force on each particle.
 *
 *****************************************************************************/

int subgrid_wall_lubrication(colloids_info_t * cinfo, wall_t * wall) {

  double drag[3];
  colloid_t * pc = NULL;

  double f[3] = {0.0, 0.0, 0.0};

  assert(cinfo);
  assert(wall);

  colloids_info_local_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextlocal) {
    if (pc->s.bc != COLLOID_BC_SUBGRID) continue;
    wall_lubr_sphere(wall, pc->s.ah, pc->s.r, drag);
    pc->fex[X] += drag[X]*pc->s.v[X];
    pc->fex[Y] += drag[Y]*pc->s.v[Y];
    pc->fex[Z] += drag[Z]*pc->s.v[Z];
    f[X] -= drag[X]*pc->s.v[X];
    f[Y] -= drag[Y]*pc->s.v[Y];
    f[Z] -= drag[Z]*pc->s.v[Z];
  }

  wall_momentum_add(wall, f);

  return 0;
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
