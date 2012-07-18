/*****************************************************************************
 *
 *  interaction.c
 *
 *  Colloid potentials and colloid-colloid interactions.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "ran.h"
#include "runtime.h"

#include "bbl.h"
#include "build.h"
#include "physics.h"
#include "magnetic_field.h"
#include "magnetic_field_rt.h"
#include "potential.h"

#include "colloids.h"
#include "interaction.h"
#include "model.h"
#include "site_map.h"
#include "cio.h"
#include "control.h"
#include "subgrid.h"
#include "stats_colloid.h"

#include "util.h"
#include "colloid_sums.h"
#include "colloids_halo.h"
#include "colloids_init.h"
#include "ewald.h"

#ifdef _GPU_
#include "interface_gpu.h"
#endif

static void    COLL_overlap(colloid_t *, colloid_t *);
static void    COLL_set_fluid_gravity(void);

void lubrication_sphere_sphere(double a1, double a2,
			       const double u1[3], const double u2[3],
			       const double r12[3], double f[3]);
static void    coll_position_update(void);
static void    lubrication_init(void);
static void    colloid_forces_check(void);

static int    gravity_ = 0;            /* Switch */
static double g_[3] = {0.0, 0.0, 0.0}; /* External gravitational force */

struct lubrication_struct {
  int corrections_on;
  double cutoff_norm;  /* Normal */
  double cutoff_tang;  /* Tangential */
} lubrication;

static double epotential_;


/*****************************************************************************
 *
 *  COLL_update
 *
 *  The colloid positions have been updated. The following are required:
 *    (1) deal with processor/periodic boundaries
 *    (2) update the cell list if necessary and sort
 *    (3) update the colloid map
 *    (4) add or remove fluid to reflect changes in the colloid map
 *    (5) reconstruct the boundary links
 *
 *  The remove and replace functions expect distributions at their
 *  proper locations (with halos up-to-date).
 *
 *****************************************************************************/

void COLL_update() {

  if (colloid_ntotal() == 0) return;


  TIMER_start(TIMER_PARTICLE_HALO);

  coll_position_update();
  colloids_cell_update();
  colloids_halo_state();

  TIMER_stop(TIMER_PARTICLE_HALO);

  if (subgrid_on()) {
    COLL_forces();
    subgrid_force_from_particles();
  }
  else {

    /* Removal or replacement of fluid requires a lattice halo update */
    TIMER_start(TIMER_HALO_LATTICE);
    #ifdef _GPU_
        halo_swap_gpu();
    #else
    distribution_halo();
    #endif
    TIMER_stop(TIMER_HALO_LATTICE);

#ifdef _GPU_
    get_f_from_gpu();  
#endif



    TIMER_start(TIMER_REBUILD);
    COLL_update_map();
    COLL_remove_or_replace_fluid();
    COLL_update_links();

    TIMER_stop(TIMER_REBUILD);

    COLL_forces();
  }

#ifdef _GPU_
  put_f_on_gpu();  
#endif

  return;
}

/*****************************************************************************
 *
 *  COLL_init
 * 
 *  Driver routine for colloid initialisation.
 *
 *****************************************************************************/

void COLL_init() {

  int n;
  int init_from_file;
  int init_random;
  int ncell[3];
  char filename[FILENAME_MAX];
  char keyvalue[128];
  double dh;
  double width;
  double g[3];

  colloid_state_t * state0;

  /* Default position: no colloids */

  init_random = 0;
  init_from_file = 0;

  RUN_get_string_parameter("colloid_init", keyvalue, 128);

  if (strcmp(keyvalue, "random") == 0) init_random = 1;
  if (strcmp(keyvalue, "from_file") == 0) init_from_file = 1;

  if (init_random || init_from_file) {

    /* Set the list list width. */
    n = RUN_get_double_parameter("colloid_cell_min", &width);
    if (n == 0) {
      info("Must set a minimum cell width for colloids colloid_cell_min\n");
      fatal("Stop.\n");
    }
    ncell[X] = L(X) / (cart_size(X)*width);
    ncell[Y] = L(Y) / (cart_size(Y)*width);
    ncell[Z] = L(Z) / (cart_size(Z)*width);

    if (ncell[X] < 2 || ncell[Y] < 2 || ncell[Z] < 2) {
      info("[Error  ] Please check the cell width (cell_list_lmin).\n");
      fatal("[Stop] Must be at least two cells in each direction.\n");
    }

    colloids_cell_ncell_set(ncell);
  }

  colloids_init();
  magnetic_field_runtime();

  if (init_random || init_from_file) {

    /* Initialisation section. */

    colloid_io_init();

    if (init_from_file) {
      if (get_step() == 0) {
	sprintf(filename, "%s", "config.cds.init");
      }
      else {
	sprintf(filename, "%s%8.8d", "config.cds", get_step());
      }

      colloid_io_read(filename);
    }

    if (init_random) {
      state0 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
      assert(state0 != NULL);

      /* Minimal error testing here at the moment. */
      RUN_get_int_parameter("colloid_random_no", &n);
      RUN_get_double_parameter("colloid_random_a0", &state0->a0);
      RUN_get_double_parameter("colloid_random_ah", &state0->ah);
      RUN_get_double_parameter_vector("colloid_random_r0", state0->r);
      RUN_get_double_parameter_vector("colloid_random_v0", state0->v);
      RUN_get_double_parameter_vector("colloid_random_w0", state0->w);
      RUN_get_double_parameter_vector("colloid_random_s0", state0->s);
      RUN_get_double_parameter_vector("colloid_random_m0", state0->m);
      RUN_get_double_parameter("colloid_random_b1", &state0->b1);
      RUN_get_double_parameter("colloid_random_b2", &state0->b2);
      RUN_get_double_parameter("colloid_random_dh", &dh);

      colloids_init_random(n, state0, dh);
      info("Initialised %d colloid%s from input\n", n, (n > 1) ? "s" : "");

      free(state0);
    }

    n = RUN_get_double_parameter_vector("colloid_gravity", g);
    if (n != 0) {
      if (g[X] != 0.0 || g[Y] != 0.0 || g[Z] != 0.0) {
	colloid_gravity_set(g);
      }
    }

    /* ewald_init(0.285, 16.0);*/

    lubrication_init();
    soft_sphere_init();
    lennard_jones_init();
    yukawa_init();
    colloid_forces_check();

    COLL_init_coordinates();

    /* Transfer any particles in the halo regions, initialise the
     * colloid map and build the particles for the first time. */

    colloids_halo_state();

    /* Active */
    RUN_get_string_parameter("colloid_type", keyvalue, 128);
    if (strcmp(keyvalue, "active") == 0) bbl_active_on_set();
    if (strcmp(keyvalue, "subgrid") == 0) subgrid_on_set();

    if (subgrid_on() == 0) {
      COLL_update_map();
      COLL_update_links();
    }

    /* Information */
    if (gravity_) {
      info("Sedimentation force on:   yes\n");
      info("Sedimentation force:      %14.7e %14.7e %14.7e",
	   g_[X], g_[Y], g_[Z]);
    }
    info("\n");
  }

  return;
}

/*****************************************************************************
 *
 *  lubrication_init
 *
 *  Initialise the parameters for corrections to the lubrication
 *  forces between colloids.
 *
 *****************************************************************************/

static void lubrication_init(void) {

  int n;

  n = RUN_get_int_parameter("lubrication_on", &(lubrication.corrections_on));

  if (lubrication.corrections_on) {
    info("\nColloid-colloid lubrication corrections\n");
    info("Lubrication corrections are switched on\n");
    n = RUN_get_double_parameter("lubrication_normal_cutoff",
				 &(lubrication.cutoff_norm));
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Normal force cutoff is %f\n", lubrication.cutoff_norm);
    
    n = RUN_get_double_parameter("lubrication_tangential_cutoff",
				 &(lubrication.cutoff_tang));
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Tangential force cutoff is %f\n", lubrication.cutoff_tang);
  }

  return;
}

/*****************************************************************************
 *
 *  COLL_forces
 *
 *  Top-level function for compuatation of external forces to be called
 *  once per time step. Note that particle copies in the halo regions
 *  must have zero external force/torque on exit from this routine.
 *
 *****************************************************************************/

void COLL_forces() {

  int nc = colloid_ntotal();
  double hminlocal;
  double hmin;
  double etotal;

  if (nc > 0) {

    COLL_zero_forces();

    hminlocal = COLL_interactions();
    COLL_set_fluid_gravity();
    ewald_sum();

    if (is_statistics_step()) {

      MPI_Reduce(&hminlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, 0, pe_comm());
      MPI_Reduce(&epotential_, &etotal, 1, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

      info("\nParticle statistics:\n");
      if (nc > 1) {
	info("Inter-particle minimum h is: %10.5e\n", hmin);
	info("Potential energy is:         %10.5e\n", etotal);
      }
      stats_colloid_velocity_minmax();
    }
  }

  return;
}


/*****************************************************************************
 *
 *  COLL_zero_forces
 *
 *  Set the external forces on the particles to zero.
 *  All additional forces are accumulated.
 *
 *****************************************************************************/

void COLL_zero_forces() {

  int       ic, jc, kc, ia;
  colloid_t * p_colloid;

  /* Only real particles need correct external force. */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {

	  for (ia = 0; ia < 3; ia++) {
	    p_colloid->force[ia] = 0.0;
	    p_colloid->torque[ia] = 0.0;
	  }

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  COLL_set_fluid_gravity
 *
 *  Set the current gravtitational force on the fluid. This should
 *  match, exactly, the force on the colloids and so depends on the
 *  current number of fluid sites globally (fluid volume).
 *
 *  Note the volume calculation involves a collective communication.
 *
 *****************************************************************************/

void COLL_set_fluid_gravity() {

  int ia, nc;
  double rvolume;
  double f[3];

  nc = colloid_ntotal();

  if (gravity_ && nc > 0) {

    rvolume = 1.0/site_map_volume(FLUID);

    /* Force per fluid node to balance is... */

    for (ia = 0; ia < 3; ia++) {
      f[ia] = -g_[ia]*rvolume*nc;
    }
    fluid_body_force_set(f);
  }

  return;
}

/*****************************************************************************
 *
 *  COLL_interactions
 *
 *  For each cell in the domain proper, look for interactions between
 *  colloids in the same cell, and all the surrounding cells. Add any
 *  resulting forces to each particle involved.
 *
 *  Double-counting of interactions is avoided by checking the unique
 *  indexes of each colloid (i < j).
 *
 *  The cell list approach maintains O(N) effort.
 *  The minumum separation between particles encountered is returned.
 *
 *****************************************************************************/

double COLL_interactions() {

  colloid_t * p_c1;
  colloid_t * p_c2;

  int    ia;
  int    ic, jc, kc, id, jd, kd, dx, dy, dz;
  double hmin = L(X);
  double h, fmod;
  double torque_mag[3];
  double f[3];

  double r12[3];

  epotential_ = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_c1 = colloids_cell_list(ic, jc, kc);

	while (p_c1) {

	  /* Single particle contributions here, if required. */

	  /* External fields - here gravity, magnetic torque */

	  magnetic_field_torque(p_c1->s.s,  torque_mag);

	  for (ia = 0; ia < 3; ia++) {
	    p_c1->force[ia] += g_[ia];
	    p_c1->torque[ia] += torque_mag[ia];
	  }

	  for (dx = -1; dx <= +1; dx++) {
	    for (dy = -1; dy <= +1; dy++) {
	      for (dz = -1; dz <= +1; dz++) {

		id = ic + dx;
		jd = jc + dy;
		kd = kc + dz;

		p_c2 = colloids_cell_list(id, jd, kd);

		while (p_c2) {

		  /* Compute the interaction once only */

		  if (p_c1->s.index < p_c2->s.index) {

		    /* Compute the unit vector r12, and the gap, h */

		    coords_minimum_distance(p_c1->s.r, p_c2->s.r, r12);
		    h = modulus(r12);
		    lubrication_sphere_sphere(p_c1->s.ah, p_c2->s.ah,
					      p_c1->s.v,  p_c2->s.v,
					      r12, f);
		    r12[X] /= h;
		    r12[Y] /= h;
		    r12[Z] /= h;

		    h = h - p_c1->s.ah - p_c2->s.ah;
		    if (h < hmin) hmin = h;
		    if (h < 0.0) COLL_overlap(p_c1, p_c2);

		    /* soft sphere */

		    fmod = soft_sphere_force(h);
		    fmod += yukawa_force(h + p_c1->s.ah + p_c2->s.ah);

		    for (ia = 0; ia < 3; ia++) {
		      p_c1->force[ia] += f[ia];
		      p_c2->force[ia] -= f[ia];
		      p_c1->force[ia] -= fmod*r12[ia];
		      p_c2->force[ia] += fmod*r12[ia];
		    }

		    epotential_ += soft_sphere_energy(h);
		    epotential_ += yukawa_potential(p_c1->s.ah + p_c2->s.ah + h);
		  }
		  
		  /* Next colloid */
		  p_c2 = p_c2->next;
		}

		/* Next search cell */
	      }
	    }
	  }

	  /* Next colloid */
	  p_c1 = p_c1->next;
	}

	/* Next cell */
      }
    }
  }

  return hmin;
}

/*****************************************************************************
 *
 *  colloid_forces_check
 *
 *  Check the cell list width against the current interaction cut-off
 *  lengths.
 *
 *  For surface-surface separation based potentials, the criterion has
 *  a contribution of the largest colloid diameter present. For centre-
 *  centre calculations (such as Yukawa), this is not required.
 *
 *****************************************************************************/

static void colloid_forces_check(void) {

  double ahmax;
  double rmax = 0.0;
  double lmin = DBL_MAX;
  double rc;

  if (potential_centre_to_centre()) {
    ahmax = 0.0;
  }
  else {
    ahmax = colloid_forces_ahmax();
  }

  /* Work out the maximum cut-off range */

  rc = 2.0*ahmax + lubrication.cutoff_norm;
  rmax = dmax(rmax, rc);
  rc = 2.0*ahmax + lubrication.cutoff_tang;
  rmax = dmax(rmax, rc);

  rc = 2.0*ahmax + get_max_potential_range();
  rmax = dmax(rmax, rc);

  /* Check against the cell list */

  lmin = dmin(lmin, colloids_lcell(X));
  lmin = dmin(lmin, colloids_lcell(Y));
  lmin = dmin(lmin, colloids_lcell(Z));

  if (colloid_ntotal() > 1 && rmax > lmin) {
    info("Cell list width too small to capture specified interactions!\n");
    info("The maximum interaction range is: %f\n", rmax);
    info("The minumum cell width is only:   %f\n", lmin);
    fatal("Please check and try again\n");
  }

  return;
}

/*****************************************************************************
 *
 *  COLL_lubrication
 *
 *  Compute the net lubrication correction for the two colloids
 *  which are separated by vector r_ij (the size of which is h
 *  i.e., centre-centre distance).
 *
 *  If noise is on, then an additional random force is required to
 *  satisfy the fluctuation-dissipation theorem. The size of the
 *  random component depends on the "temperature", and is just
 *  added to the lubrication contribution.
 *
 *****************************************************************************/

void lubrication_sphere_sphere(double a1, double a2,
			       const double u1[3], const double u2[3],
			       const double r12[3], double f[3]) {
  int ia;
  double h;        /* Separation */
  double hr;       /* Reduced separation */
  double eta;      /* viscosity */
  double fmod;
  double rh, rhr;
  double rdotdu;
  double rhat[3];

  double rn = 1.0/lubrication.cutoff_norm;
  double rt = 1.0/lubrication.cutoff_tang;

  for (ia = 0; ia < 3; ia++) {
    f[ia] = 0.0;
  }

  if (lubrication.corrections_on) {

    h = modulus(r12);
    hr = h - a1 - a2;
    eta = get_eta_shear();

    if (hr < lubrication.cutoff_norm) {

      rhr = 1.0/hr;
      fmod = -6.0*pi_*eta*a1*a1*a2*a2*(rhr - rn)/((a1 + a1)*(a2 + a2));

      /* Fluctuation/dissipation contribution */
      fmod += ran_parallel_gaussian()*sqrt(-2.0*get_kT()*fmod);

      rh = 1.0/h;
      rdotdu = 0.0;

      for (ia = 0; ia < 3; ia++) {
	rhat[ia] = rh*r12[ia];
	rdotdu += rhat[ia]*(u1[ia] - u2[ia]);
      }

      for (ia = 0; ia < 3; ia++) {
	f[ia] += fmod*rdotdu*rhat[ia];
      }

      /* Tangential lubrication correction */

      if (hr < lubrication.cutoff_tang) {

	rh = 0.5*(a1 + a2)*rhr;

	fmod = -(24.0/15.0)*pi_*eta*a1*a2*(2.0*a1*a1 + a1*a2 + 2.0*a2*a2)
	  *(log(rh) - log(0.5*(a1 + a2)*rt)) / ((a1+a2)*(a1+a2)*(a1+a2));

	for (ia = 0; ia < 3; ia++) {
	  f[ia] += fmod*((u1[ia] - u2[ia]) - rdotdu*rhat[ia]);
	}
      }

    }
  }

  return;
}

/*****************************************************************************
 *
 *  COLL_overlap
 *
 *  Action on detection of overlapping particles.
 *
 *****************************************************************************/

void COLL_overlap(colloid_t * p_c1, colloid_t * p_c2) {

  verbose("Detected overlapping particles\n");
  colloid_state_write_ascii(p_c1->s, stdout);
  colloid_state_write_ascii(p_c2->s, stdout);
  fatal("Stopping");

  return;
}

/*****************************************************************************
 *
 *  coll_position_update
 *
 *  Update the colloid positions (all cells).
 *
 *  Moving a particle more than 1 lattice unit in any direction can
 *  cause it to leave the cell list entirely, which ends in
 *  catastrophe. We therefore have a check here against a maximum
 *  velocity (effectively dr) and stop if the check fails.
 *
 *****************************************************************************/

void coll_position_update(void) {

  int ia;
  int ic, jc, kc;
  int ifail;

  const double drmax[3] = {0.8, 0.8, 0.8};

  colloid_t * p_colloid;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	  while (p_colloid) {

	    ifail = 0;
	    for (ia = 0; ia < 3; ia++) {
	      if (p_colloid->s.dr[ia] > drmax[ia]) ifail = 1;
	      p_colloid->s.r[ia] += p_colloid->s.dr[ia];
	    }

	    if (ifail == 1) {
	      verbose("Colloid velocity exceeded maximum %7.3f %7.3f %7.3f\n",
		      drmax[X], drmax[Y], drmax[Z]);
	      colloid_state_write_ascii(p_colloid->s, stdout);
	      fatal("Stopping\n");
	    }

	    p_colloid = p_colloid->next;
	  }
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_forces_ahmax
 *
 *  At start-up, we may need to examine what size of particles are
 *  present. This affects the largest interaction distance.
 *
 *****************************************************************************/

double colloid_forces_ahmax(void) {

  int ic, jc, kc;
  double ahmax;
  double ahmax_local;
  colloid_t * pc;

  ahmax_local = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	pc = colloids_cell_list(ic, jc, kc);

	while (pc) {
	  ahmax_local = dmax(ahmax_local, pc->s.ah);
	  pc = pc->next;
	}
      }
    }
  }

  MPI_Allreduce(&ahmax_local, &ahmax, 1, MPI_DOUBLE, MPI_MAX, pe_comm());

  return ahmax;
}

/*****************************************************************************
 *
 *  colloid_gravity
 *
 *****************************************************************************/

void colloid_gravity(double f[3]) {

  f[X] = g_[X];
  f[Y] = g_[Y];
  f[Z] = g_[Z];

  return;
}

/*****************************************************************************
 *
 *  colloid_gravity_set
 *
 *****************************************************************************/

void colloid_gravity_set(const double f[3]) {

  g_[X] = f[X];
  g_[Y] = f[Y];
  g_[Z] = f[Z];
  gravity_ = 1;

  return;
}
