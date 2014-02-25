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

#include "potential.h"

#include "colloids.h"
#include "colloid_io_rt.h"
#include "interaction.h"
#include "model.h"
#include "cio.h"
#include "control.h"
#include "subgrid.h"
#include "stats_colloid.h"

#include "util.h"
#include "colloid_sums.h"
#include "colloids_halo.h"
#include "colloids_init.h"
#include "ewald.h"

static int colloid_forces_fluid_gravity_set(map_t * map);
static int colloid_forces(map_t * map, psi_t * psi);

static void colloid_forces_overlap(colloid_t *, colloid_t *);
static void colloid_forces_pairwise(double * h, double * e);
static void colloid_forces_zero_set(void);
static void colloid_forces_single_particle_set(psi_t * psi);

void lubrication_sphere_sphere(double a1, double a2,
			       const double u1[3], const double u2[3],
			       const double r12[3], double f[3]);
static void    coll_position_update(void);
static void    lubrication_init(void);
static void    colloid_forces_check(void);

static int    cell_list_interactions_ = 1;

struct lubrication_struct {
  int corrections_on;
  double cutoff_norm;  /* Normal */
  double cutoff_tang;  /* Tangential */
} lubrication;

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

int COLL_update(hydro_t * hydro, map_t * map, field_t * fphi, field_t * fp,
		field_t * fq, psi_t * psi) {

  int iconserve ;
  int is_subgrid = 0;

  if (colloid_ntotal() == 0) return 0;

  subgrid_on(&is_subgrid);
  iconserve = (psi || (fphi && distribution_ndist() == 1));

  TIMER_start(TIMER_PARTICLE_HALO);

  coll_position_update();
  colloids_cell_update();
  colloids_halo_state();

  TIMER_stop(TIMER_PARTICLE_HALO);

  if (is_subgrid) {
    colloid_forces(map, psi);
    subgrid_force_from_particles(hydro);
  }
  else {

    /* Removal or replacement of fluid requires a lattice halo update */
    TIMER_start(TIMER_HALO_LATTICE);
    distribution_halo();
    TIMER_stop(TIMER_HALO_LATTICE);

    TIMER_start(TIMER_REBUILD);

    build_update_map(map);
    build_remove_or_replace_fluid(fphi, fp, fq, psi);
    build_update_links(map);

    TIMER_stop(TIMER_REBUILD);

    if (iconserve) colloid_sums_halo(COLLOID_SUM_CONSERVATION);
    if (iconserve) build_conservation(fphi, psi);

    colloid_forces(map, psi);
  }

  return 0;
}

/*****************************************************************************
 *
 *  COLL_init
 * 
 *  Driver routine for colloid initialisation.
 *
 *****************************************************************************/

int COLL_init(map_t * map) {

  int n, ncheck;
  int init_from_file;
  int init_random;
  int ncell[3];
  int gravity;
  char stub[FILENAME_MAX];
  char filename[FILENAME_MAX];
  char subdirectory[FILENAME_MAX];
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

  /* Always use the cell list for the time being */

  cell_list_interactions_ = 1;

  if (cell_list_interactions_ == 0) {
    /* We use ncell = 2 */
    ncell[X] = 2;
    ncell[Y] = 2;
    ncell[Z] = 2;
    colloids_cell_ncell_set(ncell);
  }

  if (cell_list_interactions_ == 1 && (init_random || init_from_file)) {

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

  if (init_random || init_from_file) {

      pe_subdirectory(subdirectory);

    /* Initialisation section. */

    colloid_io_run_time();

    if (init_from_file) {

      if (get_step() == 0) {
	strcpy(stub, "config.cds.init");
	RUN_get_string_parameter("colloid_file_stub", stub, FILENAME_MAX);
	sprintf(filename, "%s%s", subdirectory, stub);
      }
      else {
	strcpy(stub, "config.cds");
	RUN_get_string_parameter("colloid_file_stub", stub, FILENAME_MAX);
	sprintf(filename, "%s%s%8.8d", subdirectory, stub, get_step());
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
      RUN_get_double_parameter("colloid_random_c", &state0->c);
      RUN_get_double_parameter("colloid_random_h", &state0->h);
      RUN_get_double_parameter("colloid_random_b1", &state0->b1);
      RUN_get_double_parameter("colloid_random_b2", &state0->b2);
      RUN_get_double_parameter("colloid_random_dh", &dh);

      RUN_get_int_parameter("colloid_random_isfixedr", &state0->isfixedr);
      RUN_get_int_parameter("colloid_random_isfixedv", &state0->isfixedv);

      RUN_get_double_parameter("colloid_random_q0", &state0->q0);
      RUN_get_double_parameter("colloid_random_q1", &state0->q1);
      RUN_get_double_parameter("colloid_random_epsilon", &state0->epsilon);

      colloids_init_random(n, state0, dh);
      ncheck = colloid_ntotal();
      info("Requested   %d colloid%s from input\n", n, (n > 1) ? "s" : "");
      info("Initialised %d colloid%s\n", ncheck, (ncheck == 1) ? "" : "s");
      info("Colloid  radius a0 = %le\n", state0->a0);
      info("Hydrodyn radius ah = %le\n", state0->ah);
      info("Colloid charges q0 = %le    q1 = %le\n", state0->q0, state0->q1);

      free(state0);
    }

    n = RUN_get_double_parameter_vector("colloid_gravity", g);
    if (n != 0) physics_fgrav_set(g);

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

    if (strcmp(keyvalue, "subgrid") == 0) {
      subgrid_on_set();
    }
    else  {
      build_update_map(map);
      build_update_links(map);
    }

    /* Information */
    gravity = 0;
    gravity = (g[X] != 0.0 || g[Y] != 0.0 || g[Z] != 0.0);

    if (gravity) {
      info("Sedimentation force on:   yes\n");
      info("Sedimentation force:      %14.7e %14.7e %14.7e", g[X], g[Y], g[Z]);
    }
    info("\n");
  }

  return 0;
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
 *  colloid_forces
 *
 *  Top-level function for compuatation of external forces to be called
 *  once per time step. Note that particle copies in the halo regions
 *  must have zero external force/torque on exit from this routine.
 *
 *****************************************************************************/

static int colloid_forces(map_t * map, psi_t * psi) {

  int nc;
  double hmin, hminlocal;
  double etotal, elocal;

  nc = colloid_ntotal();

  hminlocal = L(X);
  elocal = 0.0;

  if (nc > 0) {
    colloid_forces_zero_set();
    colloid_forces_single_particle_set(psi);
    colloid_forces_fluid_gravity_set(map);

    if (nc > 1) {
      colloid_forces_pairwise(&hminlocal, &elocal);
      ewald_sum();
    }

    if (is_statistics_step()) {

      info("\nParticle statistics:\n");

      if (nc > 1) {
	MPI_Reduce(&hminlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, 0, pe_comm());
	MPI_Reduce(&elocal, &etotal, 1, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

	info("Inter-particle minimum h is: %10.5e\n", hmin);
	info("Potential energy is:         %10.5e\n", etotal);
      }

      stats_colloid_velocity_minmax();
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  colloid_forces_zero_set
 *
 *  Set the external forces on the particles to zero (including halos).
 *  All additional forces are then accumulated.
 *
 *****************************************************************************/

static void colloid_forces_zero_set(void) {

  int       ic, jc, kc, ia;
  colloid_t * p_colloid;

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
 *  colloid_forces_single_particle_set
 *
 *  Accumulate single particle force contributions.
 *
 *  psi may be NULL, in which case, assume no charged species, otherwise
 *  we assume two.
 *
 *****************************************************************************/

static void colloid_forces_single_particle_set(psi_t * psi) {

  int ic, jc, kc, ia;
  double g[3];           /* 'Gravity' */
  double b0[3];          /* external fields */
  double btorque[3];
  colloid_t * pc;

  physics_b0(b0);
  physics_fgrav(g);

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	pc = colloids_cell_list(ic, jc, kc);

	while (pc) {

	  btorque[X] = pc->s.s[Y]*b0[Z] - pc->s.s[Z]*b0[Y];
	  btorque[Y] = pc->s.s[Z]*b0[X] - pc->s.s[X]*b0[Z];
	  btorque[Z] = pc->s.s[X]*b0[Y] - pc->s.s[Y]*b0[X];

	  for (ia = 0; ia < 3; ia++) {
	    pc->force[ia] += g[ia];                /* Gravity */
	    pc->torque[ia] += btorque[ia];         /* Magnetic field */
	  }

	  pc = pc->next;
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_forces_fluid_gravity_set
 *
 *  Set the current gravtitational force on the fluid. This should
 *  match, exactly, the force on the colloids and so depends on the
 *  current number of fluid sites globally (fluid volume).
 *
 *  Note the volume calculation involves a collective communication.
 *
 *****************************************************************************/

static int colloid_forces_fluid_gravity_set(map_t * map) {

  int ia, nc;
  int nsfluid;
  int gravity = 0;
  double g[3];
  double rvolume;
  double f[3];

  nc = colloid_ntotal();
  physics_fgrav(g);
  gravity = (g[X] != 0.0 || g[Y] != 0.0 || g[Z] != 0.0);

  if (gravity && nc > 0) {
    assert(map);
    map_volume_allreduce(map, MAP_FLUID, &nsfluid);
    rvolume = 1.0/nsfluid;

    /* Force per fluid node to balance is... */

    for (ia = 0; ia < 3; ia++) {
      f[ia] = -g[ia]*rvolume*nc;
    }

    physics_fbody_set(f);
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_forces_interactions
 *
 *  For each cell in the domain proper, look for interactions between
 *  colloids in the same cell, and all the surrounding cells. Add any
 *  resulting forces to each particle involved.
 *
 *  Double-counting of interactions is avoided by checking the unique
 *  indexes of each colloid (i < j).
 *
 *  The minumum separation between particles encountered and the
 *  potential energy are returned.
 *
 *****************************************************************************/

static void colloid_forces_pairwise(double * hmin, double * epot) {

  colloid_t * p_c1;
  colloid_t * p_c2;

  int ia;
  int ic, jc, kc, id, jd, kd, dx, dy, dz;
  int dxm, dxp, dym, dyp, dzm, dzp;

  double h, fmod;
  double f[3];
  double r12[3];

  for (ic = 1; ic <= Ncell(X); ic++) {
    dxm = 2 - cell_list_interactions_;
    dxp = 2 - cell_list_interactions_;
    if (ic - dxm < 0) dxm = 1;
    if (ic + dxp > Ncell(X) + 1) dxp = 1;

    for (jc = 1; jc <= Ncell(Y); jc++) {
      dym = 2 - cell_list_interactions_;
      dyp = 2 - cell_list_interactions_;
      if (jc - dym < 0) dym = 1;
      if (jc + dyp > Ncell(Y) + 1) dyp = 1;

      for (kc = 1; kc <= Ncell(Z); kc++) {
	dzm = 2 - cell_list_interactions_;
	dzp = 2 - cell_list_interactions_;
	if (kc - dzm < 0) dzm = 1;
	if (kc + dzp > Ncell(Z) + 1) dzp = 1;

	p_c1 = colloids_cell_list(ic, jc, kc);

	while (p_c1) {

	  for (dx = -dxm; dx <= +dxp; dx++) {
	    for (dy = -dym; dy <= +dyp; dy++) {
	      for (dz = -dzm; dz <= +dzp; dz++) {

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
		    if (h < *hmin) *hmin = h;
		    if (h < 0.0) colloid_forces_overlap(p_c1, p_c2);

		    /* soft sphere */

		    fmod = soft_sphere_force(h);
		    fmod += yukawa_force(h + p_c1->s.ah + p_c2->s.ah);

		    for (ia = 0; ia < 3; ia++) {
		      p_c1->force[ia] += f[ia];
		      p_c2->force[ia] -= f[ia];
		      p_c1->force[ia] -= fmod*r12[ia];
		      p_c2->force[ia] += fmod*r12[ia];
		    }

		    *epot += soft_sphere_energy(h);
		    *epot += yukawa_potential(p_c1->s.ah + p_c2->s.ah + h);
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

  return;
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

  int ifail;
  int nhalo;
  int nlocal[3];

  double ahmax;
  double rmax = 0.0;
  double lmin = DBL_MAX;
  double rc;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

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

  if (cell_list_interactions_ == 0) {

    /* The cell list contraint is relaxed to be */
    lmin = dmin(lmin, 1.0*nlocal[X]);
    lmin = dmin(lmin, 1.0*nlocal[Y]);
    lmin = dmin(lmin, 1.0*nlocal[Z]);

    /* The particles must not be larger than nlocal - nhalo or else colloid
     * information will extend beyond 2 subdomains */
    ifail = 0;
    if (2.0*ahmax >= 1.0*(nlocal[X] - nhalo)) ifail = 1;
    if (2.0*ahmax >= 1.0*(nlocal[Y] - nhalo)) ifail = 1;
    if (2.0*ahmax >= 1.0*(nlocal[Z] - nhalo)) ifail = 1;
    if (ifail == 1) {
      fatal("Particles too large for local domain (amax = %6.2f) \n", ahmax);
    }

    /* However, we can't use this if cart size > 2 and periodic
     * boundaries are present */

    ifail = 0;
    if (cart_size(X) > 2 && is_periodic(X)) ifail = 1;
    if (cart_size(Y) > 2 && is_periodic(Y)) ifail = 1;
    if (cart_size(Z) > 2 && is_periodic(Z)) ifail = 1;
    if (ifail) fatal("Must have three cells for this system\n");
  }
  else {

    /* The usual cell list contraint applies */
    lmin = dmin(lmin, colloids_lcell(X));
    lmin = dmin(lmin, colloids_lcell(Y));
    lmin = dmin(lmin, colloids_lcell(Z));
  }

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
  double kt;

  double rn = 1.0/lubrication.cutoff_norm;
  double rt = 1.0/lubrication.cutoff_tang;

  for (ia = 0; ia < 3; ia++) {
    f[ia] = 0.0;
  }

  if (lubrication.corrections_on) {

    h = modulus(r12);
    hr = h - a1 - a2;

    if (hr < lubrication.cutoff_norm) {

      physics_kt(&kt);
      physics_eta_shear(&eta);

      rhr = 1.0/hr;
      fmod = -6.0*pi_*eta*a1*a1*a2*a2*(rhr - rn)/((a1 + a1)*(a2 + a2));

      /* Fluctuation/dissipation contribution */
      fmod += ran_parallel_gaussian()*sqrt(-2.0*kt*fmod);

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
 *  colloid_forces_overlap
 *
 *  Action on detection of overlapping particles.
 *
 *****************************************************************************/

static void colloid_forces_overlap(colloid_t * p_c1, colloid_t * p_c2) {

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

	    if (p_colloid->s.isfixedr == 0) {
	      ifail = 0;
	      for (ia = 0; ia < 3; ia++) {
		if (p_colloid->s.dr[ia] > drmax[ia]) ifail = 1;
		p_colloid->s.r[ia] += p_colloid->s.dr[ia];
		/* This should trap NaNs */
		if (p_colloid->s.dr[ia] != p_colloid->s.dr[ia]) ifail = 1;
	      }

	      if (ifail == 1) {
		verbose("Colloid velocity exceeded max %7.3f %7.3f %7.3f\n",
			drmax[X], drmax[Y], drmax[Z]);
		colloid_state_write_ascii(p_colloid->s, stdout);
		fatal("Stopping\n");
	      }
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
