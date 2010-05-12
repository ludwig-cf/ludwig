/*****************************************************************************
 *
 *  interaction.c
 *
 *  Colloid potentials and colloid-colloid interactions.
 *
 *  Refactoring is in progress.
 *
 *  $Id: interaction.c,v 1.18.4.4 2010-05-12 18:17:10 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "ran.h"
#include "runtime.h"
#include "free_energy.h"

#include "active.h"
#include "build.h"
#include "physics.h"
#include "potential.h"

#include "colloids.h"
#include "interaction.h"
#include "model.h"
#include "site_map.h"
#include "collision.h"
#include "cio.h"
#include "control.h"
#include "subgrid.h"

#include "ccomms.h"
#include "ewald.h"

static void    COLL_overlap(Colloid *, Colloid *);
static void    COLL_set_fluid_gravity(void);
static FVector COLL_lubrication(Colloid *, Colloid *, FVector, double);
static void    COLL_init_colloids_test(void);
static void    COLL_test_output(void);
static void    coll_position_update(void);
static double  coll_max_speed(void);

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

  if (get_N_colloid() == 0) return;

  TIMER_start(TIMER_PARTICLE_HALO);

  coll_position_update();
  cell_update();
  CCOM_halo_particles();

  TIMER_stop(TIMER_PARTICLE_HALO);

#ifndef _SUBGRID_

  /* Removal or replacement of fluid requires a lattice halo update */
  halo_site();

  TIMER_start(TIMER_REBUILD);
  COLL_update_map();
  COLL_remove_or_replace_fluid();
  COLL_update_links();

  TIMER_stop(TIMER_REBUILD);

  COLL_test_output();
  COLL_forces();
  active_bbl_prepass();

#else /* _SUBGRID_ */
  COLL_test_output();
  COLL_forces();
  subgrid_force_from_particles();
#endif /* SUBGRID */

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

  char filename[FILENAME_MAX];
  char tmp[128];
  int nc = 0;
  int ifrom_file = 0;
  double ahmax;

  void CMD_init_volume_fraction(void);
  void lubrication_init(void);
  void check_interactions(const double);
  void monte_carlo(void);
  void phi_gradients_set_solid(void);

  /* Default position: no colloids */

  RUN_get_string_parameter("colloid_init", tmp, 128);
  if (strcmp(tmp, "no_colloids") == 0) nc = 0;

  /* This is just to get past the start. */
  if (strcmp(tmp, "fixed_volume_fraction_monodisperse") == 0) nc = 1;
  if (strcmp(tmp, "fixed_number_monodisperse") == 0) nc = 1;
  if (strcmp(tmp, "from_file") == 0) {
    nc = 1;
    ifrom_file = 1;
  }


#ifdef _COLLOIDS_TEST_
  nc = 1;
#endif

  if (nc == 0) return;

  /* Was comm_init */
  colloid_io_init();

  nc = RUN_get_double_parameter("colloid_ah", &ahmax);
  if (nc == 0) fatal("Please set colloids_ah in the input file\n");

  /* Initialisation section. */

  colloids_init();

  if (get_step() == 0 && ifrom_file == 0) {

#ifdef _COLLOIDS_TEST_
    COLL_init_colloids_test();
#else
    CMD_init_volume_fraction();
    init_active();
#endif
  }
  else {
    /* Restart from previous configuration */

    if (get_step() == 0) {
      sprintf(filename, "%s", "config.cds.init");
    }
    else {
      sprintf(filename, "%s%6.6d", "config.cds", get_step());
    }
    info("Reading colloid information from files: %s\n", filename);

    colloid_io_read(filename);
  }

  CCOM_init_halos();

  /* ewald_init(0.285, 16.0);*/

  lubrication_init();
  soft_sphere_init();
  leonard_jones_init();
  yukawa_init();
  check_interactions(ahmax);


  COLL_init_coordinates();

  if (get_step() == 0 && ifrom_file == 0) monte_carlo();

  /* Transfer any particles in the halo regions, initialise the
   * colloid map and build the particles for the first time. */

  CCOM_halo_particles();

#ifndef _SUBGRID_
  COLL_update_map();
  COLL_update_links();
#endif /* _SUBGRID_ */

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

void lubrication_init() {

  int n;

  info("\nColloid-colloid lubrication corrections\n");

  n = RUN_get_int_parameter("lubrication_on", &(lubrication.corrections_on));
  info((n == 0) ? "[Default] " : "[User   ] ");
  info("Lubrication corrections are switched %s\n",
       (lubrication.corrections_on == 0) ? "off" : "on");

  if (lubrication.corrections_on) {
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
 *  COLL_finish
 *
 *  Execute once at the end of the model run.
 *
 *****************************************************************************/

void COLL_finish() {

  if (get_N_colloid() == 0) return;

  colloids_finish();

  return;
}


/*****************************************************************************
 *
 *  COLL_init_colloids_test
 *
 *  This is a routine which hardwires a small number
 *  of colloids for tests of colloid physics.
 *
 *  Serial.
 *
 *****************************************************************************/

void COLL_init_colloids_test() {

#ifdef _COLLOIDS_TEST_AUTOCORRELATION_

  FVector r0, v0, omega0;
  IVector cell;
  Colloid * p_colloid;
  double    a0 = 2.3;
  double    ah = 2.3;

  /* Autocorrelation test. */

  set_N_colloid(1);

  r0.x =  .0 + 1.0*L(X);
  r0.y =  .0 + 1.0*L(Y);
  r0.z =  .0 + 0.5*L(Z);

  v0.x = 1.0*get_eta_shear()/ah;
  v0.y = 0.0;
  v0.z = 0.0;

  omega0.x = 0.0;
  omega0.y = 0.0;
  omega0.z = 0.0*get_eta_shear()/(ah*ah);

  /* Look at the proposed position and decide whether it is in
   * the local domain. If it is, then it can be added. */

  cell = cell_coords(r0);
  if (cell.x < 1 || cell.x > Ncell(X)) return;
  if (cell.y < 1 || cell.y > Ncell(Y)) return;
  if (cell.z < 1 || cell.z > Ncell(Z)) return;

  p_colloid = COLL_add_colloid(1, a0, ah, r0, v0, omega0);

  p_colloid->stats[X] = v0.x;
  p_colloid->dir.x = 1.0;
  p_colloid->dir.y = 0.0;
  p_colloid->dir.z = 0.0;

#endif

  return;
}


/*****************************************************************************
 *
 *  COLL_test_output
 *
 *  Look at the particles in the domain proper and perform
 *  diagnostic output as required.
 *
 *****************************************************************************/

void COLL_test_output() {

  Colloid * p_colloid;
  int       ic, jc, kc;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {
#ifdef _COLLOIDS_TEST_AUTOCORRELATION_
	  verbose("Autocorrelation test output: %10.9f %10.9f\n",
	    p_colloid->r.x, p_colloid->v.x/p_colloid->stats[X]);
	  /*verbose("Autocorrelation omega: %10.9f %10.9f %10.9f\n",
		  p_colloid->omega.z/p_colloid->stats[X], p_colloid->dir.x,
		  p_colloid->dir.y);*/
#endif
#ifdef _COLLOIDS_TEST_CALIBRATE_
	  verbose("Calibrate: %g %g %g %g %g %g\n",
		  p_colloid->v.x, p_colloid->v.y, p_colloid->v.z,
		  p_colloid->force.x, p_colloid->force.y, p_colloid->force.z);
#endif
		  
	  p_colloid = p_colloid->next;

	}
      }
    }
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

  int nc = get_N_colloid();
  double hminlocal;
  double hmin;
  double elocal[2];
  double e[2];

  if (nc > 0) {

    COLL_zero_forces();
    hminlocal = COLL_interactions();
    COLL_set_fluid_gravity();
    ewald_sum();

    if (is_statistics_step()) {

      double ereal, efour, eself;
      double rnkt = 1.0/(nc*get_kT());

      /* Note Fourier space and self energy available on all processes */
      ewald_total_energy(elocal, elocal + 1, &eself);

      MPI_Reduce(&hminlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(elocal, e, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      ereal = e[0];
      efour = e[1];

      info("\nParticle statistics:\n");
      info("[Inter-particle gap minimum is: %f]\n", hmin);
      info("[Energies (perNkT): Ewald (r) Ewald (f) Ewald (s) Pot. Total\n");
      info("Energy: %g %g %g %g %g]\n", rnkt*ereal, rnkt*efour,
	   rnkt*eself, rnkt*epotential_,
	   rnkt*(ereal + efour + eself + epotential_));
      info("[Max particle speed: %g]\n", coll_max_speed());
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

  int       ic, jc, kc;
  Colloid * p_colloid;

  /* Only real particles need correct external force. */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {
	  p_colloid->force  = UTIL_fvector_zero();
	  p_colloid->torque = UTIL_fvector_zero();
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
 *  Issues
 *
 *****************************************************************************/

void COLL_set_fluid_gravity() {

  double volume;
  double g[3];
  extern double siteforce[3];

  volume = site_map_volume(FLUID);
  get_gravity(g);

  /* Size of force per fluid node */

  siteforce[X] = -get_N_colloid()*g[X]/volume;
  siteforce[Y] = -get_N_colloid()*g[Y]/volume;
  siteforce[Z] = -get_N_colloid()*g[Z]/volume;

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

  Colloid * p_c1;
  Colloid * p_c2;

  int    ic, jc, kc, id, jd, kd, dx, dy, dz;
  double hmin = L(X);
  double h, fmod;
  double g[3];

  FVector r_12, f;
  FVector COLL_fvector_separation(FVector, FVector);
  get_gravity(g);

  epotential_ = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_c1 = CELL_get_head_of_list(ic, jc, kc);

	while (p_c1) {

	  /* Single particle contributions here, if required. */

	  /* External gravity */
	  p_c1->force.x += g[X];
	  p_c1->force.y += g[Y];
	  p_c1->force.z += g[Z];

	  for (dx = -1; dx <= +1; dx++) {
	    for (dy = -1; dy <= +1; dy++) {
	      for (dz = -1; dz <= +1; dz++) {

		id = ic + dx;
		jd = jc + dy;
		kd = kc + dz;

		p_c2 = CELL_get_head_of_list(id, jd, kd);

		while (p_c2) {

		  /* Compute the interaction once only */

		  if (p_c1->index < p_c2->index) {

		    /* Compute the unit vector r_12, and the gap, h */

		    r_12 = COLL_fvector_separation(p_c1->r, p_c2->r);
		    h = sqrt(r_12.x*r_12.x + r_12.y*r_12.y + r_12.z*r_12.z);

		    f = COLL_lubrication(p_c1, p_c2, r_12, h);
		    p_c1->force = UTIL_fvector_add(p_c1->force, f);
		    p_c2->force = UTIL_fvector_subtract(p_c2->force, f);

		    /* Conservative forces: use unit vector */
		    r_12.x /= h;
		    r_12.y /= h;
		    r_12.z /= h;
		    h = h - p_c1->ah - p_c2->ah;
		    if (h < hmin) hmin = h;
		    if (h < 0.0) COLL_overlap(p_c1, p_c2);

		    /* soft sphere */

		    fmod = soft_sphere_force(h);
		    fmod += yukawa_force(h + p_c1->ah + p_c2->ah);

		    f.x = -fmod*r_12.x;
		    f.y = -fmod*r_12.y;
		    f.z = -fmod*r_12.z;
		    p_c1->force = UTIL_fvector_add(p_c1->force, f);
		    p_c2->force = UTIL_fvector_subtract(p_c2->force, f);

		    epotential_ += soft_sphere_energy(h);
		    epotential_ += yukawa_potential(p_c1->ah + p_c2->ah + h);
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
 *  check_interactions
 *
 *  Check the cell list width against the current interaction cut-off
 *  lengths.
 *
 *****************************************************************************/

void check_interactions(double ahmax) {

  double rmax = 0.0;
  double lmin = DBL_MAX;
  double rc;

  info("\nChecking cell list against specified interactions\n");

  /* Work out the maximum cut-off range */

  rc = 2.0*ahmax + lubrication.cutoff_norm;
  rmax = dmax(rmax, rc);
  rc = 2.0*ahmax + lubrication.cutoff_tang;
  rmax = dmax(rmax, rc);

  rc = 2.0*ahmax + get_max_potential_range();
  rmax = dmax(rmax, rc);

  /* Check against the cell list */

  lmin = dmin(lmin, Lcell(X));
  lmin = dmin(lmin, Lcell(Y));
  lmin = dmin(lmin, Lcell(Z));

  if (rmax > lmin) {
    info("Cell list width too small to capture specified interactions!\n");
    info("The maximum interaction range is: %f\n", rmax);
    info("The minumum cell width is only:   %f\n", lmin);
    fatal("Please check and try again\n");
  }
  else {
    info("The maximum interaction range is: %f\n", rmax);
    info("The minimum cell width is %f (ok)\n", lmin);
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

FVector COLL_lubrication(Colloid * p_i, Colloid * p_j, FVector r_ij, double h) {

  FVector force;

  force = UTIL_fvector_zero();

  if (lubrication.corrections_on) {

    FVector du;
    FVector runit;
    double   ai, aj;
    double   rh;
    double   rdotdu;
    double   fmod;

    /* Define the surface-surface separation */

    ai = p_i->ah;
    aj = p_j->ah;
    h  = h - ai - aj;

    if (h < lubrication.cutoff_norm) {

      double rn = 1.0/lubrication.cutoff_norm;

      /* Compute the separation unit vector in the direction of r_ij */

      rh = 1.0 / (h + ai + aj);
      runit.x = rh*r_ij.x;
      runit.y = rh*r_ij.y;
      runit.z = rh*r_ij.z;

      /* Normal lubrication correction */

      rh     = 1.0 / h;
      du     = UTIL_fvector_subtract(p_i->v, p_j->v);
      rdotdu = UTIL_dot_product(runit, du);
      fmod   = -6.0*PI*get_eta_shear()*ai*ai*aj*aj*(rh - rn)
	/ ((ai+ai)*(aj+aj));

#ifdef _NOISE_
      /* Fluctuation/dissipation */
      fmod += ran_parallel_gaussian()*sqrt(-2.0*get_kT()*fmod);
#endif
      force.x += fmod*rdotdu*runit.x;
      force.y += fmod*rdotdu*runit.y;
      force.z += fmod*rdotdu*runit.z;

      /* Tangential lubrication correction */
      if (h < lubrication.cutoff_tang) {

	double rt = 1.0/lubrication.cutoff_tang;

	rh = 0.5*(ai+aj)/h;
	fmod = -(24.0/15.0)*PI*get_eta_shear()*ai*aj*
	  (2.0*ai*ai + ai*aj + 2.0*aj*aj)*
	  (log(rh) - log(0.5*(ai+aj)*rt)) / ((ai+aj)*(ai+aj)*(ai+aj));

	force.x += fmod*(du.x - rdotdu*runit.x);
	force.y += fmod*(du.y - rdotdu*runit.y);
	force.z += fmod*(du.z - rdotdu*runit.z);
      }
    }
  }

  return force;
}

/*****************************************************************************
 *
 *  COLL_overlap
 *
 *  Action on detection of overlapping particles.
 *
 *****************************************************************************/

void COLL_overlap(Colloid * p_c1, Colloid * p_c2) {

  verbose("Detected overlapping particles\n");
  verbose("Particle[%d] at (%f,%f,%f)\n", p_c1->index, p_c1->r.x, p_c1->r.y,
	  p_c1->r.z);
  verbose("particle[%d] at (%f,%f,%f)\n", p_c2->index, p_c2->r.x, p_c2->r.y,
	  p_c2->r.z);
  fatal("Stopping");

  return;
}

/*****************************************************************************
 *
 *  coll_position_update
 *
 *  Update the colloid positions (all cells).
 *
 *****************************************************************************/

void coll_position_update(void) {

  Colloid * p_colloid;
  int ic, jc, kc;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	  while (p_colloid) {

	    p_colloid->r.x += p_colloid->dr[X];
	    p_colloid->r.y += p_colloid->dr[Y];
	    p_colloid->r.z += p_colloid->dr[Z];

	    p_colloid = p_colloid->next;
	  }
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  coll_max_speed
 *
 *  Return the largest current colloid velocity.
 *
 ****************************************************************************/ 

double coll_max_speed() {

  Colloid * p_colloid;
  int ic, jc, kc;
  double vmaxlocal = 0.0;
  double vmax = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  vmaxlocal = dmax(vmaxlocal,
			   UTIL_dot_product(p_colloid->v, p_colloid->v));

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  MPI_Reduce(&vmaxlocal, &vmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  /* Remember to take sqrt(), as we have computed v^2 */
  vmax = sqrt(vmax);

  return vmax;
}
