/*****************************************************************************
 *
 *  interaction.c
 *
 *  Colloid potentials and colloid-colloid interactions.
 *
 *  Refactoring is in progress.
 *
 *  $Id: interaction.c,v 1.14 2008-02-13 10:56:10 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

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
#include "communicate.h"
#include "model.h"
#include "lattice.h"
#include "collision.h"
#include "cio.h"
#include "control.h"
#include "subgrid.h"

#include "ccomms.h"
#include "ewald.h"

extern char * site_map;
extern int input_format;
extern int output_format;

enum { NGRAD = 27 };
static int bs_cv[NGRAD][3] = {{ 0, 0, 0}, { 1,-1,-1}, { 1,-1, 1},
			      { 1, 1,-1}, { 1, 1, 1}, { 0, 1, 0},
			      { 1, 0, 0}, { 0, 0, 1}, {-1, 0, 0},
			      { 0,-1, 0}, { 0, 0,-1}, {-1,-1,-1},
			      {-1,-1, 1}, {-1, 1,-1}, {-1, 1, 1},
			      { 1, 1, 0}, { 1,-1, 0}, {-1, 1, 0},
			      {-1,-1, 0}, { 1, 0, 1}, { 1, 0,-1},
			      {-1, 0, 1}, {-1, 0,-1}, { 0, 1, 1},
			      { 0, 1,-1}, { 0,-1, 1}, { 0,-1,-1}};


static void    COLL_compute_phi_missing(void);
static void    COLL_overlap(Colloid *, Colloid *);
static void    COLL_set_fluid_gravity(void);
static FVector COLL_lubrication(Colloid *, Colloid *, FVector, double);
static void    COLL_init_colloids_test(void);
static void    COLL_test_output(void);
static void    coll_position_update(void);
static double  coll_max_speed(void);
static int     coll_count(void);

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

  /* Default position: no colloids */

  RUN_get_string_parameter("colloid_init", tmp);
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

  nc = RUN_get_double_parameter("colloid_ah", &ahmax);
  if (nc == 0) fatal("Please set colloids_ah in the input file\n");

  /* Initialisation section. */

  colloids_init();
  CIO_set_cio_format(input_format, output_format);

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

    CIO_read_state(filename);
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

  VERBOSE(("\n"));
  VERBOSE(("Autocorrelation test\n"));
  VERBOSE(("Colloid initialised at (%f,%f,%f)\n", r0.x, r0.y, r0.z));
  VERBOSE(("Velocity               (%f,%f,%f)\n", v0.x, v0.y, v0.z));
  p_colloid = COLL_add_colloid(1, a0, ah, r0, v0, omega0);

  p_colloid->stats.x = v0.x;
  p_colloid->dir.x = 1.0;
  p_colloid->dir.y = 0.0;
  p_colloid->dir.z = 0.0;

#endif

#ifdef _COLLOIDS_TEST_OF_OPPORTUNITY_

  Colloid * tmp;

  FVector r0, v0, omega0;
  double    a0 = 0.1171875;
  double    ah = 1.546;

  set_N_colloid(1);

  r0.x = Lmin(X) + L(X) - 8.0;
  r0.y = Lmin(Y) + 8.0;
  r0.z = 0.5*L(Z);

  v0.x = 0.0;
  v0.y = 0.0;
  v0.z = 0.0;

  omega0.x = 0.0;
  omega0.y = 0.0;
  omega0.z = 0.0;

  tmp = COLL_add_colloid_no_halo(1, a0, ah, r0, v0, omega0);
  if (tmp == NULL) verbose("*** not added\n");
  info("Starting test of opportunity\n");
#endif

#ifdef _EWALD_TEST_
  /* ewald_test();*/
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
	    p_colloid->r.x, p_colloid->v.x/p_colloid->stats.x);
	  /*verbose("Autocorrelation omega: %10.9f %10.9f %10.9f\n",
		  p_colloid->omega.z/p_colloid->stats.x, p_colloid->dir.x,
		  p_colloid->dir.y);*/
#endif
#ifdef _COLLOIDS_TEST_OF_OPPORTUNITY_
	  verbose("Position: %g %g %g %g %g %g\n",
	       p_colloid->r.x, p_colloid->r.y, p_colloid->r.z,
	       p_colloid->stats.x, p_colloid->stats.y, p_colloid->stats.z);
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
 *  COLL_add_colloid_no_halo
 *
 *  Add a colloid only if the proposed position is in the domain
 *  proper (and not in the halo).
 *
 *****************************************************************************/

Colloid * COLL_add_colloid_no_halo(int index, double a0, double ah, FVector r0,
			      FVector v0, FVector omega0) {

  IVector cell;
  Colloid * p_c = NULL;

  cell = cell_coords(r0);
  if (cell.x < 1 || cell.x > Ncell(X)) return p_c;
  if (cell.y < 1 || cell.y > Ncell(Y)) return p_c;
  if (cell.z < 1 || cell.z > Ncell(Z)) return p_c;

  p_c = COLL_add_colloid(index, a0, ah, r0, v0, omega0);

  return p_c;
}


/*****************************************************************************
 *
 *  COLL_add_colloid
 *
 *  Add a colloid with the given properties to the head of the
 *  appropriate cell list.
 *
 *  Important: it is up to the caller to ensure index is correct
 *             i.e., it's unique.
 *
 *  A pointer to the new colloid is returned to allow further
 *  modification of the structure. But it's already added to
 *  the cell list.
 *
 *****************************************************************************/

Colloid * COLL_add_colloid(int index, double a0, double ah, FVector r, FVector u,
			   FVector omega) {

  Colloid * tmp;
  IVector   cell;
  int       n;

  /* Don't add to no-existant cells! */
  n = 0;
  cell = cell_coords(r);
  if (cell.x < 0 || cell.x > Ncell(X) + 1) n++;
  if (cell.y < 0 || cell.y > Ncell(Y) + 1) n++;
  if (cell.z < 0 || cell.z > Ncell(Z) + 1) n++;

  if (n) {
    verbose("Cell coords: %d %d %d position %g %g %g\n",
	    cell.x, cell.y, cell.z, r.x, r.y, r.z);
    fatal("Trying to add colloid to no-existant cell [index %d]\n", index);
  }

  tmp = allocate_colloid();

  /* Put the new colloid at the head of the appropriate cell list */

  tmp->index   = index;
  tmp->a0      = a0;
  tmp->ah      = ah;
  tmp->r.x     = r.x;
  tmp->r.y     = r.y;
  tmp->r.z     = r.z;
  tmp->v.x     = u.x;
  tmp->v.y     = u.y;
  tmp->v.z     = u.z;
  tmp->omega.x = omega.x;
  tmp->omega.y = omega.y;
  tmp->omega.z = omega.z;

  ran_parallel_unit_vector(tmp->s);

  tmp->dr[X] = 0.0;
  tmp->dr[Y] = 0.0;
  tmp->dr[Z] = 0.0;

  tmp->t0      = UTIL_fvector_zero();
  tmp->f0      = UTIL_fvector_zero();
  tmp->force   = UTIL_fvector_zero();
  tmp->torque  = UTIL_fvector_zero();
  tmp->cbar    = UTIL_fvector_zero();
  tmp->rxcbar  = UTIL_fvector_zero();

  /* Record the initial position */
  tmp->stats.x = tmp->r.x;
  tmp->stats.y = tmp->r.y;
  tmp->stats.z = tmp->r.z;

  tmp->deltam   = 0.0;
  tmp->deltaphi = 0.0;
  tmp->sumw     = 0.0;

  for (n = 0; n < 21; n++) {
    tmp->zeta[n] = 0.0;
  }

  tmp->lnk = NULL;
  tmp->rebuild = 1;

  /* Add to the cell list */

  cell_insert_colloid(tmp);

  return tmp;
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
  double hmin;

  if (nc > 0) {

    COLL_zero_forces();
    hmin = COLL_interactions();
    COLL_set_fluid_gravity();
    ewald_sum();

    if (is_statistics_step()) {

      double ereal, efour, eself;
      double rnkt = 1.0/(nc*get_kT());

      /* Note Fourier space and self energy available on all processes */
      ewald_total_energy(&ereal, &efour, &eself);
#ifdef _MPI_
      {
	double hlocal = hmin;
	double elocal[2], e[2];

	elocal[0] = ereal;
	elocal[1] = epotential_;
	MPI_Reduce(&hlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(elocal, e, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	ereal = e[0];
	epotential_ = e[1];
      }
#endif

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
  extern double MISC_fluid_volume(void); /* Move me */
  extern double siteforce[3];

  volume = MISC_fluid_volume();
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
#ifdef _EWALD_TEST_
		    /* Old soft sphere (temporary test) */
		    fmod = 0.0;
		    if (h < 0.25) {
		      fmod = 0.0002*(h + p_c1->ah + p_c2->ah)*(pow(h,-2) - 16.0);
		    }
#else
		    fmod = soft_sphere_force(h);
		    fmod += yukawa_force(h + p_c1->ah + p_c2->ah);
#endif
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

/****************************************************************************
 *
 *  COLL_compute_phi_gradients
 *
 *  Compute gradients of the order parameter phi (\nabla\phi and
 *  \nabla^2\phi) at fluid nodes only (not halo sites).
 *
 *  Neutral wetting properties are always assumed at the moment.
 *
 *  This takes account of the presence of local solid sites
 *  and uses the full 26 (NGRAD) neighbours. This function uses
 *  exactly the same method as the original gradient method
 *  in BS_fix_gradients but doesn't require boundary sites and
 *  is designed to be more efficient for high solid fractions.
 *
 ****************************************************************************/

void COLL_compute_phi_gradients() {

  int     i, j, k, index, indexn;
  int     p;
  int     xfac, yfac;

  int     isite[NGRAD];

  double   r9  = 1.0/9.0;
  double   r18 = 1.0/18.0;
  double   dphi, phi_b, delsq;
  double   gradt[NGRAD];

  FVector gradn;

  IVector count;
  int     N[3];
  double  rk = 1.0 / free_energy_K();

  extern double * phi_site;
  extern double * delsq_phi;
  extern FVector * grad_phi;

  get_N_local(N);

  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  for (i = 1; i <= N[X]; i++)
    for (j = 1; j <= N[Y]; j++)
      for (k = 1; k <= N[Z]; k++) {

	index = i*xfac + j*yfac + k;

	/* Skip solid sites */
	if (site_map[index] != FLUID) continue;

	/* Set solid/fluid flag to index neighbours */

	for (p = 1; p < NGRAD; p++) {
	  indexn = index + xfac*bs_cv[p][0] + yfac*bs_cv[p][1] + bs_cv[p][2];
	  isite[p] = -1;
	  if (site_map[indexn] == FLUID) isite[p] = indexn;
	}


	/* Estimate gradients between fluid sites */

	gradn.x = 0.0;
	gradn.y = 0.0;
	gradn.z = 0.0;
	count.x = 0;
	count.y = 0;
	count.z = 0;

	for (p = 1; p < NGRAD; p++) {

	  if (isite[p] == -1) continue;

	  dphi = phi_site[isite[p]] - phi_site[index];
	  gradn.x += bs_cv[p][0]*dphi;
	  gradn.y += bs_cv[p][1]*dphi;
	  gradn.z += bs_cv[p][2]*dphi;

	  gradt[p] = dphi;

	  count.x += bs_cv[p][0]*bs_cv[p][0];
	  count.y += bs_cv[p][1]*bs_cv[p][1];
	  count.z += bs_cv[p][2]*bs_cv[p][2];
	}

	if (count.x) gradn.x /= (double) count.x;
	if (count.y) gradn.y /= (double) count.y;
	if (count.z) gradn.z /= (double) count.z;

	/* Estimate gradient at boundaries */

	for (p = 1; p < NGRAD; p++) {

	  if (isite[p] == -1) {

	    phi_b = phi_site[index] + 0.5*
	    (bs_cv[p][0]*gradn.x + bs_cv[p][1]*gradn.y + bs_cv[p][2]*gradn.z);

	    /* Set gradient of phi at boundary following wetting properties */
	    /* C and H are always zero at the moment */

	    gradt[p] = -(0.0*phi_b - 0.0)*rk;
	  }
	}

	/* Accumulate the final gradients */

	delsq = 0.0;
	gradn.x = 0.0;
	gradn.y = 0.0;
	gradn.z = 0.0;

	for (p = 1; p < NGRAD; p++) {
	  delsq   += gradt[p];
	  gradn.x += gradt[p]*bs_cv[p][0];
	  gradn.y += gradt[p]*bs_cv[p][1];
	  gradn.z += gradt[p]*bs_cv[p][2];
	}

	delsq   *= r9;
	gradn.x *= r18;
	gradn.y *= r18;
	gradn.z *= r18;

	delsq_phi[index] = delsq;
	grad_phi[index]  = gradn;

      }

  return;
}


/*****************************************************************************
 *
 *  COLL_compute_phi_missing
 *
 *  Extrapolate (actually average) phi values to sites inside
 *  solid particles. This is done by looking at nearby sites
 *  (connected via a basis vector).
 *
 *  This has no physical meaning; it is used to avoid rubbish
 *  values in the phi field for visualisation purposes.
 *
 ****************************************************************************/

void COLL_compute_phi_missing() {

  int     i, j , k, index, indexn, p;
  int     count;
  int     xfac, yfac;
  int     N[3];

  double   phibar;

  extern double * phi_site;

  get_N_local(N);

  yfac = (N[Z] + 2);
  xfac = (N[Y] + 2)*yfac;


   for (i = 1; i <= N[X]; i++)
    for (j = 1; j <= N[Y]; j++)
      for (k = 1; k <= N[Z]; k++) {

	index = i*xfac + j*yfac + k;

	if (site_map[index] != FLUID) {

	  /* Look at the neigbours and take the average */
	  count = 0;
	  phibar = 0.0;

	  for (p = 1; p < NVEL; p++) {
	    indexn = index + xfac*cv[p][0] + yfac*cv[p][1] + cv[p][2];
	    if (site_map[indexn] == FLUID) {
	      count += 1;
	      phibar += phi_site[indexn];
	    }
	  }

	  if (count > 0)
	    phi_site[index] = phibar / (double) count;
	}
      }

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
  double vmax = 0.0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	  while (p_colloid) {

	    vmax = dmax(vmax, UTIL_dot_product(p_colloid->v, p_colloid->v));

	    p_colloid = p_colloid->next;
	  }
      }
    }
  }

#ifdef _MPI_
 {
   double vmax_local = vmax;
   MPI_Reduce(&vmax_local, &vmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 }
#endif

  return sqrt(vmax);
}

/*****************************************************************************
 *
 *  coll_count
 *
 *****************************************************************************/

int coll_count() {

  int       ic, jc, kc;
  Colloid * p_colloid;

  int nlocal = 0, ntotal = 0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {
	  nlocal++;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  ntotal = nlocal;

#ifdef _MPI_
  MPI_Reduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, 0, cart_comm());
#endif

  return ntotal;
}
