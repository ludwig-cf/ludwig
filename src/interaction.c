/*****************************************************************************
 *
 *  interaction.c
 *
 *  Colloid potentials and colloid-colloid interactions.
 *
 *  Refactoring is in progress.
 *
 *  $Id: interaction.c,v 1.5 2006-10-12 14:09:18 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "active.h"
#include "build.h"

#include "colloids.h"
#include "interaction.h"
#include "communicate.h"
#include "model.h"
#include "lattice.h"
#include "collision.h"
#include "cio.h"
#include "control.h"

#include "ccomms.h"

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "runtime.h"
#include "free_energy.h"

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
static FVector COLL_lubrication(Colloid *, Colloid *, FVector, Float);
static void    COLL_init_colloids_test(void);
static void    COLL_test_output(void);
static void    COLL_update_colloids(void);

/* Old Global_Colloid stuff */

FVector   F;             /* Force on all colloids, e.g., gravity */

struct lubrication_struct {
  int corrections_on;
  double cutoff_norm;  /* Normal */
  double cutoff_tang;  /* Tangential */
} lubrication;

static struct soft_sphere_potential_struct {
  int on;
  double epsilon;
  double sigma;
  double nu;
  double cutoff;
} soft_sphere;


/*
struct leonard_jones_potential {
  int leonard_jones_on;
  double sigma;
  double epsilon;
  double cutoff;
}
*/

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

  /* Removal or replacement of fluid requires a lattice halo update */
  COM_halo();

  TIMER_start(TIMER_PARTICLE_HALO);

  cell_update();
  CCOM_halo_particles();

  TIMER_stop(TIMER_PARTICLE_HALO);

  TIMER_start(TIMER_REBUILD);

#ifndef _SUBGRID_
  COLL_update_map();
  COLL_remove_or_replace_fluid();
  COLL_update_links();
#endif /* _SUBGRID_ */

  TIMER_stop(TIMER_REBUILD);

  COLL_test_output();
  COLL_forces();
  active_bbl_prepass();

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
  double ahmax, f0[3];

  void CMD_init_volume_fraction(int, int);
  void lubrication_init(void);
  void soft_sphere_init(void);
  void check_interactions(const double);

  /* Default position: no colloids */

  RUN_get_string_parameter("colloid_init", tmp);
  if (strcmp(tmp, "no_colloids") == 0) nc = 0;

  /* This is just to get past the start. */
  if (strcmp(tmp, "fixed_volume_fraction_monodisperse") == 0) nc = 1;
  if (strcmp(tmp, "fixed_number_monodisperse") == 0) nc = 1;

#ifdef _COLLOIDS_TEST_AUTOCORRELATION_
  nc = 1;
#endif

  if (nc == 0) return;

  nc = RUN_get_double_parameter("colloid_ah", &ahmax);
  if (nc == 0) fatal("Please set colloids_ah in the input file\n");

  nc = RUN_get_double_parameter_vector("colloid_gravity", f0);
  if (nc != 0) {
    F.x = f0[X];
    F.y = f0[Y];
    F.z = f0[Z];
  }

  /* Initialisation section. */

  colloids_init();

  COLL_init_coordinates();
  CCOM_init_halos();
  CMPI_init_messages();
  CIO_set_cio_format(input_format, output_format);
  lubrication_init();
  soft_sphere_init();
  check_interactions(ahmax);

  if (get_step() == 0) {

#ifdef _COLLOIDS_TEST_
    COLL_init_colloids_test();
#else
    CMD_init_volume_fraction(1, 0);
    init_active();
#endif
  }
  else {
    /* Restart from previous configuration */
    sprintf(filename, "%s%6.6d", "config.cds", get_step());
    CIO_read_state(filename);
  }

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

  FVector r0, v0, omega0;
  double    a0 = 1.25;
  double    ah = 1.09;

  set_N_colloid(1);

  r0.x = 0.5*L(X);
  r0.y = 0.5*L(Y);
  r0.z = Lmin(Z) + L(Z) - a0;

  v0.x = 0.0;
  v0.y = 0.0;
  v0.z = 0.0;

  omega0.x = 0.0;
  omega0.y = 0.0;
  omega0.z = 0.0;

  COLL_add_colloid(1, a0, ah, r0, v0, omega0);

  r0.z = r0.z - 2.0*a0; 
  v0.x = v0.x;

  /*COLL_add_colloid(2, a0, ah, r0, v0, omega0);*/

  info("Starting test of opportunity\n");
#endif

#ifdef _COLLOIDS_TEST_CALIBRATE_

  N_colloid = 1;

  r0.x = 0.5*L(X) + RAN_uniform();
  r0.y = 0.5*L(Y) + RAN_uniform();
  r0.z = 0.5*L(Z) + RAN_uniform();

  /* For particle Reynolds number = 0.05 the velocity we want
   * scales as Re * viscosity / a. This also applies to the
   * external force (to within a minus sign). */

  tmp = 0.05*get_eta_shear()/a0;

  v0.x = -RAN_uniform()*tmp;
  v0.y = -RAN_uniform()*tmp;
  v0.z = -RAN_uniform()*tmp;

  COLL_init_gravity(v0);

  v0 = UTIL_fvector_zero();

  omega0.x = 0.0;
  omega0.y = 0.0;
  omega0.z = 0.0;

  COLL_add_colloid(1, a0, ah, r0, v0, omega0);

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
	  verbose("Proximity test output: %f, %f, %f\n",
		  p_colloid->cbar.x, p_colloid->cbar.y, p_colloid->cbar.z);
	  verbose("Calibrate: %g %g %g %g %g %g\n",
		  p_colloid->v.x, p_colloid->v.y, p_colloid->v.z,
		  p_colloid->r.x, p_colloid->r.y, p_colloid->r.z);
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

void COLL_add_colloid_no_halo(int index, double a0, double ah, FVector r0,
			      FVector v0, FVector omega0) {

  IVector cell;

  cell = cell_coords(r0);
  if (cell.x < 1 || cell.x > Ncell(X)) return;
  if (cell.y < 1 || cell.y > Ncell(Y)) return;
  if (cell.z < 1 || cell.z > Ncell(Z)) return;

  COLL_add_colloid(index, a0, ah, r0, v0, omega0);

  return;
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

  double hmin;

  COLL_zero_forces();
  hmin = COLL_interactions();

  if (is_statistics_step() && get_N_colloid() > 1) {
#ifdef _MPI_
    {
      double hlocal = hmin;
      MPI_Reduce(&hmin, &hlocal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
#endif
    info("[Inter-particle gap minimum is: %f]\n", hmin);
  }

  COLL_set_fluid_gravity();
  COLL_set_colloid_gravity(); /* This currently zeroes halo particles */

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
 *  COLL_set_colloid_gravity
 *
 *  Add the gravtitation force to the particles.
 *
 *  This is the last force addition, so also zero the
 *  external force in the halo regions.
 *
 *****************************************************************************/

void COLL_set_colloid_gravity() {

  int       ic, jc, kc;
  int       ihalo;
  Colloid * p_colloid;

  /* Check all the particles */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  ihalo = (ic == 0 || jc == 0 || kc == 0 || ic == Ncell(X) + 1 ||
		   jc == Ncell(Y) + 1  || kc == Ncell(Z) + 1);

	  if (ihalo) {
	    p_colloid->force  = UTIL_fvector_zero();
	    p_colloid->torque = UTIL_fvector_zero();
	  }
	  else {
	    p_colloid->force  = UTIL_fvector_add(p_colloid->force, F);
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
 *  Issues
 *
 *****************************************************************************/

void COLL_set_fluid_gravity() {

  double volume;
  extern double MISC_fluid_volume(void); /* Move me */
  extern double siteforce[3];

  volume = MISC_fluid_volume();

  /* Size of force per fluid node */

  siteforce[X] = -get_N_colloid()*F.x/volume;
  siteforce[Y] = -get_N_colloid()*F.y/volume;
  siteforce[Z] = -get_N_colloid()*F.z/volume;

  return;
}

/****************************************************************************
 *
 *  COLL_interactions
 *
 *  A cell list approach is used to achieve O(N) scaling. For a given
 *  colloid, we should examine the distance to neighbours in its own
 *  cell, and those neighbouring to one side.
 *
 ****************************************************************************/

double COLL_interactions() {

  Colloid * p_c1;
  Colloid * p_c2;

  FVector r_ij;              /* Separation vector joining  c1 -> c2 */
  double   rmod;             /* Modulus thereof */
  double   ra1, ra2;          /* Hydrodynamic radius both colloids */
  double   gap;               /* Gap between the colloids */
  FVector f0;                /* Current interaction force */
  FVector f1;                /* Force on particle 1 */
  FVector fa;
  FVector COLL_fvector_separation(FVector, FVector);

  int     ic, jc, kc, id, jd, kd, nd;

  double   fmod;
  double   mingap = L(X);

  /* One-sided cell list search offsets */
  enum    {NABORS = 14};
  int     di[NABORS] = {0, 1, 1, 0, -1, 0, 1, 1, 0, -1, -1, -1,  0,  1};
  int     dj[NABORS] = {0, 0, 1, 1,  1, 0, 0, 1, 1,  1,  0, -1, -1, -1};
  int     dk[NABORS] = {0, 0, 0, 0,  0, 1, 1, 1, 1,  1,  1,  1,  1,  1};


  /* Note the loops are [0, ncell.x] etc, so that get right
   * interactions. */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {
	
	p_c1 = CELL_get_head_of_list(ic, jc, kc);

	while (p_c1) {

	  /* Total force on particle 1 */
	  f1 = UTIL_fvector_zero();
	  ra1 = p_c1->ah;

	  /* Look at each of the neighbouring cells */

	  for (nd = 0; nd < NABORS; nd++) {

	    kd = kc + dk[nd];
	    jd = jc + dj[nd];
	    id = ic + di[nd];

	    /* Make sure we don't look outside the cell list */
	    if (jd < 0 || id < 0) continue;
	    if (kd > Ncell(Z)+1 || jd > Ncell(Y)+1 ||
		id > Ncell(X)+1) continue;

	    p_c2 = CELL_get_head_of_list(id, jd, kd);

	    /* Interaction. If two cells are the same, make sure
	     * that we don't double count by looking ahead only. */

	    if (kc == kd && jc == jd && ic == id) p_c2 = p_c1->next;

	    /* Look at particles in list 2 */

	    while (p_c2) {
	      ra2 = p_c2->ah;

	      /* Compute the separation vector and the gap, and the
	       * unit vector joining the two centres */
	      r_ij = COLL_fvector_separation(p_c1->r, p_c2->r);
	      rmod  = UTIL_fvector_mod(r_ij);

	      gap = rmod - ra1 - ra2;
	      if (gap < mingap) mingap = gap;
	      if (gap < 0.0) COLL_overlap(p_c1, p_c2);

	      /* Lubrication corrections, soft sphere potential ... */
	      f0 = COLL_lubrication(p_c1, p_c2, r_ij, rmod);

	      fmod = soft_sphere_force(gap);
	      fa.x = -fmod*r_ij.x/rmod;
	      fa.y = -fmod*r_ij.y/rmod;
	      fa.z = -fmod*r_ij.z/rmod;

	      /* Total for this interaction */
	      f0 = UTIL_fvector_add(f0, fa);

	      /* Add force for this interaction to particle 1, and
	       * also accumulate it for particle 2 (with a - sign) */
	      f1 = UTIL_fvector_add(f1, f0);
	      p_c2->force = UTIL_fvector_subtract(p_c2->force, f0);

	      /* Next colloid 2 */
	      p_c2 = p_c2->next;
	    }
	    /* Next cell 2 */
	  }

	  /* Add total force on particle 1 */
	  p_c1->force = UTIL_fvector_add(p_c1->force, f1);

	  /* Next colloid 1 */
	  p_c1 = p_c1->next;
	}
	/* Next cell 1 */
      }
    }
  }

  return mingap;
}

/*****************************************************************************
 *
 *  soft_sphere_init
 *
 *  Initialise the parameters for the soft-sphere interaction between
 *  colloids.
 *
 *****************************************************************************/

void soft_sphere_init() {

  int n;

  info("\nColloid-colloid soft-sphere potential\n");

  n = RUN_get_int_parameter("soft_sphere_on", &soft_sphere.on);
  info((n == 0) ? "[Default] " : "[User   ] ");
  info("Soft sphere potential is switched %s\n",
       (soft_sphere.on == 0) ? "off" : "on");

  if (soft_sphere.on) {
    n = RUN_get_double_parameter("soft_sphere_epsilon", &soft_sphere.epsilon);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Soft-sphere energy (epsilon) is %f (%f kT)\n", soft_sphere.epsilon,
	 soft_sphere.epsilon/get_kT());

    n = RUN_get_double_parameter("soft_sphere_sigma", &soft_sphere.sigma);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Soft-sphere width (sigma) is %f\n", soft_sphere.sigma);

    n = RUN_get_double_parameter("soft_sphere_nu", &soft_sphere.nu);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Soft-sphere exponent (nu) is %f\n", soft_sphere.nu);
    if (soft_sphere.nu <= 0.0) fatal("Please check nu is positive\n");

    n = RUN_get_double_parameter("soft_sphere_cutoff", &soft_sphere.cutoff);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Soft-sphere cutoff range is %f\n", soft_sphere.cutoff);
  }

  return;
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

  rc = 2.0*ahmax + soft_sphere.cutoff;
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
 *  soft_sphere_energy
 *
 *  Return the energy of interaction between two particles with
 *  (surface-surface) separation h.
 *
 *****************************************************************************/

double soft_sphere_energy(const double h) {

  double e = 0.0;

  if (soft_sphere.on) {
    double hc = soft_sphere.cutoff;
    double nu = soft_sphere.nu;

    if (h > 0 && h < hc) {
      e = pow(h, -nu) - pow(hc, -nu)*(1.0 - (h-hc)*nu/hc);
      e = e*soft_sphere.epsilon*pow(soft_sphere.sigma, nu);
    }
  }

  return e;
}

/*****************************************************************************
 *
 *  soft_sphere_force
 *
 *  Return the magnitude of the 'soft-sphere' interaction force between
 *  two particles with (surface-surface) separation h.
 *
 ****************************************************************************/

double soft_sphere_force(const double h) {

  double f = 0.0;

  if (soft_sphere.on) {
    double hc = soft_sphere.cutoff;
    double nu = soft_sphere.nu;

    if (h > 0 && h < hc) {
      f = pow(h, -(nu+1)) - pow(hc, -(nu+1));
      f = f*soft_sphere.epsilon*pow(soft_sphere.sigma, nu)*nu;
    }
  }

  return f;
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

  Float   r9  = 1.0/9.0;
  Float   r18 = 1.0/18.0;
  Float   dphi, phi_b, delsq;
  Float   gradt[NGRAD];

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

	if (count.x) gradn.x /= (Float) count.x;
	if (count.y) gradn.y /= (Float) count.y;
	if (count.z) gradn.z /= (Float) count.z;

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

  Float   phibar;

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
	    phi_site[index] = phibar / (Float) count;
	}
      }

  return;
}

