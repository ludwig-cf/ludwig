/*****************************************************************************
 *
 *  colloids.c
 *
 *  General driver routines for particles.
 *
 *  A full description of the implementation of particles is given in
 *  the Ludwig Technical Notes [ludwig/doc/ludwig.tex].
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk).
 *
 *****************************************************************************/

#include "globals.h"
#include "ccomms.h"
#include "cells.h"
#include "cmem.h"
#include "cio.h"

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "control.h"

const Float c2rcs2 = 2.0*3.0; /* The constant 2.0 / c_s^2  MOVEME! */

Colloid ** coll_map;        /* Colloid map. A pointer to a colloid at
			     * each lattice node. */
Colloid ** coll_old;        /* Map at the previous time step */

static void    COLL_overlap(Colloid *, Colloid *);
static void    COLL_set_fluid_gravity(void);
static FVector COLL_lubrication(Colloid *, Colloid *, FVector, Float);
static FVector COLL_soft_sphere(Colloid *, Colloid *, FVector, Float);

static void    COLL_init_colloids_test(void);
static void    COLL_init_coordinates(void);
static void    COLL_set_gravity(void);
static void    COLL_test_output(void);
static void    COLL_update_colloids(void);

static void    COLL_init_colloids_lattice(int, int, int);


/*****************************************************************************
 *
 *  COLL_init
 * 
 *  Driver routine for colloid initialisation.
 *
 *  All colloid routines are called after COM_init(), so run
 *  time input is available. A few remaining values in the
 *  global structure are initialised here.
 *
 *****************************************************************************/

void COLL_init() {

#ifdef _COLLOIDS_

  char filename[FILENAME_MAX];
  void CMD_init_volume_fraction(int, int);

  Global_Colloid.N_colloid  = 0;
  Global_Colloid.fr         = 1;
  Global_Colloid.rho        = 1.0;
  Global_Colloid.deltaf     = 0.0;
  Global_Colloid.deltag     = 0.0;

  CMPI_init_global();
  COLL_init_coordinates();
  CELL_init_cells();
  CCOM_init_halos();
  CMPI_init_messages();
  CIO_set_cio_format(gbl.input_format, gbl.output_format);

  if (get_step() == 0) {

#ifdef _COLLOIDS_TEST_
    COLL_init_colloids_test();
#else
    CMD_init_volume_fraction(1, 0);
#endif

  }
  else {
    /* Restart from previous configuration */
    sprintf(filename, "%s%6.6d", "config.cds", get_step());
    CIO_read_state(filename);
  }

  /* Transfer any particles in the halo regions, initialise the
   * colloid map and build the particles for the first time. */

  CMPI_count_colloids();
  CCOM_halo_particles(); 
  CCOM_sort_halo_lists();
  COLL_update_map();
  COLL_update_links();

#endif

  return;
}


/*****************************************************************************
 *
 *  COLL_init_coordinates
 *
 *  _L, _Lmin, _Lmax describe the physical system size.
 *
 *****************************************************************************/

void COLL_init_coordinates() {

  int n;
  int N[3];

  void get_N_local(int []);

  /* Allocate space for the local colloid map */

  get_N_local(N);
  n = (N[X] + 2)*(N[Y] + 2)*(N[Z] + 2);

  info("Requesting %d bytes for colloid maps\n", 2*n*sizeof(Colloid*));

  coll_map = (Colloid **) malloc(n*sizeof(Colloid *));
  coll_old = (Colloid **) malloc(n*sizeof(Colloid *));

  if (coll_map == (Colloid **) NULL) fatal("malloc (coll_map) failed");
  if (coll_old == (Colloid **) NULL) fatal("malloc (coll_old) failed");

  return;
}


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

  TIMER_start(TIMER_PARTICLE_HALO);

  CELL_update_cell_lists();
  CCOM_halo_particles();

  TIMER_stop(TIMER_PARTICLE_HALO);

  TIMER_start(TIMER_REBUILD);
  CCOM_sort_halo_lists();

  COLL_update_map();
  COLL_remove_or_replace_fluid();
  COLL_update_links();

  TIMER_stop(TIMER_REBUILD);

#ifdef _COLLOIDS_TEST_
  COLL_test_output();
#endif

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

#ifdef _COLLOIDS_

  info("\nCOLL_finish releasing memory ...\n");
  CMEM_free_all_colloids();
  CMEM_report_memory();

  if (coll_map) free(coll_map);
  if (coll_old) free(coll_old);

#endif

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
  Float   a0, ah;
  Colloid * p_colloid;

  a0 = Global_Colloid.a0;
  ah = Global_Colloid.ah;

  /* Autocorrelation test. System size should be 40x40x40.
   * For rotational part, swap values of v0 and omega0.
   * Magnitude of the velocity typically 0.01; there may be small
   * differences in the value of the ACF if different initial
   * values are used. */

  Global_Colloid.N_colloid = 1;

  r0.x =  .0 + 0.5*L(X);
  r0.y =  .0 + 0.5*L(Y);
  r0.z =  .0 + 0.5*L(Z);

  v0.x = 1.0*get_eta_shear()/ah;
  v0.y = 0.0;
  v0.z = 0.0;

  omega0.x = 0.0;
  omega0.y = 0.0;
  omega0.z = 0.0*get_eta_shear()/(ah*ah);

  /* Look at the proposed position and decide whether it is in
   * the local domain. If it is, then it can be added. */

  cell = CELL_cell_coords(r0);
  if (cell.x < 1 || cell.x > Global_Colloid.Ncell.x) return;
  if (cell.y < 1 || cell.y > Global_Colloid.Ncell.y) return;
  if (cell.z < 1 || cell.z > Global_Colloid.Ncell.z) return;

  VERBOSE(("\n"));
  VERBOSE(("Autocorrelation test\n"));
  VERBOSE(("Colloid initialised at (%f,%f,%f)\n", r0.x, r0.y, r0.z));
  VERBOSE(("Velocity               (%f,%f,%f)\n", v0.x, v0.y, v0.z));
  p_colloid = COLL_add_colloid(1, a0, ah, r0, v0, omega0);

  p_colloid->stats.x = v0.x;
  p_colloid->dir.x = 1.0;
  p_colloid->dir.y = 0.0;
  p_colloid->dir.z = 0.0;

  Global_Colloid.nlocal = 1;

#endif

#ifdef _COLLOIDS_TEST_OF_OPPORTUNITY_

  FVector r0, v0, omega0;
  Float   a0, ah;

  COLL_init_colloids_lattice(4, 4, 4);


  a0 = Global_Colloid.a0;
  ah = Global_Colloid.ah;

  Global_Colloid.N_colloid = 2;

  r0.x = 0.5*L(X);
  r0.y = 0.0*L(Y) + _Lmin.y + RAN_uniform();
  r0.z = 0.0*L(Z) + _Lmin.z + RAN_uniform();

  v0.x = 0.1*get_eta_shear()/ah;
  v0.y = 0.0;
  v0.z = 0.0;

  omega0.x = 0.0;
  omega0.y = 0.0;
  omega0.z = 0.0;

  v0.x = 0.01*get_eta_shear()/ah;
  r0.x = 7.352999440561971;
  r0.y = 7.954383520227735;
  r0.z = 8.652245545163171;

  COLL_add_colloid_no_halo(1, a0, ah, r0, v0, omega0);

  v0.x = -0.01*get_eta_shear()/ah;
  r0.x = 11.367262769237062;
  r0.y = 10.313033605508135;
  r0.z = 10.090403708490734;

  COLL_add_colloid_no_halo(2, a0, ah, r0, v0, omega0);


  /*
  r0.x -= 23.0;
  COLL_add_colloid_no_halo(3, a0, ah, r0, v0, omega0);

  r0.y -= 2.7*2.0;
  COLL_add_colloid_no_halo(4, a0, ah, r0, v0, omega0);
  */
#endif

#if defined _COLLOIDS_TEST_COLLISION_

  /* Collision test to check the extra lubrication forces
   * at small separation. If the correct implicit velocity
   * update is used there should be instability or overlap. */

  /* Start with two (or more) colloids at rest. An additional
   * attractive force is present between them so
   * they move together. */

  /* Ther system size should be 10x10x10 and run for 400 steps. */

  Global_Colloid.N_colloid = 2;

  r0.x = 3.0;
  r0.y = 3.;
  r0.z = 3.;

  v0.x = 0.0;
  v0.y = 0.0;
  v0.z = 0.0;

  omega0.x = 0.0;
  omega0.y = 0.0;
  omega0.z = 0.0;

  COLL_add_colloid(1, a0, ah, r0, v0, omega0);

  r0.x = r0.x + 5.0;
  r0.y = r0.y + 0.0;
  r0.z = r0.z + 0.0;

  COLL_add_colloid(2, a0, ah, r0, v0, omega0);

#endif

#ifdef _COLLOIDS_TEST_CALIBRATE_

  Global_Colloid.N_colloid = 1;

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
  IVector   ncell;

  ncell = Global_Colloid.Ncell;

  for (ic = 1; ic <= ncell.x; ic++) {
    for (jc = 1; jc <= ncell.y; jc++) {
      for (kc = 1; kc <= ncell.z; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {
#ifdef _COLLOIDS_TEST_AUTOCORRELATION_
	  verbose("Autocorrelation test output: %10.9f %10.9f\n",
	    p_colloid->r.x, p_colloid->v.x/p_colloid->stats.x);
	  /*verbose("Autocorrelation omega: %10.9f %10.9f %10.9f\n",
		  p_colloid->omega.z/p_colloid->stats.x, p_colloid->dir.x,
		  p_colloid->dir.y);*/
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

void COLL_add_colloid_no_halo(int index, Float a0, Float ah, FVector r0,
			      FVector v0, FVector omega0) {

  IVector cell;

  cell = CELL_cell_coords(r0);
  if (cell.x < 1 || cell.x > Global_Colloid.Ncell.x) return;
  if (cell.y < 1 || cell.y > Global_Colloid.Ncell.y) return;
  if (cell.z < 1 || cell.z > Global_Colloid.Ncell.z) return;

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

Colloid * COLL_add_colloid(int index, Float a0, Float ah, FVector r, FVector u,
			   FVector omega) {

  Colloid * tmp;
  IVector   cell;
  int       n;

  /* Don't add to no-existant cells! */
  n = 0;
  cell = CELL_cell_coords(r);
  if (cell.x < 0 || cell.x > Global_Colloid.Ncell.x + 1) n++;
  if (cell.y < 0 || cell.y > Global_Colloid.Ncell.y + 1) n++;
  if (cell.z < 0 || cell.z > Global_Colloid.Ncell.z + 1) n++;

  if (n) {
    fatal("Trying to add colloid to no-existant cell [index %d]\n", index);
  }

  tmp = CMEM_allocate_colloid();

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
  tmp->stats   = UTIL_fvector_zero();

#ifdef _COLLOIDS_TEST_TEMP_
  tmp->stats.x = tmp->r.x;
  tmp->stats.y = tmp->r.y;
  tmp->stats.z = tmp->r.z;
#endif

  tmp->deltam   = 0.0;
  tmp->deltaphi = 0.0;
  tmp->sumw     = 0.0;

  for (n = 0; n < 21; n++) {
    tmp->zeta[n] = 0.0;
  }

  tmp->lnk = NULL;
  tmp->rebuild = 1;
  tmp->export = 1;

  /* Add to the cell list */

  CELL_insert_at_head_of_list(tmp);

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

  TIMER_start(TIMER_FORCES);

  COLL_zero_forces();
  hmin = COLL_interactions();
  info("[Inter-particle gap minimum is: %f]\n", hmin);
  COLL_set_fluid_gravity();
  COLL_set_colloid_gravity(); /* This currently zeroes halo particles */

  TIMER_stop(TIMER_FORCES);

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
  IVector   ncell;
  Colloid * p_colloid;

  ncell = Global_Colloid.Ncell;

  /* Only real particles need correct external force. */

  for (ic = 0; ic <= ncell.x + 1; ic++) {
    for (jc = 0; jc <= ncell.y + 1; jc++) {
      for (kc = 0; kc <= ncell.z + 1; kc++) {

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
  IVector   ncell;
  Colloid * p_colloid;

  ncell = Global_Colloid.Ncell;

  /* Check all the particles */

  for (ic = 0; ic <= ncell.x + 1; ic++) {
    for (jc = 0; jc <= ncell.y + 1; jc++) {
      for (kc = 0; kc <= ncell.z + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  ihalo = (ic == 0 || jc == 0 || kc == 0 || ic == ncell.x + 1 ||
		   jc == ncell.y + 1  || kc == ncell.z + 1);

	  if (ihalo) {
	    p_colloid->force  = UTIL_fvector_zero();
	    p_colloid->torque = UTIL_fvector_zero();
	  }
	  else {
	    p_colloid->force  = UTIL_fvector_add(p_colloid->force,
						 Global_Colloid.F);
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

  siteforce[X] = -Global_Colloid.N_colloid*Global_Colloid.F.x/volume;
  siteforce[Y] = -Global_Colloid.N_colloid*Global_Colloid.F.y/volume;
  siteforce[Z] = -Global_Colloid.N_colloid*Global_Colloid.F.z/volume;

  return;
}


/****************************************************************************
 *
 *  COLL_interactions
 *
 *  The lubrication force between pairs of colloids closer than the
 *  cut-off distance is added explicitly following Ladd and Verberg
 *  [J Stat. Phys. {\bf 104}, 1191, 2001].
 *
 *  This is always an explicit treatment. It is therefore essential
 *  that the user ensures that colloids do not approach closer than
 *  the distance where implicit treatment is required. This is
 *  typically done using an extra repulsive "drop in" potential.
 *
 *  A cell list approach is used to achieve O(N) scaling. For a given
 *  colloid, we should examine the distance to neighbours in its own
 *  cell, and those neighbouring to one side.
 *
 *  Notes:
 *
 ****************************************************************************/

Float COLL_interactions() {

  Colloid * p_c1;
  Colloid * p_c2;

  FVector r_ij;              /* Separation vector joining  c1 -> c2 */
  Float   rmod;              /* Modulus thereof */
  Float   ra1, ra2;          /* Hydrodynamic radius both colloids */
  Float   gap;               /* Gap between the colloids */
  FVector f0;                /* Current interaction force */
  FVector f1;                /* Force on particle 1 */
  FVector fa;

  IVector ncell;
  int     cl1, cl2;
  int     ic, jc, kc, id, jd, kd, nd;
  int     cifac, cjfac;

  Float   mingap = L(X);

  /* One-sided cell list search offsets */
  enum    {NABORS = 14};
  int     di[NABORS] = {0, 1, 1, 0, -1, 0, 1, 1, 0, -1, -1, -1,  0,  1};
  int     dj[NABORS] = {0, 0, 1, 1,  1, 0, 0, 1, 1,  1,  0, -1, -1, -1};
  int     dk[NABORS] = {0, 0, 0, 0,  0, 1, 1, 1, 1,  1,  1,  1,  1,  1};

  ncell = Global_Colloid.Ncell;
  cifac = (ncell.y + 2)*(ncell.z + 2);
  cjfac = (ncell.z + 2);

  /* Note the loops are [0, ncell.x] etc, so that get right
   * interactions. */

  for (ic = 0; ic <= ncell.x + 1; ic++) {
    for (jc = 0; jc <= ncell.y + 1; jc++) {
      for (kc = 0; kc <= ncell.z + 1; kc++) {
	
	cl1 = kc + cjfac*jc + cifac*ic;
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
	    if (kd > ncell.z+1 || jd > ncell.y+1 || id > ncell.x+1) continue;

	    cl2 = kd + cjfac*jd + cifac*id;
	    p_c2 = CELL_get_head_of_list(id, jd, kd);

	    /* Interaction. If two cells are the same, make sure
	     * that we don't double count by looking ahead only. */

	    if (cl1 == cl2) p_c2 = p_c1->next;

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
	      fa = COLL_soft_sphere(p_c1, p_c2, r_ij, rmod);

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
 *  COLL_lubrication
 *
 *  Compute the net lubrication correction for the two colloids
 *  which are separated by vector r_ij (the size of which is h
 *  i.e., centre-centre distance).
 *
 *  Corrections for two components are included:
 *     Normal component if h < r_lu_n
 *     Tangential       if h < r_lu_t
 *
 *  If noise is on, then an additional random force is required to
 *  satisfy the fluctuation-dissipation theorem. The size of the
 *  random component depends on the "temperature", and is just
 *  added to the lubrication contribution.
 *
 *****************************************************************************/

FVector COLL_lubrication(Colloid * p_i, Colloid * p_j, FVector r_ij, Float h) {

  FVector force;
  FVector du;
  FVector runit;
  Float   ai, aj;
  Float   rh;
  Float   rdotdu;
  Float   fmod;
  Float   r_lu_n = Global_Colloid.r_lu_n;
  Float   r_lu_t = Global_Colloid.r_lu_t;

  force = UTIL_fvector_zero();

#ifdef _NO_LUBRICATION_CORRECTIONS_
  return force;
#endif

  /* Define the surface-surface separation */

  ai = p_i->ah;
  aj = p_j->ah;
  h  = h - ai - aj;

  if (h < r_lu_n) {
    
    /* Compute the separation unit vector in the direction of r_ij */

    rh = 1.0 / (h + ai + aj);
    runit.x = rh*r_ij.x;
    runit.y = rh*r_ij.y;
    runit.z = rh*r_ij.z;

    /* Normal lubrication correction */

    rh     = 1.0 / h;
    du     = UTIL_fvector_subtract(p_i->v, p_j->v);
    rdotdu = UTIL_dot_product(runit, du);
    fmod   = -6.0*PI*get_eta_shear()*ai*ai*aj*aj*(rh - 1.0/r_lu_n)
      / ((ai+ai)*(aj+aj));

    force.x += fmod*rdotdu*runit.x;
    force.y += fmod*rdotdu*runit.y;
    force.z += fmod*rdotdu*runit.z;

    /* Tangential lubrication correction */
    if (h < r_lu_t) {
      rh = 0.5*(ai+aj)/h;
      fmod = -(24.0/15.0)*PI*get_eta_shear()*ai*aj*
	(2.0*ai*ai + ai*aj + 2.0*aj*aj)*
	(log(rh) - log(0.5*(ai+aj)/r_lu_t)) / ((ai+aj)*(ai+aj)*(ai+aj));

      force.x += fmod*(du.x - rdotdu*runit.x);
      force.y += fmod*(du.y - rdotdu*runit.y);
      force.z += fmod*(du.z - rdotdu*runit.z);
    }
  }

  return force;
}


/*****************************************************************************
 *
 *  COLL_soft_sphere
 *
 *  Extra interaction potential between two particles which goes like
 *  alpha / h^beta. The additional potential is smoothly truncated at
 *  r_ssph, and diverges at separation of r_clus (cf lubrication
 *  corrections, which diverge at zero separation).
 *
 *****************************************************************************/

FVector COLL_soft_sphere(Colloid * p_i, Colloid * p_j, FVector r_ij, Float h) {

  FVector force;
  Float   r_ssph = Global_Colloid.r_ssph;
  Float   r_clus = Global_Colloid.r_clus;
  Float   alpha  = Global_Colloid.drop_in_p1;
  Float   beta   = Global_Colloid.drop_in_p2;
  Float   fmod;

  force = UTIL_fvector_zero();

#ifdef _NO_SOFT_SPHERE_
  return force;
#endif


  /* Reduced separation here includes r_clus */
  
  h = h - p_i->ah - p_j->ah - r_clus; 

  if (h < r_ssph && h > 0.0) {
    fmod    = -alpha*(pow(h, beta) - pow((r_ssph - r_clus), beta));
    force.x = fmod*r_ij.x;
    force.y = fmod*r_ij.y;
    force.z = fmod*r_ij.z;
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
 *  COLL_fcoords_from_ijk
 *
 *  Return the physical coordinates (x,y,z) of the lattice site with
 *  index (i,j,k) as an FVector.
 *
 *  So the convention is:
 *         i = 1 => x = 1.0 etc. so the 'control volume' for lattice
 *         site i extends from x(i)-1/2 to x(i)+1/2.
 *         Halo points at i = 0 and i = N.x+1 are images of i = N.x
 *         and i = 1, respectively. At the moment, the halo points
 *         retain 'unphysical' coordinates 0 and N.x+1.
 *
 *****************************************************************************/

FVector COLL_fcoords_from_ijk(int i, int j, int k) {

  FVector coord;

  coord.x = (Float) i;
  coord.y = (Float) j;
  coord.z = (Float) k;

  return coord;
}



/*****************************************************************************
 *
 *  COLL_fvector_separation
 *
 *  Returns the vector which joins the centre of two colloids. The
 *  vector starts at position r1, and finishes at r2.
 *  If the shortest distance between colloids is across the periodic
 *  boundaries, the appropriate separation should be returned.
 *
 *****************************************************************************/

FVector COLL_fvector_separation(FVector r1, FVector r2) {

  FVector rsep;

  rsep.x = r2.x - r1.x;
  rsep.y = r2.y - r1.y;
  rsep.z = r2.z - r1.z;

  /* Correct if boundaries are periodic. */
  /* I expect something like: if (periodic.x && ...) */

  if (rsep.x > L(X)/2.0)
    rsep.x -= L(X);
  if (rsep.x < -L(X)/2.0)
    rsep.x += L(X);

  if (rsep.y > L(Y)/2.0)
    rsep.y -= L(Y);
  if (rsep.y < -L(Y)/2.0)
    rsep.y += L(Y);

  if (rsep.z > L(Z)/2.0)
    rsep.z -= L(Z);
  if (rsep.z < -L(Z)/2.0)
    rsep.z += L(Z);

  return rsep;
}


/*****************************************************************************
 *
 *  COLL_bounce_back
 *
 *  This is the top-level routine for update of colloid velocities
 *  following the approach of Nguyen and Ladd [Phys. Rev. E {\bf 66},
 *  046708 (2002)]. "Velocities" refers to linear and angular
 *  velocities.
 *
 *  An implicit velocity update requires two sweeps through the
 *  boundary nodes:
 *
 *  (1) The first sweep invloves computing the velocity-independent
 *      force and torque on each colloid and the elements of the
 *      drag matrix for each colloid.
 *
 *  [(2) For domain decomposition, communication will be required
 *       here to make sure sums over boundary links are available
 *       to all processes.]
 *
 *  (3) Update the velocity of each colloid. There are two cases:
 *      (a) an isolated colloid can undergo an independent update
 *          involving a 6x6 linear algebra problem. This is the
 *          code below.
 *      (b) clusters of colloids (2 or more) closer than the
 *          lubracition cut-off criterion must undergo a combined
 *          update to take account of additional velocity-dependent
 *          lubrication forces. See CLUS_bounce_back_clusters().
 *
 *  (4) Having computed the velocity update for each colloid,
 *      the bounce-back on links can then be performed in the
 *      usual way using the updated colloid velocities.
 *
 *****************************************************************************/

void COLL_bounce_back() {

  /* First pass through the links */

  TIMER_start(TIMER_PARTICLE_HALO);
  CCOM_halo_sum(CHALO_TYPE1);
  TIMER_stop(TIMER_PARTICLE_HALO);

  COLL_bounce_back_pass1();

  TIMER_start(TIMER_PARTICLE_HALO);
  CCOM_halo_sum(CHALO_TYPE2);
  TIMER_stop(TIMER_PARTICLE_HALO);

  COLL_update_colloids();

  COLL_bounce_back_pass2();

  return;
}

/*****************************************************************************
 *
 *  COLL_update_colloids
 *
 *  Update the velocity and position of each particle.
 *
 *  This is a linear algebra problem, which is always 6x6, and is
 *  solved using a bog-standard Gaussian elimination with partial
 *  pivoting, followed by  backsubstitution.
 *
 *  Issues:
 *    - Could eliminate lower triangle storage (or use library routine)
 *    - Dianostics could be moved elsewhere, although this is the only
 *      place where the true force on the particle is available at the
 *      moment.
 *
 *****************************************************************************/

void COLL_update_colloids() {

  Colloid   * p_colloid;

  int         ic, jc, kc;
  IVector     ncell = Global_Colloid.Ncell;

  Float xb[6];
  Float a[6][6];
  int   ipivot[6];
  int   iprow = 0;                 /* The pivot row */
  int   idash, j, k;

  Float mass;
  Float moment;
  Float tmp;

  Float c4r3pirho = (4.0/3.0)*PI*Global_Colloid.rho;
  Float c2r5      = 2.0/5.0;

  FVector wl;

  TIMER_start(TIMER_PARTICLE_UPDATE);

  /* Loop round cells and update each particle velocity */

  for (ic = 0; ic <= ncell.x + 1; ic++)
    for (jc = 0; jc <= ncell.y + 1; jc++)
      for (kc = 0; kc <= ncell.z + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  wl = WALL_lubrication(p_colloid);

	  /* Set up the matrix problem and solve it here. */


	  /* Mass and moment of inertia are those of a hard sphere
	  * with the input radius */

	  mass = c4r3pirho*pow(p_colloid->a0, 3);
	  moment = c2r5*mass*pow(p_colloid->a0, 2);

	  /* Add inertial terms to diagonal elements */

	  a[0][0] = mass +   p_colloid->zeta[0]  - wl.x;
	  a[0][1] =          p_colloid->zeta[1];
	  a[0][2] =          p_colloid->zeta[2];
	  a[0][3] =          p_colloid->zeta[3];
	  a[0][4] =          p_colloid->zeta[4];
	  a[0][5] =          p_colloid->zeta[5];
	  a[1][1] = mass +   p_colloid->zeta[6]  - wl.y;
	  a[1][2] =          p_colloid->zeta[7];
	  a[1][3] =          p_colloid->zeta[8];
	  a[1][4] =          p_colloid->zeta[9];
	  a[1][5] =          p_colloid->zeta[10];
	  a[2][2] = mass +   p_colloid->zeta[11] - wl.z;
	  a[2][3] =          p_colloid->zeta[12];
	  a[2][4] =          p_colloid->zeta[13];
	  a[2][5] =          p_colloid->zeta[14];
	  a[3][3] = moment + p_colloid->zeta[15];
	  a[3][4] =          p_colloid->zeta[16];
	  a[3][5] =          p_colloid->zeta[17];
	  a[4][4] = moment + p_colloid->zeta[18];
	  a[4][5] =          p_colloid->zeta[19];
	  a[5][5] = moment + p_colloid->zeta[20];

	  /* Lower triangle */

	  a[1][0] = a[0][1];
	  a[2][0] = a[0][2];
	  a[2][1] = a[1][2];
	  a[3][0] = a[0][3];
	  a[3][1] = a[1][3];
	  a[3][2] = a[2][3];
	  a[4][0] = a[0][4];
	  a[4][1] = a[1][4];
	  a[4][2] = a[2][4];
	  a[4][3] = a[3][4];
	  a[5][0] = a[0][5];
	  a[5][1] = a[1][5];
	  a[5][2] = a[2][5];
	  a[5][3] = a[3][5];
	  a[5][4] = a[4][5];

	  /* Form the right-hand side */

	  xb[0] = mass*p_colloid->v.x  + p_colloid->f0.x + p_colloid->force.x;
	  xb[1] = mass*p_colloid->v.y  + p_colloid->f0.y + p_colloid->force.y;
	  xb[2] = mass*p_colloid->v.z  + p_colloid->f0.z + p_colloid->force.z;

	  xb[3] = moment*p_colloid->omega.x + p_colloid->t0.x
	        + p_colloid->torque.x;
	  xb[4] = moment*p_colloid->omega.y + p_colloid->t0.y
	        + p_colloid->torque.y;
	  xb[5] = moment*p_colloid->omega.z + p_colloid->t0.z
	        + p_colloid->torque.z;

	  /* Begin the Gaussian elimination */

	  for (k = 0; k < 6; k++) {
	    ipivot[k] = -1;
	  }

	  for (k = 0; k < 6; k++) {

	    /* Find pivot row */
	    tmp = 0.0;
	    for (idash = 0; idash < 6; idash++) {
	      if (ipivot[idash] == -1) {
		if (fabs(a[idash][k]) >= tmp) {
		  tmp = fabs(a[idash][k]);
		  iprow = idash;
		}
	      }
	    }
	    ipivot[k] = iprow;

	    /* divide pivot row by the pivot element a[iprow][k] */

	    if (a[iprow][k] == 0.0) {
	      fatal("Gaussain elimination failed in COLL_update\n");
	    }

	    tmp = 1.0 / a[iprow][k];

	    for (j = k; j < 6; j++) {
	      a[iprow][j] *= tmp;
	    }
	    xb[iprow] *= tmp;

	    /* Subtract the pivot row (scaled) from remaining rows */

	    for (idash = 0; idash < 6; idash++) {
	      if (ipivot[idash] == -1) {
		tmp = a[idash][k];
		for (j = k; j < 6; j++) {
		  a[idash][j] -= tmp*a[iprow][j];
		}
		xb[idash] -= tmp*xb[iprow];
	      }
	    }
	  }

	  /* Now do the back substitution */

	  for (idash = 5; idash > -1; idash--) {
	    iprow = ipivot[idash];
	    tmp = xb[iprow];
	    for (k = idash+1; k < 6; k++) {
	      tmp -= a[iprow][k]*xb[ipivot[k]];
	    }
	    xb[iprow] = tmp;
	  }


#if defined _NO_POSITION_UPDATE_
	  /* Don't update the position */
#else

	  /* Use mean of old and new velocity to update position */
	  p_colloid->r.x += (0.5*(p_colloid->v.x + xb[0]));
	  p_colloid->r.y += (0.5*(p_colloid->v.y + xb[1]));
	  p_colloid->r.z += (0.5*(p_colloid->v.z + xb[2]));
#endif

#if defined _NO_VELOCITY_UPDATE_
	  VERBOSE(("[%d] pos      (%f,%f,%f)\n", p_colloid->index,
		   p_colloid->r.x, p_colloid->r.y, p_colloid->r.z));
	  VERBOSE(("*cbar    (%g,%g,%g)\n",
		   p_colloid->cbar.x, p_colloid->cbar.y, p_colloid->cbar.z));
	  VERBOSE(("rxcbar   (%g,%g,%g)\n", p_colloid->rxcbar.x,
		   p_colloid->rxcbar.y, p_colloid->rxcbar.z));
	  VERBOSE(("f0       (%g,%g,%g)\n",
		   p_colloid->f0.x, p_colloid->f0.y, p_colloid->f0.z));
	  VERBOSE(("t0       (%g,%g,%g)\n",
		   p_colloid->t0.x, p_colloid->t0.y, p_colloid->t0.z));
	  /* Don't update the velocity */
	  {
	    Float tol = 0.0001;
	    if (Global_Colloid.N_colloid == 1) {
	      if (fabs(p_colloid->cbar.x) > tol ||
		  fabs(p_colloid->cbar.y) > tol ||
		  fabs(p_colloid->cbar.z) > tol) {
		verbose("incomplete particle at (%f,%f,%f)\n",
			p_colloid->r.x, p_colloid->r.y, p_colloid->r.z);
		fatal("");
	      }
	    }
	  }
#else
	  /* Unpack the solution vector. */

	  p_colloid->v.x = xb[0];
	  p_colloid->v.y = xb[1];
	  p_colloid->v.z = xb[2];

	  p_colloid->omega.x = xb[3];
	  p_colloid->omega.y = xb[4];
	  p_colloid->omega.z = xb[5];

#endif

#ifdef _NO_DIRECTOR_UPDATE_
#else
	  p_colloid->dir = UTIL_rotate_vector(p_colloid->dir, p_colloid->omega);
#endif

#if defined _FORCE_OUTPUT_
	  {
	  FVector fh, th;
	  fh = UTIL_fvector_zero();
	  th = UTIL_fvector_zero();

	  wl.x *= p_colloid->v.x;
	  wl.y *= p_colloid->v.y;
	  wl.z *= p_colloid->v.z;

	  fh.x = -(p_colloid->zeta[0]*p_colloid->v.x +
	           p_colloid->zeta[1]*p_colloid->v.y +
                   p_colloid->zeta[2]*p_colloid->v.z +
                   p_colloid->zeta[3]*p_colloid->omega.x +
                   p_colloid->zeta[4]*p_colloid->omega.y +
                   p_colloid->zeta[5]*p_colloid->omega.z);
	  fh.y = -(p_colloid->zeta[ 1]*p_colloid->v.x +
	           p_colloid->zeta[ 6]*p_colloid->v.y +
	           p_colloid->zeta[ 7]*p_colloid->v.z +
	           p_colloid->zeta[ 8]*p_colloid->omega.x +
	           p_colloid->zeta[ 9]*p_colloid->omega.y +
	           p_colloid->zeta[10]*p_colloid->omega.z);
	  fh.z = -(p_colloid->zeta[ 2]*p_colloid->v.x +
	           p_colloid->zeta[ 7]*p_colloid->v.y +
	           p_colloid->zeta[11]*p_colloid->v.z +
	           p_colloid->zeta[12]*p_colloid->omega.x +
	           p_colloid->zeta[13]*p_colloid->omega.y +
	           p_colloid->zeta[14]*p_colloid->omega.z);

#if defined _COLLOIDS_TEST_CALIBRATE_
	  /* Require the hydrodynamic force and te velocity of
	   * particle relative to the fluid. wl is used for the
	   * mean fluid velocity. */
	  wl = TEST_fluid_momentum();
	  fprintf(_fp_output, "%8.4e %8.4e %8.4e %8.4e %8.4e %8.4e\n",
		  p_colloid->f0.x + fh.x + 0.0*p_colloid->force.x,
		  p_colloid->f0.y + fh.y + 0.0*p_colloid->force.y,
		  p_colloid->f0.z + fh.z + 0.0*p_colloid->force.z,
		  p_colloid->v.x - wl.x,
		  p_colloid->v.y - wl.y,
		  p_colloid->v.z - wl.z);
#endif
#if defined _COLLOIDS_TEST_SPLASH_
	  /* As above, plus current position */
	  wl = TEST_fluid_momentum();
	  fprintf(_fp_output, "%8.4e %8.4e %8.4e\n",
		  p_colloid->f0.x + fh.x,
		  p_colloid->v.x - wl.x,
		  p_colloid->r.x);
#endif
#ifdef _SOMETHING_
	  fprintf(_fp_output, "[%7d]f.x: %8.4e %8.4e %8.4e %8.4e %8.4e\n",
		  p_colloid->index,
		  p_colloid->f0.x, fh.x, p_colloid->force.x, wl.x,
		  p_colloid->f0.x +fh.x +p_colloid->force.x +wl.x);
	  fprintf(_fp_output, "[%7d]f.y: %8.4e %8.4e %8.4e %8.4e %8.4e\n",
		  p_colloid->index,
		  p_colloid->f0.y, fh.y, p_colloid->force.y, wl.y,
		  p_colloid->f0.y +fh.y +p_colloid->force.y +wl.y);
	  fprintf(_fp_output, "[%7d]f.z: %8.4e %8.4e %8.4e %8.4e %8.4e\n",
		  p_colloid->index,
		  p_colloid->f0.z, fh.z, p_colloid->force.z, wl.z,
		  p_colloid->f0.z +fh.z +p_colloid->force.z +wl.z);
#endif

	  th.x = -(p_colloid->zeta[ 3]*p_colloid->v.x +
		   p_colloid->zeta[ 8]*p_colloid->v.y +
		   p_colloid->zeta[12]*p_colloid->v.z +
		   p_colloid->zeta[15]*p_colloid->omega.x +
		   p_colloid->zeta[16]*p_colloid->omega.y +
		   p_colloid->zeta[17]*p_colloid->omega.z);
	  th.y = -(p_colloid->zeta[ 4]*p_colloid->v.x +
		   p_colloid->zeta[ 9]*p_colloid->v.y +
		   p_colloid->zeta[13]*p_colloid->v.z +
		   p_colloid->zeta[16]*p_colloid->omega.x +
		   p_colloid->zeta[18]*p_colloid->omega.y +
		   p_colloid->zeta[19]*p_colloid->omega.z);
	  th.z = -(p_colloid->zeta[ 5]*p_colloid->v.x +
		   p_colloid->zeta[10]*p_colloid->v.y +
		   p_colloid->zeta[14]*p_colloid->v.z +
		   p_colloid->zeta[17]*p_colloid->omega.x +
		   p_colloid->zeta[19]*p_colloid->omega.y +
		   p_colloid->zeta[20]*p_colloid->omega.z);

#if defined _SOMETHING_
	  fprintf(_fp_output, "[%7d]t.x: %8.4e %8.4e %8.4e %8.4e %8.4e\n",
		  p_colloid->index,
		  p_colloid->t0.x, th.x, p_colloid->torque.x, 0.0,
		  p_colloid->t0.x +th.x +p_colloid->torque.x);
	  fprintf(_fp_output, "[%7d]t.y: %8.4e %8.4e %8.4e %8.4e %8.4e\n",
		  p_colloid->index,
		  p_colloid->t0.y, th.y, p_colloid->torque.y, 0.0,
		  p_colloid->t0.y +th.y +p_colloid->torque.y);
	  fprintf(_fp_output, "[%7d]t.z: %8.4e %8.4e %8.4e %8.4e %8.4e\n",
		  p_colloid->index,
		  p_colloid->t0.z, th.z, p_colloid->torque.z, 0.0,
		  p_colloid->t0.z +th.z +p_colloid->torque.z);
#endif /* _COLLOIDS_TEST_CALIBRATE_ */
	  }
#endif /* _FORCE_OUTPUT_ */

	  p_colloid = p_colloid->next;
	}
      }

  TIMER_stop(TIMER_PARTICLE_UPDATE);

  return;
}


/****************************************************************************
 *
 *  COLL_init_colloids_lattice
 *
 *  Initialise a regular array, or lattice, of colloids in 3 dimensions.
 *
 *  The number of colloids in each dimension (ncx, ncy, ncz)
 *  determines the (equal) spacing. Each colloid has the same input
 *  radius Global_Colloid->a0.
 *
 *  Initial momentum zero.
 *
 *  If the given number will not fit in the present system size,
 *  this function will barf (gracefully).
 *
 *  The total number of colloids is ncx*ncy*ncz: this function must
 *  be preceded by a call to CELL_init_cells!
 *
 ****************************************************************************/

void COLL_init_colloids_lattice(int ncx, int ncy, int ncz) {

  Colloid * tmp;
  Float   dx, dy, dz;
  FVector r0;
  Float   a0, ah;
  int     nx, ny, nz;
  int     n = 1;       /* Index of first colloid is 1 (not zero!) */

  a0 = Global_Colloid.a0;
  ah = Global_Colloid.ah;

  if (ncx < 1 || ncy < 1 || ncz < 1) fatal("n < 1 in init_colloids_lattice\n");

  /* The system must be long enough to accomodate the
   * given number of colloids, i.e., the separation
   * should not be less than zero */

  dx = (L(X) - 2.0*ncx*ah)/(1.0 + ncx);
  dy = (L(Y) - 2.0*ncy*ah)/(1.0 + ncy);
  dz = (L(Z) - 2.0*ncz*ah)/(1.0 + ncz);

  if (dx < 0.0) fatal("dx < 0 in init_colloids_lattice\n");
  if (dy < 0.0) fatal("dy < 0 in init_colloids_lattice\n");
  if (dz < 0.0) fatal("dz < 0 in init_colloids_lattice\n");

  /* Allocate and initialise. */

  for (nx = 1; nx <= ncx; nx++) {
    for (ny = 1; ny <= ncy; ny++) {
      for (nz = 1; nz <= ncz; nz++) {

	r0.x = Lmin(X) + dx*nx + (2.0*nx - 1.0)*ah;
	r0.y = Lmin(Y) + dy*ny + (2.0*ny - 1.0)*ah;
	r0.z = Lmin(Z) + dz*nz + (2.0*nz - 1.0)*ah;

	VERBOSE(("%3d: %f %f %f\n", n++, r0.x, r0.y, r0.z));

#ifdef _HARMONIC_TRAP_
	tmp->stats.x = tmp->r.x;
	tmp->stats.y = tmp->r.y;
	tmp->stats.z = tmp->r.z;
#endif
      }
    }
  }

  fatal("Stop.\n");
  return;
}
