/*****************************************************************************
 *
 *  blue_phase_rt.c
 *
 *  Run time input for blue phase free energy, and related parameters.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "field.h"
#include "field_grad.h"
#include "colloids_Q_tensor.h"
#include "free_energy.h"
#include "blue_phase.h"
#include "blue_phase_init.h"
#include "blue_phase_rt.h"

/*****************************************************************************
 *
 *  blue_phase_run_time
 *
 *  Pick up the liquid crystal parameters from the input.
 *
 *****************************************************************************/

void blue_phase_run_time(void) {

  int n;
  int redshift_update;
  char method[FILENAME_MAX];
  char type[FILENAME_MAX];
  char type_wall[FILENAME_MAX];
  double a0, gamma, q0, kappa0, kappa1;
  double amplitude;
  double xi;
  double zeta;
  double w,w_wall;
  double redshift;
  double epsilon;
  double electric[3];

  info("Blue phase free energy selected.\n");

  /* PARAMETERS */

  n = RUN_get_double_parameter("lc_a0", &a0);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_gamma", &gamma);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_q0", &q0);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_kappa0", &kappa0);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_kappa1", &kappa1);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_xi", &xi);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_q_init_amplitude", &amplitude);
  assert(n == 1);

  /* Use a default redshift of 1 */
  redshift = 1.0;
  RUN_get_double_parameter("lc_init_redshift", &redshift);

  redshift_update = 0;
  RUN_get_int_parameter("lc_redshift_update", &redshift_update);

  /* Use a default zeta (no activity) of 0 */
  zeta = 0.0;
  RUN_get_double_parameter("lc_active_zeta", &zeta);

  info("\n");
  info("Liquid crystal blue phase free energy\n");
  info("Bulk parameter A0:         = %14.7e\n", a0);
  info("Magnitude of order gamma   = %14.7e\n", gamma);
  info("Pitch wavevector q0        = %14.7e\n", q0);
  info("... gives pitch length     = %14.7e\n", 2.0*4.0*atan(1.0)/q0);
  info("Elastic constant kappa0    = %14.7e\n", kappa0);
  info("Elastic constant kappa1    = %14.7e\n", kappa1);
  info("Amplitude (uniaxial) order = %14.7e\n", amplitude);

  /* One-constant approximation enforced. */
  assert(kappa0 == kappa1);

  blue_phase_set_free_energy_parameters(a0, gamma, kappa0, q0);
  blue_phase_init_amplitude_set(amplitude);
  blue_phase_set_xi(xi);
  blue_phase_redshift_set(redshift);
  blue_phase_redshift_update_set(redshift_update);
  blue_phase_set_zeta(zeta);

  info("Effective aspect ratio xi  = %14.7e\n", xi);
  info("Chirality                  = %14.7e\n", blue_phase_chirality());
  info("Reduced temperature        = %14.7e\n",
       blue_phase_reduced_temperature());
  info("Initial redshift           = %14.7e\n", redshift);
  info("Dynamic redshift update    = %14s\n",
       redshift_update == 0 ? "no" : "yes");
  info("LC activity constant zeta  = %14.7e\n", zeta);


  /* Default electric field stuff zero */

  epsilon = 0.0;
  RUN_get_double_parameter("lc_dielectric_anisotropy", &epsilon);
  electric[X] = 0.0;
  electric[Y] = 0.0;
  electric[Z] = 0.0;

  n = RUN_get_double_parameter_vector("electric_e0", electric);

  if (n == 1) {
    blue_phase_dielectric_anisotropy_set(epsilon);
    blue_phase_electric_field_set(electric);
    info("\n");
    info("Dielectric anisotropy      = %14.7e\n", epsilon);
    info("Electric field             = %14.7e %14.7e %14.7e\n",
	 electric[X], electric[Y], electric[Z]);
    info("Dimensionless field e      = %14.7e\n",
         blue_phase_dimensionless_field_strength());
  }

  fe_density_set(blue_phase_free_energy_density);
  fe_chemical_stress_set(blue_phase_chemical_stress);

  /* Surface anchoring */

  RUN_get_string_parameter("lc_anchoring_method", method, FILENAME_MAX);

  if (strcmp(method, "two") != 0) {
    /* There's a bit of an historical problem here, as 'two'
     * is now the only valid choice. However, it is worth
     * not getting a load a irrelevant output if no solids.
     * So I assert 'none' is the only other option. */
    if (strcmp(method, "none") != 0) fatal("Check anchoring method input\n");
  }
  else {

    /* Find out type */

    n = RUN_get_string_parameter("lc_anchoring", type, FILENAME_MAX);

    if (n == 1) {
      info("Please replace lc_anchoring by lc_wall_anchoring and/or\n");
      info("lc_coll_anchoring types\n");
      fatal("Please check input file and try agains.\n");
    }

    RUN_get_string_parameter("lc_coll_anchoring", type, FILENAME_MAX);

    if (strcmp(type, "normal") == 0) {
      colloids_q_tensor_anchoring_set(ANCHORING_NORMAL);
    }

    if (strcmp(type, "planar") == 0) {
      colloids_q_tensor_anchoring_set(ANCHORING_PLANAR);
    }

    /* Surface free energy parameter (method two only) */

    RUN_get_double_parameter("lc_anchoring_strength", &w);
    
    info("\n");
    info("Liquid crystal anchoring\n");
    info("Anchoring method:          = %14s\n", method);
    info("Anchoring type (colloids): = %14s\n", type);

    /* Walls (if present) separate type allowed but same strength */

    RUN_get_string_parameter("lc_wall_anchoring", type_wall, FILENAME_MAX);

    if (strcmp(type_wall, "normal") == 0) {
      wall_anchoring_set(ANCHORING_NORMAL);
    }

    if (strcmp(type_wall, "planar") == 0) {
      wall_anchoring_set(ANCHORING_PLANAR);
    }

    if (strcmp(type_wall, "fixed") == 0) {
      wall_anchoring_set(ANCHORING_FIXED);
    }

    /* Set the anchoring strength the same for colloid and wall */
    colloids_q_tensor_w_set(w);
    w_wall = w;
    wall_w_set(w_wall);
      
    /* Try if the specific parameter for colloid/wall exists */
    n =  RUN_get_double_parameter("lc_anchoring_strength_colloid", &w);
    if ( n == 1 ) colloids_q_tensor_w_set(w);
      
    n =  RUN_get_double_parameter("lc_anchoring_strength_wall", &w_wall);
    if( n == 1 ) wall_w_set(w_wall);
      
    info("Anchoring type (walls):          = %14s\n", type_wall);
    info("Surface free energy (colloid) w: = %14.7e\n", w);
    info("Surface free energy (wall) w:    = %14.7e\n", w_wall);
    info("Ratio (colloid) w/kappa0:        = %14.7e\n", w/kappa0);
    info("Ratio (wall) w/kappa0:           = %14.7e\n", w_wall/kappa0);
    info("Computed surface order f(gamma)  = %14.7e\n",
	 blue_phase_amplitude_compute());

    /* For computed anchoring order [see blue_phase_amplitude_compute()] */
    if (gamma < (8.0/3.0)) fatal("Please check anchoring amplitude\n");
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_rt_initial_conditions
 *
 *  There are several choices:
 *
 *****************************************************************************/

int blue_phase_rt_initial_conditions(field_t * q) {

  int  n;
  char key1[FILENAME_MAX];

  double nhat[3] = {1.0, 0.0, 0.0};
  double nhat2[3] = {64.0, 3.0, 1.0};

  info("\n");

  n = RUN_get_string_parameter("lc_q_initialisation", key1, FILENAME_MAX);
  assert(n == 1);

  info("\n");

  if (strcmp(key1, "twist") == 0) {
    /* This gives cholesteric_z (for backwards compatibility) */
    info("Initialising Q_ab to cholesteric\n");
    info("Helical axis Z\n");
    blue_phase_twist_init(q, Z);
  }

  if (strcmp(key1, "cholesteric_x") == 0) {
    info("Initialising Q_ab to cholesteric\n");
    info("Helical axis X\n");
    blue_phase_twist_init(q, X);
  }

  if (strcmp(key1, "cholesteric_y") == 0) {
    info("Initialising Q_ab to cholesteric\n");
    info("Helical axis Y\n");
    blue_phase_twist_init(q, Y);
  }

  if (strcmp(key1, "cholesteric_z") == 0) {
    info("Initialising Q_ab to cholesteric\n");
    info("Helical axis Z\n");
    blue_phase_twist_init(q, Z);
  }

  if (strcmp(key1, "nematic") == 0) {
    info("Initialising Q_ab to nematic\n");
    RUN_get_double_parameter_vector("lc_init_nematic", nhat);
    info("Director:  %14.7e %14.7e %14.7e\n", nhat[X], nhat[Y], nhat[Z]);
    blue_phase_nematic_init(q, nhat);
  }

  if (strcmp(key1, "o8m") == 0) {
    info("Initialising Q_ab using O8M (BPI)\n");
    blue_phase_O8M_init(q);
  }

  if (strcmp(key1, "o2") == 0) {
    info("Initialising Q_ab using O2 (BPII)\n");
    blue_phase_O2_init(q);
  }

  if (strcmp(key1, "o5") == 0) {
    info("Initialising Q_ab using O5\n");
    blue_phase_O5_init(q);
  }

  if (strcmp(key1, "h2d") == 0) {
    info("Initialising Q_ab using H2D\n");
    blue_phase_H2D_init(q);
  }

  if (strcmp(key1, "h3da") == 0) {
    info("Initialising Q_ab using H3DA\n");
    blue_phase_H3DA_init(q);
  }

  if (strcmp(key1, "h3db") == 0) {
    info("Initialising Q_ab using H3DB\n");
    blue_phase_H3DB_init(q);
  }

  if (strcmp(key1, "dtc") == 0) {
    info("Initialising Q_ab using DTC\n");
    blue_phase_DTC_init(q);
  }

  if (strcmp(key1, "bp3") == 0) {
    info("Initialising Q_ab using BPIII\n");
    RUN_get_double_parameter_vector("lc_init_bp3", nhat2);
    info("BPIII specifications: N_DTC=%g,  R_DTC=%g,  ", nhat2[0], nhat2[1]);
    if (nhat2[2] == 0) info("isotropic environment\n");
    if (nhat2[2] == 1) info("cholesteric environment\n");
    blue_phase_BPIII_init(q, nhat2);
  }

  if (strcmp(key1, "random") == 0) {
    info("Initialising Q_ab randomly\n");
    blue_set_random_q_init(q);
  }

  return 0;
}
