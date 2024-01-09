/*****************************************************************************
 *
 *  blue_phase_rt.c
 *
 *  Run time input for blue phase free energy, and related parameters.
 *  Also relevant Beris Edwards parameters.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "coords.h"
#include "blue_phase_init.h"
#include "blue_phase_rt.h"
#include "physics.h"
#include "util_bits.h"

int blue_phase_rt_coll_anchoring(pe_t * pe, rt_t * rt, rt_enum_t rt_err_level,
				 lc_anchoring_param_t * coll);
int blue_phase_rt_wall_anchoring(pe_t * pe, rt_t * rt, rt_enum_t rt_err_level,
				 lc_anchoring_param_t * wall);

/*****************************************************************************
 *
 *  blue_phase_init_rt
 *
 *  Pick up the liquid crystal parameters from the input.
 *
 *****************************************************************************/

__host__ int blue_phase_init_rt(pe_t * pe, rt_t *rt,
				fe_lc_t * fe,
				beris_edw_t * be) {
  int n;
  int fe_is_lc_droplet = 0;
  int redshift_update;
  char method[BUFSIZ] = "s7"; /* This is the default */
  char type[BUFSIZ] = "none";
  char type_wall[BUFSIZ] = "none";

  double epsilon;
  double redshift;
  double zeta0, zeta1, zeta2;
  double gamma;

  /* Wall anchoring strengths */

  double w1;
  double w2;
  double w1_wall;
  double w2_wall;

  /* Derived quantities */
  double amp0;
  double ck;
  double tred;

  fe_lc_param_t fe_param = {0};
  beris_edw_param_t be_param = {0};

  assert(pe);
  assert(rt);
  assert(fe);
  assert(be);

  pe_info(pe, "Blue phase free energy selected.\n");

  {
    char description[BUFSIZ] = {0};
    rt_string_parameter(rt, "free_energy", description, BUFSIZ);
    fe_is_lc_droplet = (strcmp(description, "lc_droplet") == 0);
  }
  /* PARAMETERS */
  /* Note that for LC droplet, we should not specify gamma here. */

  n = rt_double_parameter(rt, "lc_a0", &fe_param.a0);
  if (n != 1) pe_fatal(pe, "Please specify lc_a0 <value>\n");

  n = rt_double_parameter(rt, "lc_gamma", &fe_param.gamma);
  if (n != 1 && fe_is_lc_droplet == 0) {
    pe_fatal(pe, "Please specify lc_gamma <value>\n");
  }

  n = rt_double_parameter(rt, "lc_q0", &fe_param.q0);
  if (n != 1) pe_fatal(pe, "Please specify lc_q0 <value>\n");

  n = rt_double_parameter(rt, "lc_kappa0", &fe_param.kappa0);
  if (n != 1) pe_fatal(pe, "Please specify lc_kappa0 <value>\n");

  n = rt_double_parameter(rt, "lc_kappa1", &fe_param.kappa1);
  if (n != 1) pe_fatal(pe, "Please specify lc_kappa1 <value>\n");

  n = rt_double_parameter(rt, "lc_xi", &fe_param.xi);
  if (n != 1) pe_fatal(pe, "Please specify lc_xi <value>\n");

  n = rt_double_parameter(rt, "lc_q_init_amplitude", &fe_param.amplitude0);
  if (n != 1) pe_fatal(pe, "Please specify lc_q_init_amplitude <value>\n");

  /* Use a default redshift of 1 */
  redshift = 1.0;
  rt_double_parameter(rt, "lc_init_redshift", &redshift);
  fe_param.redshift = redshift;

  redshift_update = 0;
  rt_int_parameter(rt, "lc_redshift_update", &redshift_update);
  fe_param.is_redshift_updated = redshift_update;

  pe_info(pe, "\n");
  pe_info(pe, "Liquid crystal blue phase free energy\n");
  pe_info(pe, "Bulk parameter A0:         = %14.7e\n", fe_param.a0);
  pe_info(pe, "Magnitude of order gamma   = %14.7e\n", fe_param.gamma);
  pe_info(pe, "Pitch wavevector q0        = %14.7e\n", fe_param.q0);
  pe_info(pe, "... gives pitch length     = %14.7e\n", 2.0*4.0*atan(1.0)/fe_param.q0);
  pe_info(pe, "Elastic constant kappa0    = %14.7e\n", fe_param.kappa0);
  pe_info(pe, "Elastic constant kappa1    = %14.7e\n", fe_param.kappa1);
  pe_info(pe, "Amplitude (uniaxial) order = %14.7e\n", fe_param.amplitude0);

  /* One-constant approximation enforced. Exactly. */

  if (0 == util_double_same(fe_param.kappa0, fe_param.kappa1)) {
    pe_info(pe,  "Must have elastic constants the same\n");
    pe_fatal(pe, "Please check and try again\n");
  }

  fe_lc_param_set(fe, &fe_param);

  fe_lc_chirality(fe, &ck);
  fe_lc_reduced_temperature(fe, &tred);

  pe_info(pe, "Effective aspect ratio xi  = %14.7e\n", fe_param.xi);
  pe_info(pe, "Chirality                  = %14.7e\n", ck);
  pe_info(pe, "Reduced temperature        = %14.7e\n", tred);
  pe_info(pe, "Initial redshift           = %14.7e\n", fe_param.redshift);
  pe_info(pe, "Dynamic redshift update    = %14s\n",
	  redshift_update == 0 ? "no" : "yes");


  /* Use a default zeta (no activity) of 0 */
  /* Active stress is:
   *   s_ab = zeta0 d_ab - zeta1 Q_ab - zeta2 (d_a p_b  + d_b p_a)
   * with p_a = Q_ak d_m Q_mk
   */

  fe_param.is_active = rt_switch(rt, "lc_activity");
  pe_info(pe, "Liquid crystal activity      %14s\n",
	  fe_param.is_active == 0 ? "No" : "Yes");

  if (fe_param.is_active) {
    zeta0 = 0.0;
    zeta1 = 0.0;
    zeta2 = 0.0;
    rt_double_parameter(rt, "lc_active_zeta0", &zeta0);
    rt_double_parameter(rt, "lc_active_zeta1", &zeta1);
    rt_double_parameter(rt, "lc_active_zeta2", &zeta2);
    fe_param.zeta0 = zeta0;
    fe_param.zeta1 = zeta1;
    fe_param.zeta2 = zeta2;

    pe_info(pe, "Activity constant zeta0    = %14.7e\n", fe_param.zeta0);
    pe_info(pe, "Activity constant zeta1    = %14.7e\n", fe_param.zeta1);
    pe_info(pe, "Activity constant zeta2    = %14.7e\n", fe_param.zeta2);
  }

  /* Default electric field stuff zero */

  epsilon = 0.0;
  rt_double_parameter(rt, "lc_dielectric_anisotropy", &epsilon);
  fe_param.epsilon = epsilon;

  n = rt_double_parameter_vector(rt, "electric_e0", fe_param.e0);

  if (n == 1) {
    double ered = 0.0;
    fe_lc_dimensionless_field_strength(&fe_param, &ered);
    pe_info(pe, "Dielectric anisotropy      = %14.7e\n", epsilon);
    pe_info(pe, "Dimensionless field e      = %14.7e\n", ered);
  }

  /* Surface anchoring. The default meoth si "s7" (set above). */

  rt_string_parameter(rt, "lc_anchoring_method", method, FILENAME_MAX);

  if (strcmp(method, "s7")  == 0) {

    /* Check what is wanted for walls/colloids */ 
    blue_phase_rt_wall_anchoring(pe, rt, RT_FATAL, &fe_param.wall);
    blue_phase_rt_coll_anchoring(pe, rt, RT_FATAL, &fe_param.coll);

    if (fe_param.wall.type != LC_ANCHORING_NONE) {
      pe_info(pe, "\n");
      pe_info(pe, "Liquid crystal anchoring:\n");

      pe_info(pe, "Wall anchoring type:          %s\n",
	      lc_anchoring_type_from_enum(fe_param.wall.type));

      if (fe_param.wall.type == LC_ANCHORING_FIXED) {
	pe_info(pe, "Preferred orientation:       %14.7e %14.7e %14.7e\n",
		fe_param.wall.nfix[X],
		fe_param.wall.nfix[Y],
		fe_param.wall.nfix[Z]);
      }

      pe_info(pe, "Wall anchoring w1:           %14.7e\n", fe_param.wall.w1);

      if (fe_param.wall.type == LC_ANCHORING_PLANAR) {
	pe_info(pe, "Wall anchoring w2:           %14.7e\n", fe_param.wall.w2);
      }
    }

    if (fe_param.coll.type != LC_ANCHORING_NONE) {

      pe_info(pe, "\n");
      pe_info(pe, "Liquid crystal anchoring:\n");

      pe_info(pe, "Colloid anchoring type:       %s\n",
	      lc_anchoring_type_from_enum(fe_param.coll.type));

      if (fe_param.coll.type == LC_ANCHORING_NORMAL) {
	pe_info(pe, "Colloid anchoring w1:        %14.7e\n", fe_param.coll.w1);
      }
      if (fe_param.coll.type == LC_ANCHORING_PLANAR) {
	pe_info(pe, "Colloid anchoring w1:        %14.7e\n", fe_param.coll.w1);
	pe_info(pe, "Colloid anchoring w2:        %14.7e\n", fe_param.coll.w2);
      }
    }
  }
  else if (strcmp(method, "two") == 0) {

    lc_anchoring_enum_t anchoring_wall = LC_ANCHORING_NONE;
    lc_anchoring_enum_t anchoring_coll = LC_ANCHORING_NONE;
    double w1_coll;
    double w2_coll;

    /* Older-style input for "lc_anchoring_method". The name "two"
     * is because, historically, it was the second method tried. */

    /* Find out type */

    n = rt_string_parameter(rt, "lc_anchoring", type, FILENAME_MAX);

    if (n == 1) {
      pe_info(pe, "Please replace lc_anchoring by lc_wall_anchoring and/or\n");
      pe_info(pe, "lc_coll_anchoring types\n");
      pe_fatal(pe, "Please check input file and try agains.\n");
    }

    rt_string_parameter(rt, "lc_coll_anchoring", type, FILENAME_MAX);

    if (strcmp(type, "normal") == 0) {
      anchoring_coll = LC_ANCHORING_NORMAL;
    }

    if (strcmp(type, "planar") == 0) {
      anchoring_coll = LC_ANCHORING_PLANAR;
    }

    /* Surface free energy parameter */

    w1 = 0.0;
    w2 = 0.0;
    rt_double_parameter(rt, "lc_anchoring_strength", &w1);
    rt_double_parameter(rt, "lc_anchoring_strength_2", &w2);

    pe_info(pe, "\n");
    pe_info(pe, "Liquid crystal anchoring\n");
    pe_info(pe, "Anchoring method:          = %14s\n", method);
    pe_info(pe, "Anchoring type (colloids): = %14s\n", type);

    /* Walls (if present) separate type allowed but same strength */

    w1_wall = 0.0;
    w2_wall = 0.0;
    strncpy(type_wall, type, BUFSIZ - strnlen(type, BUFSIZ) - 1);

    rt_string_parameter(rt, "lc_wall_anchoring", type_wall, FILENAME_MAX);

    if (strcmp(type_wall, "normal") == 0) {
      anchoring_wall = LC_ANCHORING_NORMAL;
      w1_wall = w1;
      w2_wall = 0.0;
    }

    if (strcmp(type_wall, "planar") == 0) {
      anchoring_wall = LC_ANCHORING_PLANAR;
      w1_wall = w1;
      w2_wall = w2;
    }

    if (strcmp(type_wall, "fixed") == 0) {
      double nfix[3] = {0.0, 1.0, 0.0}; /* default orientation */
      double rmod;
      anchoring_wall = LC_ANCHORING_FIXED;
      w1_wall = w1;
      w2_wall = 0.0;
      rt_double_parameter_vector(rt, "lc_wall_fixed_orientation", nfix);
      /* Make sure it's a unit vector */
      rmod = 1.0/sqrt(nfix[X]*nfix[X] + nfix[Y]*nfix[Y] + nfix[Z]*nfix[Z]);
      nfix[X] = rmod*nfix[X];
      nfix[Y] = rmod*nfix[Y];
      nfix[Z] = rmod*nfix[Z];
      fe_param.wall.nfix[X] = nfix[X];
      fe_param.wall.nfix[Y] = nfix[Y];
      fe_param.wall.nfix[Z] = nfix[Z];
    }

    /* Colloids default, then look for specific value */

    if (strcmp(type, "normal") == 0) w2 = 0.0;
    if (strcmp(type, "fixed")  == 0) w2 = 0.0;
      
    n =  rt_double_parameter(rt, "lc_anchoring_strength_colloid", &w1);

    if ( n == 1 ) {
      if (strcmp(type, "normal") == 0) w2 = 0.0;
      if (strcmp(type, "planar") == 0) w2 = w1;
      if (strcmp(type, "fixed")  == 0) w2 = 0.0;
    }

    w1_coll = w1;
    w2_coll = w2;

    /* Wall */

    n =  rt_double_parameter(rt, "lc_anchoring_strength_wall", &w1_wall);
    if ( n == 1 ) {
      if (strcmp(type_wall, "normal") == 0) w2_wall = 0.0;
      if (strcmp(type_wall, "planar") == 0) w2_wall = w1_wall;
      if (strcmp(type_wall, "fixed")  == 0) w2_wall = 0.0;
    }

    fe_lc_amplitude_compute(&fe_param, &amp0);

    pe_info(pe, "Anchoring type (walls):          = %14s\n",   type_wall);
    pe_info(pe, "Surface free energy (colloid)w1: = %14.7e\n", w1);
    pe_info(pe, "Surface free energy (colloid)w2: = %14.7e\n", w2);
    pe_info(pe, "Surface free energy (wall) w1:   = %14.7e\n", w1_wall);
    pe_info(pe, "Surface free energy (wall) w2:   = %14.7e\n", w2_wall);
    pe_info(pe, "Ratio (colloid) w1/kappa0:       = %14.7e\n",
	    w1/fe_param.kappa0);
    pe_info(pe, "Ratio (wall) w1/kappa0:          = %14.7e\n",
	    w1_wall/fe_param.kappa0);
    pe_info(pe, "Computed surface order f(gamma)  = %14.7e\n", amp0);

    if (anchoring_wall == LC_ANCHORING_FIXED) {
      pe_info(pe, "Wall fixed anchoring orientation = %14.7e %14.7e %14.7e\n",
	      fe_param.wall.nfix[X], fe_param.wall.nfix[Y], fe_param.wall.nfix[Z]);
    }

    /* For computed anchoring order [see fe_lc_amplitude_compute()] */
    if (fe_param.gamma < (8.0/3.0)) {
      pe_fatal(pe, "Please check anchoring amplitude\n");
    }

    fe_param.coll.type    = anchoring_coll;
    fe_param.coll.w1      = w1_coll;
    fe_param.coll.w2      = w2_coll;
    fe_param.wall.type    = anchoring_wall;
    fe_param.wall.w1      = w1_wall;
    fe_param.wall.w2      = w2_wall;
  }
  else {
    /* not recognised */
    pe_fatal(pe, "lc_anchoring_method must be either s7 or two\n");
  }

  fe_lc_param_set(fe, &fe_param);


  /* Beris Edwards */

  pe_info(pe, "\n");
  pe_info(pe, "Using Beris-Edwards solver:\n");

  n = rt_double_parameter(rt, "lc_Gamma", &gamma);

  if (n == 0) {
    pe_fatal(pe, "Please specify diffusion constant lc_Gamma in the input\n");
  }
  else {
    be_param.gamma = gamma;
    be_param.xi = fe_param.xi;
    beris_edw_param_set(be, &be_param);
    pe_info(pe, "Rotational diffusion const = %14.7e\n", gamma);
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_rt_initial_conditions
 *
 *  There are several choices:
 *
 *****************************************************************************/

__host__ int blue_phase_rt_initial_conditions(pe_t * pe, rt_t * rt, cs_t * cs,
					      fe_lc_t * fe, field_t * q) {

  int  n1, n2;
  int  rmin[3], rmax[3];
  char key1[FILENAME_MAX];

  double nhat[3] = {1.0, 0.0, 0.0};
  double nhat2[3] = {64.0, 3.0, 1.0};

  fe_lc_param_t param;
  fe_lc_param_t * feparam = &param;

  assert(pe);
  assert(cs);
  assert(fe);
  assert(q);

  fe_lc_param(fe, feparam);

  pe_info(pe, "\n");

  n1 = rt_string_parameter(rt, "lc_q_initialisation", key1, FILENAME_MAX);
  if (n1 != 1) pe_fatal(pe, "Please specify lc_q_initialisation <value>\n");

  pe_info(pe, "\n");

  if (strcmp(key1, "twist") == 0) {
    /* This gives cholesteric_z (for backwards compatibility) */
    pe_info(pe, "Initialising Q_ab to cholesteric\n");
    pe_info(pe, "Helical axis Z\n");
    blue_phase_twist_init(cs, feparam, q, Z);
  }

  if (strcmp(key1, "cholesteric_x") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric\n");
    pe_info(pe, "Helical axis X\n");
    blue_phase_twist_init(cs, feparam, q, X);
  }

  if (strcmp(key1, "cholesteric_y") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric\n");
    pe_info(pe, "Helical axis Y\n");
    blue_phase_twist_init(cs, feparam, q, Y);
  }

  if (strcmp(key1, "cholesteric_z") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric\n");
    pe_info(pe, "Helical axis Z\n");
    blue_phase_twist_init(cs, feparam, q, Z);
  }

  if (strcmp(key1, "nematic") == 0) {
    pe_info(pe, "Initialising Q_ab to nematic\n");
    rt_double_parameter_vector(rt, "lc_init_nematic", nhat);
    pe_info(pe, "Director:  %14.7e %14.7e %14.7e\n", nhat[X], nhat[Y], nhat[Z]);
    blue_phase_nematic_init(cs, feparam, q, nhat);
  }

  if (strcmp(key1, "active_nematic") == 0) {
    pe_info(pe, "Initialising Q_ab to active nematic\n");
    rt_double_parameter_vector(rt, "lc_init_nematic", nhat);
    pe_info(pe, "Director:  %14.7e %14.7e %14.7e\n", nhat[X], nhat[Y], nhat[Z]);
    blue_phase_active_nematic_init(cs, feparam, q, nhat);
  }

  if (strcmp(key1, "active_nematic_q2d_x") == 0) {
    pe_info(pe, "Initialising Q_ab to quasi-2d with strip parallel to X\n");
    lc_active_nematic_init_q2d(cs, feparam, q, X);
  }

  if (strcmp(key1, "active_nematic_q2d_y") == 0) {
    pe_info(pe, "Initialising Q_ab to quasi-2d with strip parallel to Y\n");
    lc_active_nematic_init_q2d(cs, feparam, q, Y);
  }

  if (strcmp(key1, "o8m") == 0) {

    int   is_rot = 0;                   /* Default no rotation. */
    double angles[3] = {0.0, 0.0, 0.0}; /* Default Euler rotation (degrees) */

    pe_info(pe, "Initialising Q_ab using O8M (BPI)\n");
    is_rot = rt_double_parameter_vector(rt, "lc_q_init_euler_angles", angles);

    if (is_rot) {
      pe_info(pe, "... initial conidition to be rotated ...\n");
      pe_info(pe, "Euler angle (deg): alpha_z = %14.7e\n", angles[0]);
      pe_info(pe, "Euler angle (deg): beta_x' = %14.7e\n", angles[1]);
      pe_info(pe, "Euler angle (deg): gamma_z'= %14.7e\n", angles[2]);
    }

    blue_phase_O8M_init(cs, feparam, q, angles);
  }

  if (strcmp(key1, "o2") == 0) {

    int   is_rot = 0;                   /* Default no rotation. */
    double angles[3] = {0.0, 0.0, 0.0}; /* Default Euler rotation (degrees) */

    pe_info(pe, "Initialising Q_ab using O2 (BPII)\n");
    is_rot = rt_double_parameter_vector(rt, "lc_q_init_euler_angles", angles);

    if (is_rot) {
      pe_info(pe, "... initial conidition to be rotated ...\n");
      pe_info(pe, "Euler angle (deg): alpha_z = %14.7e\n", angles[0]);
      pe_info(pe, "Euler angle (deg): beta_x' = %14.7e\n", angles[1]);
      pe_info(pe, "Euler angle (deg): gamma_z'= %14.7e\n", angles[2]);
    }

    blue_phase_O2_init(cs, feparam, q, angles);
  }

  if (strcmp(key1, "o5") == 0) {
    pe_info(pe, "Initialising Q_ab using O5\n");
    blue_phase_O5_init(cs, feparam, q);
  }

  if (strcmp(key1, "h2d") == 0) {
    pe_info(pe, "Initialising Q_ab using H2D\n");
    blue_phase_H2D_init(cs, feparam, q);
  }

  if (strcmp(key1, "h3da") == 0) {
    pe_info(pe, "Initialising Q_ab using H3DA\n");
    blue_phase_H3DA_init(cs, feparam, q);
  }

  if (strcmp(key1, "h3db") == 0) {
    pe_info(pe, "Initialising Q_ab using H3DB\n");
    blue_phase_H3DB_init(cs, feparam, q);
  }

  if (strcmp(key1, "dtc") == 0) {
    pe_info(pe, "Initialising Q_ab using DTC\n");
    blue_phase_DTC_init(cs, feparam, q);
  }

  if (strcmp(key1, "bp3") == 0) {
    pe_info(pe, "Initialising Q_ab using BPIII\n");
    rt_double_parameter_vector(rt, "lc_init_bp3", nhat2);
    pe_info(pe, "BPIII specifications: N_DTC=%g,  R_DTC=%g,  ", nhat2[0], nhat2[1]);
    if (nhat2[2] == 0) pe_info(pe, "isotropic environment\n");
    if (nhat2[2] == 1) pe_info(pe, "cholesteric environment\n");
    blue_phase_BPIII_init(cs, feparam, q, nhat2);
  }

  if (strcmp(key1, "cf1_x") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
    pe_info(pe, "Finger axis X, helical axis Y\n");
    blue_phase_cf1_init(cs, feparam, q, X);
  }

  if (strcmp(key1, "cf1_y") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
    pe_info(pe, "Finger axis Y, helical axis Z\n");
    blue_phase_cf1_init(cs, feparam, q, Y);
  }

  if (strcmp(key1, "cf1_z") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
    pe_info(pe, "Finger axis Z, helical axis X\n");
    blue_phase_cf1_init(cs, feparam, q, Z);
  }

  if (strcmp(key1, "cf1_fluc_x") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
    pe_info(pe, "with added traceless symmetric random fluctuation.\n");
    pe_info(pe, "Finger axis X, helical axis Y\n");
    blue_phase_random_cf1_init(cs, feparam, q, X);
  }

  if (strcmp(key1, "cf1_fluc_y") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
    pe_info(pe, "with added traceless symmetric random fluctuation.\n");
    pe_info(pe, "Finger axis Y, helical axis Z\n");
    blue_phase_random_cf1_init(cs, feparam, q, Y);
  }

  if (strcmp(key1, "cf1_fluc_z") == 0) {
    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
    pe_info(pe, "with added traceless symmetric random fluctuation.\n");
    pe_info(pe, "Finger axis Z, helical axis X\n");
    blue_phase_random_cf1_init(cs, feparam, q, Z);
  }

  if (strcmp(key1, "random") == 0) {
    pe_info(pe, "Initialising Q_ab randomly\n");
    blue_phase_random_q_init(cs, feparam, q);
  }

  if (strcmp(key1, "random_xy") == 0) {
    pe_info(pe, "Initialising Q_ab at random in (x,y)\n");
    blue_phase_random_q_2d(cs, feparam, q);
  }

  /* Superpose a rectangle of random Q_ab on whatever was above */

  n1 = rt_int_parameter_vector(rt, "lc_q_init_rectangle_min", rmin);
  n2 = rt_int_parameter_vector(rt, "lc_q_init_rectangle_max", rmax);

  if (n1 == 1 && n2 == 1) {
    pe_info(pe, "Superposing random rectangle\n");
    blue_phase_random_q_rectangle(cs, feparam, q, rmin, rmax);
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_rt_coll_anchoring
 *
 *  Newer style anchoring input which is documented for colloids.
 *  Normal or planar only for colloids.
 *
 *****************************************************************************/

int blue_phase_rt_coll_anchoring(pe_t * pe, rt_t * rt, rt_enum_t rt_err_level,
				 lc_anchoring_param_t * coll) {

  assert(pe);
  assert(rt);
  assert(coll);

  /* No colloids at all returns 0. */

  int ierr = 0;
  char atype[BUFSIZ] = {0};

  if (rt_string_parameter(rt, "lc_coll_anchoring", atype, BUFSIZ)) {

    coll->type = lc_anchoring_type_from_string(atype);

    switch (coll->type) {
    case LC_ANCHORING_NORMAL:
      ierr += rt_key_required(rt, "lc_coll_anchoring_w1", rt_err_level);
      rt_double_parameter(rt, "lc_coll_anchoring_w1", &coll->w1);
      break;
    case LC_ANCHORING_PLANAR:
      ierr += rt_key_required(rt, "lc_coll_anchoring_w1", rt_err_level);
      ierr += rt_key_required(rt, "lc_coll_anchoring_w2", rt_err_level);
      rt_double_parameter(rt, "lc_coll_anchoring_w1", &coll->w1);
      rt_double_parameter(rt, "lc_coll_anchoring_w2", &coll->w2);
      break;
    default:
      /* Not valid. */
      rt_vinfo(rt, rt_err_level, "%s: %s\n",
	       "Input key `lc_coll_anchoring` had invalid value", atype);
      ierr += 1;
    }
  }

  return ierr;
}

/*****************************************************************************
 *
 *  blue_phase_rt_wall_anchoring
 *
 *  Newer style anchoring input which is documented (unlike the old type).
 *
 *****************************************************************************/

int blue_phase_rt_wall_anchoring(pe_t * pe, rt_t * rt, rt_enum_t rt_err_level,
				 lc_anchoring_param_t * wall) {

  assert(pe);
  assert(rt);
  assert(wall);

  /* No wall at all is fine; return 0. */

  int ierr = 0;
  char atype[BUFSIZ] = {0};

  if (rt_string_parameter(rt, "lc_wall_anchoring", atype, BUFSIZ)) {
    wall->type = lc_anchoring_type_from_string(atype);

    switch (wall->type) {
    case LC_ANCHORING_NORMAL:
      ierr += rt_key_required(rt, "lc_wall_anchoring_w1", rt_err_level);
      rt_double_parameter(rt, "lc_wall_anchoring_w1", &wall->w1);
      break;
    case LC_ANCHORING_PLANAR:
      ierr += rt_key_required(rt, "lc_wall_anchoring_w1", rt_err_level);
      ierr += rt_key_required(rt, "lc_wall_anchoring_w2", rt_err_level);
      rt_double_parameter(rt, "lc_wall_anchoring_w1", &wall->w1);
      rt_double_parameter(rt, "lc_wall_anchoring_w2", &wall->w2);
      break;
    case LC_ANCHORING_FIXED:
      ierr += rt_key_required(rt, "lc_wall_anchoring_w1", rt_err_level);
      ierr += rt_key_required(rt, "lc_wall_fixed_orientation", rt_err_level);
      rt_double_parameter(rt, "lc_wall_anchoring_w1", &wall->w1);
      rt_double_parameter_vector(rt, "lc_wall_fixed_orientation", wall->nfix);

      /* Make sure this is a vlaid unit vector here */
      {
	double x2 = wall->nfix[X]*wall->nfix[X];
	double y2 = wall->nfix[Y]*wall->nfix[Y];
	double z2 = wall->nfix[Z]*wall->nfix[Z];
	if (fabs(x2 + y2 + z2) < DBL_EPSILON) {
	  ierr += 1;
	  rt_vinfo(rt, rt_err_level, "%s'n",
		   "lc_wall_fixed_orientation must be non-zero\n");
	}
	wall->nfix[X] /= sqrt(x2 + y2 + z2);
	wall->nfix[Y] /= sqrt(x2 + y2 + z2);
	wall->nfix[Z] /= sqrt(x2 + y2 + z2);
      }
      break;
    default:
      /* Not valid. */
      rt_vinfo(rt, rt_err_level, "%s: %s\n",
	       "Input key `lc_wall_anchoring` had invalid value", atype);
      ierr += 1;
    }
  }

  return ierr;
}
