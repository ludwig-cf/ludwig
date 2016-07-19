/*****************************************************************************
 *
 *  polar_active_rt.c
 *
 *  Run time initialisation for active gel free energy.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "io_harness.h"
#include "polar_active_rt.h"

static int polar_active_init_code(field_t * p);

/*****************************************************************************
 *
 *  polar_active_run_time
 *
 *  Sort out the active gel input parameters.
 *
 *****************************************************************************/

__host__ int polar_active_run_time(fe_polar_t * fe) {

  fe_polar_param_t param;

  assert(fe);

  info("Polar active free energy selected.\n");

  /* PARAMETERS */

  RUN_get_double_parameter("polar_active_a", &param.a);
  RUN_get_double_parameter("polar_active_b", &param.b);
  RUN_get_double_parameter("polar_active_k", &param.kappa1);
  RUN_get_double_parameter("polar_active_dk", &param.delta);
  param.delta = 0.0; /* Pending molecular field */
  RUN_get_double_parameter("polar_active_klc", &param.kappa2);
  RUN_get_double_parameter("polar_active_zeta", &param.zeta);
  RUN_get_double_parameter("polar_active_lambda", &param.lambda);

  info("\n");

  info("Parameters:\n");
  info("Quadratic term a     = %14.7e\n", param.a);
  info("Quartic term b       = %14.7e\n", param.b);
  info("Elastic constant k   = %14.7e\n", param.kappa1);
  info("Elastic constant dk  = %14.7e\n", param.delta);
  info("Elastic constant klc = %14.7e\n", param.kappa2);
  info("Activity zeta        = %14.7e\n", param.zeta);
  info("Lambda               = %14.7e\n", param.lambda);

  fe_polar_param_set(fe, param);

  return 0;
}

/*****************************************************************************
 *
 *  polar_active_rt_initial_conditions
 *
 *****************************************************************************/

int polar_active_rt_initial_conditions(field_t * p) {

  char key[BUFSIZ];

  assert(p);

  RUN_get_string_parameter("polar_active_initialisation", key, BUFSIZ);

  if (strcmp(key, "from_file") == 0) {
    assert(0);
    /* Read from file */
  }

  if (strcmp(key, "from_code") == 0) {
    info("Initial polar order parameter from code\n");
    polar_active_init_code(p);
  }

  if (strcmp(key, "aster") == 0) {
    info("Initialise standard aster\n");
    polar_active_init_aster(p);
  }

  return 0;
}

/*****************************************************************************
 *
 *  polar_active_rt_code
 *
 *  Initialise P_\alpha as a function of (x,y,z).
 *
 *****************************************************************************/

static int polar_active_init_code(field_t * fp) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];

  double x, y, z;            /* Global coordinates */
  double p[3];               /* Local order parameter */

  assert(fp);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = 1.0*(noffset[X] + ic);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = 1.0*(noffset[Y] + jc);
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = 1.0*(noffset[Z] + kc);

        index = coords_index(ic, jc, kc);

        /* Set p as a function of true position (x,y,z) as required */

        p[X] = 1.0 + 0.0*x;
        p[Y] = 0.0 + 0.0*y;
        p[Z] = 0.0 + 0.0*z; 

	field_vector_set(fp, index, p);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  polar_active_init_aster
 *
 *****************************************************************************/

int polar_active_init_aster(field_t * fp) {

    int nlocal[3];
    int noffset[3];
    int ic, jc, kc, index;

    double p[3];
    double r;
    double x, y, z, x0, y0, z0;

    assert(fp);

    coords_nlocal(nlocal);
    coords_nlocal_offset(noffset);

    x0 = 0.5*L(X);
    y0 = 0.5*L(Y);
    z0 = 0.5*L(Z);

    if (nlocal[Z] == 1) z0 = 0.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      x = 1.0*(noffset[X] + ic - 1);
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	y = 1.0*(noffset[Y] + jc - 1);
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  z = 1.0*(noffset[Z] + kc - 1);

	  p[X] = 0.0;
	  p[Y] = 1.0;
	  p[Z] = 0.0;

	  r = sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
	  if (r > FLT_EPSILON) {
	    p[X] = -(x - x0)/r;
	    p[Y] = -(y - y0)/r;
	    p[Z] = -(z - z0)/r;
	  }
	  index = coords_index(ic, jc, kc);
	  field_vector_set(fp, index, p);
	}
      }
    }

    return 0;
}
