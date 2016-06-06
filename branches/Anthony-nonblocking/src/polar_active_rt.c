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
 *  (c) 2011 The University of Edinburgh
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
#include "field.h"
#include "field_grad.h"
#include "io_harness.h"
#include "free_energy_vector.h"
#include "polar_active.h"
#include "polar_active_rt.h"

static int polar_active_init_code(field_t * p);

/*****************************************************************************
 *
 *  polar_active_run_time
 *
 *  Sort out the active gel input parameters.
 *
 *****************************************************************************/

void polar_active_run_time(void) {

  double a;
  double b;
  double k1;
  double delta;
  double klc;
  double zeta;
  double lambda;

  info("Polar active free energy selected.\n");

  /* PARAMETERS */

  RUN_get_double_parameter("polar_active_a", &a);
  RUN_get_double_parameter("polar_active_b", &b);
  RUN_get_double_parameter("polar_active_k", &k1);
  RUN_get_double_parameter("polar_active_dk", &delta);
  delta = 0.0; /* Pending molecular field */
  RUN_get_double_parameter("polar_active_klc", &klc);
  RUN_get_double_parameter("polar_active_zeta", &zeta);
  RUN_get_double_parameter("polar_active_lambda", &lambda);

  info("\n");

  info("Parameters:\n");
  info("Quadratic term a     = %14.7e\n", a);
  info("Quartic term b       = %14.7e\n", b);
  info("Elastic constant k   = %14.7e\n", k1);
  info("Elastic constant dk  = %14.7e\n", delta);
  info("Elastic constant klc = %14.7e\n", klc);
  info("Activity zeta        = %14.7e\n", zeta);
  info("Lambda               = %14.7e\n", lambda);

  polar_active_parameters_set(a, b, k1, klc);
  polar_active_zeta_set(zeta);

  fe_density_set(polar_active_free_energy_density);
  fe_chemical_stress_set(polar_active_chemical_stress);
  fe_v_lambda_set(lambda);
  fe_v_molecular_field_set(polar_active_molecular_field);

  return;
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
