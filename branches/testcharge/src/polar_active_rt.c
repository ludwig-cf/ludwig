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
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "phi.h"
#include "phi_gradients.h"
#include "io_harness.h"
#include "free_energy_vector.h"
#include "polar_active.h"
#include "polar_active_rt.h"

static void polar_active_rt_init(void);

/*****************************************************************************
 *
 *  polar_active_run_time
 *
 *  Sort out the active gel input parameters.
 *
 *****************************************************************************/

void polar_active_run_time(void) {

  int n;
  double a;
  double b;
  double k1;
  double delta;
  double klc;
  double zeta;
  double lambda;

  /* Vector order parameter (nop = 3) and del^2 required. */

  phi_nop_set(3);
  phi_gradients_level_set(2);
  coords_nhalo_set(2);

  /* PARAMETERS */

  n = RUN_get_double_parameter("polar_active_a", &a);
  n = RUN_get_double_parameter("polar_active_b", &b);
  n = RUN_get_double_parameter("polar_active_k", &k1);
  n = RUN_get_double_parameter("polar_active_dk", &delta);
  delta = 0.0; /* Pending molecular field */
  n = RUN_get_double_parameter("polar_active_klc", &klc);
  n = RUN_get_double_parameter("polar_active_zeta", &zeta);
  n = RUN_get_double_parameter("polar_active_lambda", &lambda);

  info("Polar active free energy selected.\n");
  info("Vector order parameter nop = 3\n");
  info("Requires up to del^2 derivatives so setting nhalo = 2\n");

  phi_gradients_dyadic_set(1);
  info("Requires dyadic term in gradients\n");

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

void polar_active_rt_initial_conditions(void) {

  char key[FILENAME_MAX];
  io_info_t * iohandler = NULL;

  assert(phi_nop() == 3);

  RUN_get_string_parameter("polar_active_initialisation", key, FILENAME_MAX);

  if (strcmp(key, "from_file") == 0) {
    phi_io_info(&iohandler);
    assert(iohandler);
    info("Initial polar order parameter requested from file\n");
    info("Reading with serial file stub phi-init\n");
    io_info_set_processor_independent(iohandler);
    io_read("phi-init", iohandler);
    io_info_set_processor_dependent(iohandler);
  }

  if (strcmp(key, "from_code") == 0) {
    info("Initial polar order parameter from code\n");
    polar_active_rt_init();
  }

  return;
}

/*****************************************************************************
 *
 *  polar_active_rt_code
 *
 *  Initialise P_\alpha as a function of (x,y,z).
 *
 *****************************************************************************/

static void polar_active_rt_init(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];

  double x, y, z;            /* Global coordinates */
  double p[3];               /* Local order parameter */

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

        p[X] = 1.0;
        p[Y] = 0.0;
        p[Z] = 0.0; 

        phi_vector_set(index, p);
      }
    }
  }

  return;
}
