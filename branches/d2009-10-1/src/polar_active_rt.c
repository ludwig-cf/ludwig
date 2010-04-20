/*****************************************************************************
 *
 *  polar_active_rt.c
 *
 *  Run time initialisation for active gel free energy.
 *
 *  $Id: polar_active_rt.c,v 1.1.2.3 2010-04-20 08:38:04 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2010)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "phi.h"
#include "phi_gradients.h"
#include "free_energy_vector.h"
#include "polar_active.h"
#include "polar_active_rt.h"

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
  info("Quadratic term a     = %12.5e\n", a);
  info("Quartic term b       = %12.5e\n", b);
  info("Elastic constant k   = %12.5e\n", k1);
  info("Elastic constant klc = %12.5e\n", klc);
  info("Activity zeta        = %12.5e\n", zeta);
  info("Lambda               = %12.5e\n", lambda);

  polar_active_parameters_set(a, b, k1, klc);
  polar_active_zeta_set(zeta);

  fe_density_set(polar_active_free_energy_density);
  fe_chemical_stress_set(polar_active_chemical_stress);
  fe_v_lambda_set(lambda);
  fe_v_molecular_field_set(polar_active_molecular_field);

  return;
}
