/*****************************************************************************
 *
 *  gelx_rt.c
 *
 *  Run time initialisation for active gel free energy.
 *
 *  $Id: gelx_rt.c,v 1.1.2.4 2010-03-27 11:03:13 kevin Exp $
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
#include "phi.h"
#include "coords.h"
#include "runtime.h"
#include "free_energy_vector.h"
#include "gelx.h"
#include "gelx_rt.h"

/*****************************************************************************
 *
 *  gelx_run_time
 *
 *  Sort out the active gel input parameters.
 *
 *****************************************************************************/

void gelx_run_time(void) {

  int n;
  double a;
  double b;
  double k1;
  double k2;
  double zeta;
  double lambda;

  /* Vector order parameter (nop = 3) and del^2 required. */

  phi_nop_set(3);
  phi_gradient_level_set(2);
  coords_nhalo_set(2);

  info("Gel X free energy selected.\n");
  info("Vector order parameter nop = 3\n");
  info("Requires up to del^2 derivatives so setting nhalo = 2\n");
  info("\n");

  /* PARAMETERS */

  /* Set as required. */

  fe_density_set(gelx_free_energy_density);
  fe_chemical_stress_set(gelx_chemical_stress);

  n = RUN_get_double_parameter("gelx_a", &a);
  n = RUN_get_double_parameter("gelx_b", &b);
  n = RUN_get_double_parameter("gelx_kappa1", &k1);
  n = RUN_get_double_parameter("gelx_kappa2", &k2);
  n = RUN_get_double_parameter("gelx_zeta", &zeta);
  n = RUN_get_double_parameter("gelx_lambda", &lambda);

  info("Parameters:\n");
  info("Bulk parameter A      = %12.5e\n", a);
  info("Bulk parameter B      = %12.5e\n", b);
  info("Interfacial kappa1    = %12.5e\n", k1);
  info("Interfacial kappa2    = %12.5e\n", k2);
  info("Active parameter zeta = %12.5e\n", zeta);
  info("Lambda                = %12.5e\n", lambda);

  gelx_parameters_set(a, b, k1, k2);
  gelx_zeta_set(zeta);

  fe_v_lambda_set(lambda);
  fe_v_molecular_field_set(gelx_molecular_field);

  return;
}
