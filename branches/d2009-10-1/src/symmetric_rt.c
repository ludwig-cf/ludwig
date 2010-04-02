/****************************************************************************
 *
 *  symmetric_rt.c
 *
 *  Run time initialisation for the symmetric phi^4 free energy.
 *
 *  $Id: symmetric_rt.c,v 1.1.2.4 2010-04-02 07:56:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "phi.h"
#include "phi_gradients.h"
#include "runtime.h"
#include "free_energy.h"
#include "symmetric.h"

/****************************************************************************
 *
 *  symmetric_run_time
 *
 ****************************************************************************/

void symmetric_run_time(void) {

  int n;
  double a;
  double b;
  double kappa;

  /* Single order parameter, del^2 phi required. */

  /* There's a slight complication in that halo width one is enough
   * at the moment when using full LB. */

  phi_nop_set(1);
  phi_gradients_level_set(2);
  coords_nhalo_set(2);

  info("Symmetric phi^4 free energy selected.\n");
  info("Single conserved order parameter nop = 1\n");
  info("Requires up to del^2 derivatives so setting nhalo = 2\n");
  info("\n");

  /* Parameters */

  n = RUN_get_double_parameter("A", &a);
  n = RUN_get_double_parameter("B", &b);
  n = RUN_get_double_parameter("K", &kappa);

  info("Parameters:\n");
  info("Bulk parameter A      = %12.5e\n", a);
  info("Bulk parameter B      = %12.5e\n", b);
  info("Surface penalty kappa = %12.5e\n", kappa);

  symmetric_free_energy_parameters_set(a, b, kappa);

  info("Surface tension       = %12.5e\n", symmetric_interfacial_tension());
  info("Interfacial width     = %12.5e\n", symmetric_interfacial_width());

  /* For the symmetric... */

  assert(kappa > 0.0);

  /* Set free energy function pointers. */

  fe_density_set(symmetric_free_energy_density);
  fe_chemical_potential_set(symmetric_chemical_potential);
  fe_isotropic_pressure_set(symmetric_isotropic_pressure);
  fe_chemical_stress_set(symmetric_chemical_stress);

  return;
}
