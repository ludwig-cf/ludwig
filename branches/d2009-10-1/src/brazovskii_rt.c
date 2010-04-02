/****************************************************************************
 *
 *  brazovskii_rt.c
 *
 *  Run time initialisation for Brazovskii free energy.
 *
 *  $Id: brazovskii_rt.c,v 1.1.2.5 2010-04-02 07:56:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "phi.h"
#include "phi_gradients.h"
#include "free_energy.h"
#include "brazovskii.h"
#include "brazovskii_rt.h"

/****************************************************************************
 *
 *  brazovskii_run_time
 *
 ****************************************************************************/

void brazovskii_run_time(void) {

  int n;
  double a;
  double b;
  double c;
  double kappa;

  /* Single order parameter, del^4 phi required */

  phi_nop_set(1);
  phi_gradients_level_set(4);
  coords_nhalo_set(3);

  info("Brazovskii free energy selected.\n");
  info("Single conserved order parameter nop = 1\n");
  info("Requires up to del^4 derivatives so setting nhalo = %1d\n", 3);
  info("\n");

  /* Parameters */

  n = RUN_get_double_parameter("A", &a);
  n = RUN_get_double_parameter("B", &b);
  n = RUN_get_double_parameter("K", &kappa);
  n = RUN_get_double_parameter("C", &c);

  info("Brazovskii free energy parameters:\n");
  info("Bulk parameter A      = %12.5e\n", a);
  info("Bulk parameter B      = %12.5e\n", b);
  info("Ext. parameter C      = %12.5e\n", c);
  info("Surface penalty kappa = %12.5e\n", kappa);

  brazovskii_free_energy_parameters_set(a, b, kappa, c);

  info("Wavelength 2pi/q_0    = %12.5e\n", brazovskii_wavelength());
  info("Amplitude             = %12.5e\n", brazovskii_amplitude());

  /* For stability ... */

  assert(b > 0.0);
  assert(c > 0.0);

  /* Set free energy function pointers. */

  fe_density_set(brazovskii_free_energy_density);
  fe_chemical_potential_set(brazovskii_chemical_potential);
  fe_isotropic_pressure_set(brazovskii_isotropic_pressure);
  fe_chemical_stress_set(brazovskii_chemical_stress);

  return;
}
