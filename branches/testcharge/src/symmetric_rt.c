/****************************************************************************
 *
 *  symmetric_rt.c
 *
 *  Run time initialisation for the symmetric phi^4 free energy.
 *
 *  $Id$
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
#include "runtime.h"
#include "free_energy.h"
#include "symmetric.h"

static void symmetric_init(const int nhalo);

/****************************************************************************
 *
 *  symmetric_run_time
 *
 *  This means hybrid, requiring nhalo = 2.
 *
 ****************************************************************************/

void symmetric_run_time(void) {

  symmetric_init(2);
  return;
}

/****************************************************************************
 *
 *  symmetric_run_time_lb
 *
 *  This means full lattice Boltzmann, where nhalo = 1 is enough.
 *
 ****************************************************************************/

void symmetric_run_time_lb(void) {

  symmetric_init(1);
  return;
}

/****************************************************************************
 *
 *  symmetric_run_time_noise
 *
 *  This is hybrid with discretisation following Sumesh et al.
 *  requiring nhalo = 3.
 *
 ****************************************************************************/

void symmetric_run_time_noise(void) {

  symmetric_init(3);

  return;
}

/****************************************************************************
 *
 *  symmetric_init
 *
 ****************************************************************************/

static void symmetric_init(const int nhalo) {

  int n;
  double a;
  double b;
  double kappa;

  info("Symmetric phi^4 free energy selected.\n");
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

  /* Set free energy function pointers. */

  fe_density_set(symmetric_free_energy_density);
  fe_chemical_potential_set(symmetric_chemical_potential);
  fe_isotropic_pressure_set(symmetric_isotropic_pressure);
  fe_chemical_stress_set(symmetric_chemical_stress);

  return;
}
