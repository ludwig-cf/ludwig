/****************************************************************************
 *
 *  surfactant_rt.c
 *
 *  Run time initialisation for the surfactant free energy.
 *
 *  $Id: surfactant_rt.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2016 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "surfactant.h"
#include "surfactant_rt.h"

/****************************************************************************
 *
 *  fe_surfactant1_run_time
 *
 ****************************************************************************/

__host__ int fe_surfactant1_run_time(field_t * phi, field_grad_t * dphi,
				     fe_surfactant1_t ** pobj) {


  fe_surfactant1_t * fe = NULL;
  fe_surfactant1_param_t param;
  double sigma;
  double xi0;
  double psi_c;

  /* Parameters */

  RUN_get_double_parameter("surf_A", &param.a);
  RUN_get_double_parameter("surf_B", &param.b);
  RUN_get_double_parameter("surf_K", &param.kappa);

  RUN_get_double_parameter("surf_kt", &param.kt);
  RUN_get_double_parameter("surf_epsilon", &param.epsilon);
  RUN_get_double_parameter("surf_beta", &param.beta);
  RUN_get_double_parameter("surf_w", &param.w);

  info("Surfactant free energy parameters:\n");
  info("Bulk parameter A      = %12.5e\n", param.a);
  info("Bulk parameter B      = %12.5e\n", param.b);
  info("Surface penalty kappa = %12.5e\n", param.kappa);

  /* For the surfactant should have... */

  assert(param.kappa > 0.0);
  assert(param.kt > 0.0);
  assert(param.epsilon > 0.0);
  assert(param.beta >= 0.0);
  assert(param.w >= 0.0);

  fe_surfactant1_create(phi, dphi, &fe);
  fe_surfactant1_param_set(fe, param);
  fe_surfactant1_sigma(fe, &sigma);
  fe_surfactant1_xi0(fe, &xi0);
  fe_surfactant1_langmuir_isotherm(fe, &psi_c);


  info("Fluid parameters:\n");
  info("Interfacial tension   = %12.5e\n", sigma);
  info("Interfacial width     = %12.5e\n", xi0);

  info("\n");
  info("Surface adsorption e  = %12.5e\n", param.epsilon);
  info("Surface psi^2 beta    = %12.5e\n", param.beta);
  info("Enthalpic term W      = %12.5e\n", param.w);
  info("Scale energy kT       = %12.5e\n", param.kt);
  info("Langmuir isotherm     = %12.5e\n", psi_c);

  /* Set free energy function pointers. */

  /*
  fe_density_set(surfactant_free_energy_density);
  fe_chemical_potential_set(surfactant_chemical_potential);
  fe_isotropic_pressure_set(surfactant_isotropic_pressure);
  fe_chemical_stress_set(surfactant_chemical_stress);
  */
  assert(0);

  *pobj = fe;

  return 0;
}
