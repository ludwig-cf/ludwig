/****************************************************************************
 *
 *  brazovskii_rt.c
 *
 *  Run time initialisation for Brazovskii free energy.
 *
 *  $Id: brazovskii_rt.c,v 1.2 2010-10-15 12:40:02 kevin Exp $
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
#include "runtime.h"
#include "brazovskii_rt.h"

/****************************************************************************
 *
 *  fe_brazovskii_run_time
 *
 ****************************************************************************/

__host__ int  fe_brazovskii_run_time(fe_brazovskii_t * fe) {

  fe_brazovskii_param_t param;

  double amplitude;
  double lambda;

  assert(fe);

  info("Brazovskii free energy selected.\n");
  info("\n");

  /* Parameters */

  RUN_get_double_parameter("A", &param.a);
  RUN_get_double_parameter("B", &param.b);
  RUN_get_double_parameter("K", &param.kappa);
  RUN_get_double_parameter("C", &param.c);

  info("Brazovskii free energy parameters:\n");
  info("Bulk parameter A      = %12.5e\n", param.a);
  info("Bulk parameter B      = %12.5e\n", param.b);
  info("Ext. parameter C      = %12.5e\n", param.c);
  info("Surface penalty kappa = %12.5e\n", param.kappa);

  fe_brazovskii_param_set(fe, param);

  fe_brazovskii_wavelength(fe, &lambda);
  fe_brazovskii_amplitude(fe, &amplitude);

  info("Wavelength 2pi/q_0    = %12.5e\n", lambda);
  info("Amplitude             = %12.5e\n", amplitude);

  /* For stability ... */

  assert(param.b > 0.0);
  assert(param.c > 0.0);

  return 0;
}

/*****************************************************************************
 *
 *  fe_brazovskii_rt_init_phi
 *
 *****************************************************************************/

__host__ int fe_brazovskii_rt_init_phi(fe_brazovskii_t * fe, field_t * phi) {

  double xi;
  int field_phi_rt(field_t * phi, double xi); /* SHIT sort me out */

  assert(fe);
  assert(phi);

  /* Only the spinodal condition is usually used for Brazovskii */
  xi = -1.0;
  field_phi_rt(phi, xi);

  return 0;
}
