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

#include "field_phi_init_rt.h"
#include "brazovskii_rt.h"

/****************************************************************************
 *
 *  fe_brazovskii_init_rt
 *
 ****************************************************************************/

__host__
int fe_brazovskii_init_rt(pe_t * pe, rt_t * rt, fe_brazovskii_t * fe) {

  fe_brazovskii_param_t param;

  double amplitude;
  double lambda;

  assert(pe);
  assert(rt);
  assert(fe);

  pe_info(pe, "Brazovskii free energy selected.\n");
  pe_info(pe, "\n");

  /* Parameters */

  rt_double_parameter(rt, "A", &param.a);
  rt_double_parameter(rt, "B", &param.b);
  rt_double_parameter(rt, "K", &param.kappa);
  rt_double_parameter(rt, "C", &param.c);

  pe_info(pe, "Brazovskii free energy parameters:\n");
  pe_info(pe, "Bulk parameter A      = %12.5e\n", param.a);
  pe_info(pe, "Bulk parameter B      = %12.5e\n", param.b);
  pe_info(pe, "Ext. parameter C      = %12.5e\n", param.c);
  pe_info(pe, "Surface penalty kappa = %12.5e\n", param.kappa);

  fe_brazovskii_param_set(fe, param);

  fe_brazovskii_wavelength(fe, &lambda);
  fe_brazovskii_amplitude(fe, &amplitude);

  pe_info(pe, "Wavelength 2pi/q_0    = %12.5e\n", lambda);
  pe_info(pe, "Amplitude             = %12.5e\n", amplitude);

  /* For stability ... */

  assert(param.b > 0.0);
  assert(param.c > 0.0);

  return 0;
}

/*****************************************************************************
 *
 *  fe_brazovskii_phi_init_rt
 *
 *****************************************************************************/

__host__
int fe_brazovskii_phi_init_rt(pe_t * pe, rt_t * rt, fe_brazovskii_t * fe,
			      field_t * phi) {

  field_phi_info_t param = {0};

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  /* Actually no dependency on Brazovskii parameters at the moment. */
  /* Force mean composition to be zero. */

  param.phi0 = 0.0;

  field_phi_init_rt(pe, rt, param, phi);

  return 0;
}
