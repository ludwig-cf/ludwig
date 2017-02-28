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
 *  (c) 2009-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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

__host__ int fe_surfactant1_run_time(pe_t * pe, cs_t * cs, rt_t * rt,
				     field_t * phi, field_grad_t * dphi,
				     fe_surfactant1_t ** pobj) {


  fe_surfactant1_t * fe = NULL;
  fe_surfactant1_param_t param;
  double sigma;
  double xi0;
  double psi_c;

  /* Parameters */

  rt_double_parameter(rt, "surf_A", &param.a);
  rt_double_parameter(rt, "surf_B", &param.b);
  rt_double_parameter(rt, "surf_K", &param.kappa);

  rt_double_parameter(rt, "surf_kt", &param.kt);
  rt_double_parameter(rt, "surf_epsilon", &param.epsilon);
  rt_double_parameter(rt, "surf_beta", &param.beta);
  rt_double_parameter(rt, "surf_w", &param.w);

  pe_info(pe, "Surfactant free energy parameters:\n");
  pe_info(pe, "Bulk parameter A      = %12.5e\n", param.a);
  pe_info(pe, "Bulk parameter B      = %12.5e\n", param.b);
  pe_info(pe, "Surface penalty kappa = %12.5e\n", param.kappa);

  /* For the surfactant should have... */

  assert(param.kappa > 0.0);
  assert(param.kt > 0.0);
  assert(param.epsilon > 0.0);
  assert(param.beta >= 0.0);
  assert(param.w >= 0.0);

  fe_surfactant1_create(pe, cs, phi, dphi, &fe);
  fe_surfactant1_param_set(fe, param);
  fe_surfactant1_sigma(fe, &sigma);
  fe_surfactant1_xi0(fe, &xi0);
  fe_surfactant1_langmuir_isotherm(fe, &psi_c);


  pe_info(pe, "Fluid parameters:\n");
  pe_info(pe, "Interfacial tension   = %12.5e\n", sigma);
  pe_info(pe, "Interfacial width     = %12.5e\n", xi0);

  pe_info(pe, "\n");
  pe_info(pe, "Surface adsorption e  = %12.5e\n", param.epsilon);
  pe_info(pe, "Surface psi^2 beta    = %12.5e\n", param.beta);
  pe_info(pe, "Enthalpic term W      = %12.5e\n", param.w);
  pe_info(pe, "Scale energy kT       = %12.5e\n", param.kt);
  pe_info(pe, "Langmuir isotherm     = %12.5e\n", psi_c);

  assert(0);

  *pobj = fe;

  return 0;
}
