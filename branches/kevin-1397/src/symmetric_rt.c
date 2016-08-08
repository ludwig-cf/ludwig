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
 *  (c) 2010-2016 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>


#include "field_phi_init_rt.h"
#include "symmetric_rt.h"

#include "gradient_3d_27pt_solid.h"
#include "physics.h"

/****************************************************************************
 *
 *  fe_symmetric_init_rt
 *
 ****************************************************************************/

int fe_symmetric_init_rt(pe_t * pe, rt_t * rt, fe_symm_t * fe) {

  double sigma;
  double xi;
  fe_symm_param_t param;

  assert(pe);
  assert(rt);
  assert(fe);

  pe_info(pe, "Symmetric phi^4 free energy selected.\n");
  pe_info(pe, "\n");

  /* Parameters */

  rt_double_parameter(rt, "A", &param.a);
  rt_double_parameter(rt, "B", &param.b);
  rt_double_parameter(rt, "K", &param.kappa);

  pe_info(pe, "Parameters:\n");
  pe_info(pe, "Bulk parameter A      = %12.5e\n", param.a);
  pe_info(pe, "Bulk parameter B      = %12.5e\n", param.b);
  pe_info(pe, "Surface penalty kappa = %12.5e\n", param.kappa);

  fe_symm_param_set(fe, param);

  fe_symm_interfacial_tension(fe, &sigma);
  fe_symm_interfacial_width(fe, &xi);

  pe_info(pe, "Surface tension       = %12.5e\n", sigma);
  pe_info(pe, "Interfacial width     = %12.5e\n", xi);

  /* Initialise */
  grad_3d_27pt_solid_fe_set(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_phi_init_rt
 *
 *****************************************************************************/

__host__
int fe_symmetric_phi_init_rt(pe_t * pe, rt_t * rt, fe_symm_t * fe,
			     field_t * phi) {

  physics_t * phys = NULL;
  field_phi_info_t param = {0};

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  physics_ref(&phys);
  physics_phi0(phys, &param.phi0);

  fe_symm_interfacial_width(fe, &param.xi0);

  field_phi_init_rt(pe, rt, param, phi);

  return 0;
}
