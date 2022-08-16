/****************************************************************************
 *
 *  two_symm_oft_rt.c
 *
 *  Run time initialisation for the temperature-dependent two_symm free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "runtime.h"
#include "two_symm_oft.h"
#include "two_symm_oft_rt.h"

#include "field_s.h"
#include "field_phi_init_rt.h"
#include "field_psi_init_rt.h"
#include "field_temperature_init_rt.h"

#include "gradient_3d_27pt_solid.h"
#include "physics.h"
#include "util.h"

/****************************************************************************
 *
 *  fe_two_symm_oft_param_rt
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_param_rt(pe_t * pe, rt_t * rt, fe_two_symm_oft_param_t * p) {

  assert(pe);
  assert(rt);
  assert(p);
  int have_wetting = 0;
  double theta = 0.0;
  /* Parameters */

  rt_double_parameter(rt, "phi_A",       &p->phi_a);
  rt_double_parameter(rt, "phi_B",       &p->phi_b);
  rt_double_parameter(rt, "phi_kappa0",   &p->phi_kappa0);
  rt_double_parameter(rt, "phi_kappa1",   &p->phi_kappa1);

  rt_double_parameter(rt, "psi_A",       &p->psi_a);
  rt_double_parameter(rt, "psi_B",       &p->psi_b);
  rt_double_parameter(rt, "psi_kappa",   &p->psi_kappa);

/* <-------------- CHECK WETTING PARAMETERS -------------------> */
  /* Uniform wetting */
  rt_double_parameter(rt, "c", &p->c);
  rt_double_parameter(rt, "h", &p->h);

  /* For the two_symm should have... */

  //assert(p->phi_kappa0 > 0.0);
  //assert(p->psi_kappa > 0.0);

  return 0;
}

/*****************************************************************************
 *
 *  fe_two_symm_oft_phi_init_rt
 *
 *  Initialise the composition part of the order parameter.
 *
 *****************************************************************************/

__host__ int fe_two_symm_oft_phi_init_rt(pe_t * pe, rt_t * rt, fe_two_symm_oft_t * fe,
				  field_t * phi) {

  double phi_xi;
  double psi_xi;
  field_phi_info_t param = {0};
  field_t * tmp = NULL;

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  /* Parameters xi0, phi0, phistar */
  fe_two_symm_oft_xi0(fe, &phi_xi, &psi_xi);
  param.xi0 = phi_xi;

  /* Initialise phi via a temporary scalar field */

  field_create(pe, phi->cs, 1, "tmp", &tmp);
  field_init(tmp, 0, NULL);

  field_phi_init_rt(pe, rt, param, tmp);
  field_init_combine_insert(phi, tmp, 0);

  field_free(tmp);

  return 0;
}

/*****************************************************************************
 *
 *  fe_two_symm_oft_psi_init_rt
 *
 *  Note: phi is the full two-component field used by the free energy.
 *
 *****************************************************************************/

__host__ int fe_two_symm_oft_psi_init_rt(pe_t * pe, rt_t * rt, fe_two_symm_oft_t * fe,
				  field_t * phi) {
  double phi_xi;
  double psi_xi;

  field_t * tmp = NULL;
  field_psi_info_t param = {0};

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  /* Parameters xi0, phi0, phistar */
  fe_two_symm_oft_xi0(fe, &phi_xi, &psi_xi);
  param.xi0 = psi_xi;

  /* Initialise two_symm via a temporary field */

  field_create(pe, phi->cs, 1, "tmp", &tmp);
  field_init(tmp, 0, NULL);

  field_psi_init_rt(pe, rt, param, tmp);
  field_init_combine_insert(phi, tmp, 1);

  field_free(tmp);

  return 0;
}

/*****************************************************************************
 *
 *  fe_two_symm_oft_temperature_init_rt
 *
 *****************************************************************************/

__host__
int fe_two_symm_oft_temperature_init_rt(pe_t * pe, rt_t * rt, fe_two_symm_oft_t * fe, field_t * temperature) {

  assert(pe);
  assert(rt);
  assert(fe);
  assert(temperature);

  field_temperature_init_rt(pe, rt, temperature);

  return 0;
}

