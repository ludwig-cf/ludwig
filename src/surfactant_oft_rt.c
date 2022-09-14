/****************************************************************************
 *
 *  surfactant_oft_rt.c
 *
 *  Run time initialisation for the temperature-dependent surfactant free energy.
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
#include "surfactant_oft.h"
#include "surfactant_oft_rt.h"
#include "surfactant_rt.h"

#include "field_s.h"
#include "field_phi_init_rt.h"
#include "field_psi_init_rt.h"
#include "field_temperature_init_rt.h"

#include "gradient_3d_27pt_solid.h"
#include "physics.h"
#include "util.h"

/****************************************************************************
 *
 *  fe_surfactant_oft_param_rt
 *
 ****************************************************************************/

__host__ int fe_surf_oft_param_rt(pe_t * pe, rt_t * rt, fe_surf_oft_param_t * p) {

  assert(pe);
  assert(rt);
  assert(p);
  int have_wetting = 0;
  double theta = 0.0;
  /* Parameters */

  rt_double_parameter(rt, "surf_oft_A",       &p->a);
  rt_double_parameter(rt, "surf_oft_B",       &p->b);
  rt_double_parameter(rt, "surf_oft_kappa",   &p->kappa);
  rt_double_parameter(rt, "surf_oft_kappa1",   &p->kappa1);
  rt_double_parameter(rt, "surf_oft_kappa2",   &p->kappa2);

  rt_double_parameter(rt, "surf_oft_kT",      &p->kt);
  rt_double_parameter(rt, "surf_oft_epsilon", &p->epsilon);
  rt_double_parameter(rt, "surf_oft_beta",    &p->beta);
  rt_double_parameter(rt, "surf_oft_W",       &p->w);


/* <-------------- CHECK WETTING PARAMETERS -------------------> */
  /* Uniform wetting */
  rt_double_parameter(rt, "surf_oft_c", &p->c);
  rt_double_parameter(rt, "surf_oft_h", &p->h);

  /* For the surfactant should have... */

  assert(p->kappa > 0.0);
  assert(p->kt > 0.0);
  assert(p->epsilon > 0.0);
  assert(p->beta >= 0.0);
  assert(p->w >= 0.0);

  return 0;
}

/*****************************************************************************
 *
 *  fe_surf_oft_phi_init_rt
 *
 *  Initialise the composition part of the order parameter.
 *
 *****************************************************************************/

__host__ int fe_surf_oft_phi_init_rt(pe_t * pe, rt_t * rt, fe_surf_oft_t * fe,
				  field_t * phi) {

  field_phi_info_t param = {0};
  field_t * tmp = NULL;

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  /* Parameters xi0, phi0, phistar */
  fe_surf_oft_xi0(fe, &param.xi0);

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
 *  fe_surf_oft_psi_init_rt
 *
 *  Note: phi is the full two-component field used by the free energy.
 *
 *****************************************************************************/

__host__ int fe_surf_oft_psi_init_rt(pe_t * pe, rt_t * rt, fe_surf_oft_t * fe,
				  field_t * phi) {
  field_t * tmp = NULL;
  field_psi_info_t param = {0};

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  /* Initialise surfactant via a temporary field */

  field_create(pe, phi->cs, 1, "tmp", &tmp);
  field_init(tmp, 0, NULL);

  field_psi_init_rt(pe, rt, param, tmp);
  field_init_combine_insert(phi, tmp, 1);

  field_free(tmp);

  return 0;
}

/*****************************************************************************
 *
 *  fe_surf_oft_temperature_init_rt
 *
 *****************************************************************************/

__host__
int fe_surf_oft_temperature_init_rt(pe_t * pe, rt_t * rt, fe_surf_oft_t * fe, field_t * temperature) {

  assert(pe);
  assert(rt);
  assert(fe);
  assert(temperature);

  field_temperature_init_rt(pe, rt, temperature);

  return 0;
}

