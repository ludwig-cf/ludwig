/****************************************************************************
 *
 *  surfactant_rt.c
 *
 *  Run time initialisation for the surfactant free energy.
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

#include "pe.h"
#include "runtime.h"
#include "surfactant.h"
#include "surfactant_rt.h"

#include "field_s.h"
#include "field_phi_init_rt.h"
#include "field_psi_init_rt.h"

int field_init_combine_insert(field_t * array, field_t * scalar, int nfin);

/****************************************************************************
 *
 *  fe_surfactant1_param_rt
 *
 ****************************************************************************/

__host__ int fe_surf_param_rt(pe_t * pe, rt_t * rt, fe_surf_param_t * p) {

  assert(pe);
  assert(rt);
  assert(p);

  /* Parameters */

  rt_double_parameter(rt, "surf_A",       &p->a);
  rt_double_parameter(rt, "surf_B",       &p->b);
  rt_double_parameter(rt, "surf_kappa",   &p->kappa);

  rt_double_parameter(rt, "surf_kT",      &p->kt);
  rt_double_parameter(rt, "surf_epsilon", &p->epsilon);
  rt_double_parameter(rt, "surf_beta",    &p->beta);
  rt_double_parameter(rt, "surf_W",       &p->w);

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
 *  fe_surf_phi_init_rt
 *
 *  Initialise the composition part of the order parameter.
 *
 *****************************************************************************/

__host__ int fe_surf_phi_init_rt(pe_t * pe, rt_t * rt, fe_surf_t * fe,
				  field_t * phi) {

  field_phi_info_t param = {0};
  field_t * tmp = NULL;

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  /* Parameters xi0, phi0, phistar */
  fe_surf_xi0(fe, &param.xi0);

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
 *  fe_surf_psi_init_rt
 *
 *  Note: phi is the full two-component field used by the free energy.
 *
 *****************************************************************************/

__host__ int fe_surf_psi_init_rt(pe_t * pe, rt_t * rt, fe_surf_t * fe,
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
 *  field_init_combine_insert
 *
 *  Insert scalar field into array field at position nfin
 *
 ****************************************************************************/

int field_init_combine_insert(field_t * array, field_t * scalar, int nfin) {

  int nlocal[3];
  int ic, jc, kc, index;
  double val[2];

  assert(array);
  assert(scalar);
  assert(array->nf == 2);
  assert(scalar->nf == 1);
  assert(nfin <= array->nf);
  
  cs_nlocal(array->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(array->cs, ic, jc, kc);
	field_scalar_array(array, index, val);
	field_scalar(scalar, index, val + nfin);

	field_scalar_array_set(array, index, val);
      }
    }
  }

  return 0;
}
