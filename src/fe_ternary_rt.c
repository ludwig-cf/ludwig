/****************************************************************************
 *
 *  fe_ternary_rt.c
 *
 *  Run time initialisation for the surfactant free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "fe_ternary.h"
#include "fe_ternary_rt.h"
#include "field_s.h"
#include "field_ternary_init.h"

int field_init_combine_insert(field_t * array, field_t * scalar, int nfin);

/****************************************************************************
 *
 *  fe_ternary_param_rt
 *
 ****************************************************************************/

__host__ int fe_ternary_param_rt(pe_t * pe, rt_t * rt,
				 fe_ternary_param_t * p) {
  assert(pe);
  assert(rt);
  assert(p);

  /* Parameters */
    
  rt_double_parameter(rt, "ternary_kappa1", &p->kappa1);
  rt_double_parameter(rt, "ternary_kappa2", &p->kappa2);
  rt_double_parameter(rt, "ternary_kappa3", &p->kappa3);
  rt_double_parameter(rt, "ternary_alpha",  &p->alpha);
    
  /* For the surfactant should have... */

  if (p->kappa1 < 0.0) pe_fatal(pe, "Please use ternary_kappa1 >= 0\n");
  if (p->kappa2 < 0.0) pe_fatal(pe, "Please use ternary_kappa2 >= 0\n");
  if (p->kappa3 < 0.0) pe_fatal(pe, "Please use ternary_kappa3 >= 0\n");
  if (p->alpha <= 0.0) pe_fatal(pe, "Please use ternary_alpha > 0\n");

  return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_init_rt
 *
 *  Initialise fields: phi, psi. These are related.
 *
 *****************************************************************************/

__host__ int fe_ternary_init_rt(pe_t * pe, rt_t * rt, fe_ternary_t * fe,
				field_t * phi) {
  int p;
  char value[BUFSIZ];
    
  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  p = rt_string_parameter(rt, "ternary_phi_psi_initialisation", value, BUFSIZ);

  if (p != 0 && strcmp(value, "ternary_X") == 0) {
    field_ternary_init_X(phi);
  }

  if (p != 0 && strcmp(value, "ternary_XY") == 0) {
    field_ternary_init_XY(phi);
  }

  if (p != 0 && strcmp(value, "ternary_bbb") == 0) {
    field_ternary_init_bbb(phi);
  }

  if (p != 0 && strcmp(value, "ternary_ggg") == 0) {
    field_ternary_init_ggg(phi);
  }
    
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
