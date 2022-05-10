/****************************************************************************
 *
 *  fe_symmetric_ll_rt.c
 *
 *  Run time initialisation for the symmetric_ll free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
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
#include "fe_symmetric_ll.h"
#include "fe_symmetric_ll_rt.h"
#include "field_s.h"
#include "field_symmetric_ll_init.h"

/****************************************************************************
 *
 *  fe_symmetric_ll_param_rt
 *
 ****************************************************************************/

__host__ int fe_symmetric_ll_param_rt(pe_t * pe, rt_t * rt,
				 fe_symmetric_ll_param_t * p) {
  assert(pe);
  assert(rt);
  assert(p);

  /* Parameters */

  rt_key_required(rt, "symmetric_ll_a1", RT_FATAL);
  rt_key_required(rt, "symmetric_ll_b1", RT_FATAL);
  rt_key_required(rt, "symmetric_ll_kappa1", RT_FATAL);

  rt_key_required(rt, "symmetric_ll_a2", RT_FATAL);
  rt_key_required(rt, "symmetric_ll_b2", RT_FATAL);
  rt_key_required(rt, "symmetric_ll_kappa2", RT_FATAL);

  rt_double_parameter(rt, "symmetric_ll_a1", &p->a1);
  rt_double_parameter(rt, "symmetric_ll_b1", &p->b1);
  rt_double_parameter(rt, "symmetric_ll_kappa1", &p->kappa1);

  rt_double_parameter(rt, "symmetric_ll_a2", &p->a2);
  rt_double_parameter(rt, "symmetric_ll_b2", &p->b2);
  rt_double_parameter(rt, "symmetric_ll_kappa2", &p->kappa2);

  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_ll_init_rt
 *
 *  Initialise fields: phi, psi, rho. These are related.
 *
 *****************************************************************************/

__host__ int fe_symmetric_ll_init_rt(pe_t * pe, rt_t * rt, fe_symmetric_ll_t * fe,
				field_t * phi) {
  int p;
  double phi0, psi0;
  char value[BUFSIZ];
    
  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  p = rt_string_parameter(rt, "phi_ll_initialisation", value, BUFSIZ);

  if (p == 0) pe_fatal(pe, "Please specify phi_ll_initialisation");

  pe_info(pe, "\n");
  pe_info(pe, "Initialising PHI[0] for symmetric_ll fluid\n");

  if (p != 0 && strcmp(value, "uniform") == 0) {
    rt_double_parameter(rt, "phi0", &phi0);
    field_symmetric_ll_phi_uniform(phi, phi0);
  }

  p = rt_string_parameter(rt, "psi_ll_initialisation", value, BUFSIZ);
  if (p == 0) pe_fatal(pe, "Please specify psi_ll_initialisation");

  pe_info(pe, "\n");
  pe_info(pe, "Initialising PHI[1] for symmetric_ll fluid\n");

  if (p != 0 && strcmp(value, "uniform") == 0) {
    rt_double_parameter(rt, "psi0", &psi0);
    field_symmetric_ll_psi_uniform(phi, psi0);
  }

  return 0;
}
