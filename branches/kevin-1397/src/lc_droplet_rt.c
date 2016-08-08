/****************************************************************************
 *
 *  lc_droplet_rt.c
 *
 *  Run time initiliasation for the liquid crystal droplet free energy
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2016 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "lc_droplet_rt.h"

/*****************************************************************************
 *
 *  fe_lc_droplet_run_time
 *
 *  Pick up the liquid crystal droplet specific  parameters from the input.
 *
 *  In addition to this also routines symmetric_runtime and 
 *  blue_phase_runtime are called.
 *
 *****************************************************************************/

int fe_lc_droplet_run_time(pe_t * pe, rt_t * rt, fe_lc_droplet_t * fe) {

  int n;
  fe_lc_droplet_param_t param;

  assert(fe);

  pe_info(pe, "\n");
  pe_info(pe, "Liquid crystal droplet coupling parameters\n");

  n = rt_double_parameter(rt, "lc_droplet_gamma", &param.gamma0);
  if (n == 0) pe_fatal(pe, "Please specify lc_droplet_gamma in input\n");

  n = rt_double_parameter(rt, "lc_droplet_delta", &param.delta);
  if (n == 0) pe_fatal(pe, "Please specify lc_droplet_delta in input\n");

  n = rt_double_parameter(rt, "lc_droplet_W", &param.w);
  if (n == 0) pe_fatal(pe, "Please specify lc_droplet_W in input\n");
  
  pe_info(pe, "Isotropic/LC control gamma0 = %12.5e\n", param.gamma0);
  pe_info(pe, "Isotropic/LC control delta  = %12.5e\n", param.delta);
  pe_info(pe, "Anchoring parameter  W      = %12.5e\n", param.w);
  
  fe_lc_droplet_param_set(fe, param);

  return 0;
}
  
  
