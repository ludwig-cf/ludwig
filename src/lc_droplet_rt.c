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

#include "pe.h"
#include "runtime.h"
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

__host__ int fe_lc_droplet_run_time(fe_lc_droplet_t * fe) {

  int n;
  fe_lc_droplet_param_t param;

  assert(fe);

  info("\n");
  info("Liquid crystal droplet coupling parameters\n");

  n = RUN_get_double_parameter("lc_droplet_gamma", &param.gamma0);
  if (n == 0) fatal("Please specify lc_droplet_gamma in input\n");

  n = RUN_get_double_parameter("lc_droplet_delta", &param.delta);
  if (n == 0) fatal("Please specify lc_droplet_delta in input\n");

  n = RUN_get_double_parameter("lc_droplet_W", &param.w);
  if (n == 0) fatal("Please specify lc_droplet_W in input\n");
  
  info("Isotropic/LC control gamma0 = %12.5e\n", param.gamma0);
  info("Isotropic/LC control delta  = %12.5e\n", param.delta);
  info("Anchoring parameter  W      = %12.5e\n", param.w);
  
  fe_lc_droplet_param_set(fe, param);

  return 0;
}
  
  
