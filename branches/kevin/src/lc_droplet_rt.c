/****************************************************************************
 *
 *  lc_droplet_rt.c
 *
 *  Run time initiliasation for the liquid crystal droplet free energy
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2015 The University of Edinburgh
 *  Contributing authors:
 *    Juho Lintuvuori ()
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "free_energy.h"
#include "lc_droplet.h"
#include "lc_droplet_rt.h"

/*****************************************************************************
 *
 *  lc_droplet_run_time
 *
 *  Pick up the liquid crystal droplet specific  parameters from the input.
 *
 *  In addition to this also routines symmetric_runtime and 
 *  blue_phase_runtime are called.
 *
 *****************************************************************************/

int lc_droplet_run_time(rt_t * rt) {

  int n;
  double gamma0, delta, W;

  assert(rt);

  info("liquid crystal droplet free energy selected.\n");

  n = rt_double_parameter(rt, "lc_droplet_gamma", &gamma0);
  if (n == 0) fatal("Please specify lc_droplet_gamma in input\n");

  n = rt_double_parameter(rt, "lc_droplet_delta", &delta);
  if (n == 0) fatal("Please specify lc_droplet_delta in input\n");

  n = rt_double_parameter(rt, "lc_droplet_W", &W);
  if (n == 0) fatal("Please specify lc_droplet_W in input\n");
  
  info("Parameters:\n");
  info("parameter gamma0      = %12.5e\n", gamma0);
  info("parameter delta       = %12.5e\n", delta);
  info("parameter W           = %12.5e\n", W);
  
  lc_droplet_set_parameters(gamma0, delta, W);

  /* set the free energy function pointers */
  /* currently molecular field is missing */

  fe_density_set(lc_droplet_free_energy_density);
  fe_chemical_potential_set(lc_droplet_chemical_potential);
  fe_chemical_stress_set(blue_phase_antisymmetric_stress);

  return 0;
}
  
  
