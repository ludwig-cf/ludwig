/****************************************************************************
 *
 *  lc_droplet_rt.c
 *
 *  Run time initiliasation for the liquid crystal droplet free energy
 *
 *  $Id: $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2012 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "free_energy.h"
#include "runtime.h"
#include "pe.h"
#include "lc_droplet.h"

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

void lc_droplet_run_time(void) {

  int n;
  double gamma0, delta, W;

  info("liquid crystal droplet free energy selected.\n");

  n = RUN_get_double_parameter("lc_droplet_gamma", &gamma0);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_droplet_delta", &delta);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_droplet_W", &W);
  assert(n == 1);
  
  info("Parameters:\n");
  info("parameter gamma0      = %12.5e\n", gamma0);
  info("parameter delta       = %12.5e\n", delta);
  info("parameter W           = %12.5e\n", W);
  
  lc_droplet_set_parameters(gamma0, delta, W);

  /* set the free energy function pointers */
  /* currently molecular field is missing */
  fe_density_set(lc_droplet_free_energy_density);
  fe_chemical_potential_set(lc_droplet_chemical_potential);
  fe_chemical_stress_set(lc_droplet_chemical_stress);
  /*fe_molecular_field_set(lc_droplet_molecular_field);*/

  return;
}
  
  
