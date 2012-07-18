/*****************************************************************************
 *
 *  blue_phase_rt.c
 *
 *  Run time input for blue phase free energy, and related parameters.
 *
 *  $Id: blue_phase_rt.c,v 1.3 2010-11-01 14:52:55 jlintuvu Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "phi.h"
#include "phi_gradients.h"
#include "free_energy.h"
#include "blue_phase.h"
#include "blue_phase_rt.h"

/*****************************************************************************
 *
 *  blue_phase_run_time
 *
 *  Pick up the liquid crystal parameters from the input.
 *
 *****************************************************************************/

void blue_phase_run_time(void) {

  int n;
  double a0, gamma, q0, kappa0, kappa1;
  double xi;
  double zeta;
  double redshift;

  /* Tensor order parameter (nop = 5); del^2 required; */

  phi_nop_set(5);
  phi_gradients_level_set(2);
  coords_nhalo_set(2);

  info("Blue phase free energy selected.\n");
  info("Tensor order parameter nop = 5\n");
  info("Requires up to del^2 derivatives so setting nhalo = 2\n");

  /* PARAMETERS */

  n = RUN_get_double_parameter("lc_a0", &a0);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_gamma", &gamma);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_q0", &q0);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_kappa0", &kappa0);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_kappa1", &kappa1);
  assert(n == 1);
  n = RUN_get_double_parameter("lc_xi", &xi);
  assert(n == 1);

  /* Use a default redshift of 1 */
  redshift = 1.0;
  RUN_get_double_parameter("lc_init_redshift", &redshift);

  /* Use a default zeta (no activity) of 0 */
  zeta = 0.0;
  RUN_get_double_parameter("lc_active_zeta", &zeta);

  info("\n");
  info("Liquid crystal blue phase free energy\n");
  info("Bulk parameter A0:         = %12.5e\n", a0);
  info("Magnitude of order gamma   = %12.5e\n", gamma);
  info("Pitch wavevector q0        = %12.5e\n", q0);
  info("... gives pitch length     = %12.5e\n", 2.0*4.0*atan(1.0)/q0);
  info("Elastic constant kappa0    = %12.5e\n", kappa0);
  info("Elastic constant kappa1    = %12.5e\n", kappa1);

  /* One-constant approximation enforced. */
  assert(kappa0 == kappa1);

  blue_phase_set_free_energy_parameters(a0, gamma, kappa0, q0);
  blue_phase_set_xi(xi);
  blue_phase_redshift_set(redshift);
  blue_phase_set_zeta(zeta);

  info("Effective aspect ratio xi  = %12.5e\n", xi);
  info("Chirality                  = %12.5e\n", blue_phase_chirality());
  info("Reduced temperature        = %12.5e\n",
       blue_phase_reduced_temperature());
  info("Initial redshift           = %12.5e\n", redshift);
  info("LC activity constant zeta  = %12.5e\n", zeta);

  fe_density_set(blue_phase_free_energy_density);
  fe_chemical_stress_set(blue_phase_chemical_stress);

  return;

}
