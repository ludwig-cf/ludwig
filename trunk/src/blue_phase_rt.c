/*****************************************************************************
 *
 *  blue_phase_rt.c
 *
 *  Run time input for blue phase free energy, and related parameters.
 *
 *  $Id$
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
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "phi.h"
#include "phi_gradients.h"
#include "colloids_Q_tensor.h"
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
  int redshift_update;
  char method[FILENAME_MAX];
  char type[FILENAME_MAX];
  double a0, gamma, q0, kappa0, kappa1;
  double xi;
  double zeta;
  double w;
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

  redshift_update = 0;
  RUN_get_int_parameter("lc_redshift_update", &redshift_update);

  /* Use a default zeta (no activity) of 0 */
  zeta = 0.0;
  RUN_get_double_parameter("lc_active_zeta", &zeta);

  info("\n");
  info("Liquid crystal blue phase free energy\n");
  info("Bulk parameter A0:         = %14.7e\n", a0);
  info("Magnitude of order gamma   = %14.7e\n", gamma);
  info("Pitch wavevector q0        = %14.7e\n", q0);
  info("... gives pitch length     = %14.7e\n", 2.0*4.0*atan(1.0)/q0);
  info("Elastic constant kappa0    = %14.7e\n", kappa0);
  info("Elastic constant kappa1    = %14.7e\n", kappa1);

  /* One-constant approximation enforced. */
  assert(kappa0 == kappa1);

  blue_phase_set_free_energy_parameters(a0, gamma, kappa0, q0);
  blue_phase_set_xi(xi);
  blue_phase_redshift_set(redshift);
  blue_phase_redshift_update_set(redshift_update);
  blue_phase_set_zeta(zeta);

  info("Effective aspect ratio xi  = %14.7e\n", xi);
  info("Chirality                  = %14.7e\n", blue_phase_chirality());
  info("Reduced temperature        = %14.7e\n",
       blue_phase_reduced_temperature());
  info("Initial redshift           = %14.7e\n", redshift);
  info("Dynamic redshift update    = %14s\n",
       redshift_update == 0 ? "no" : "yes");
  info("LC activity constant zeta  = %14.7e\n", zeta);

  fe_density_set(blue_phase_free_energy_density);
  fe_chemical_stress_set(blue_phase_chemical_stress);

  /* Surface anchoring (default "none") */

  RUN_get_string_parameter("lc_anchoring_method", method, FILENAME_MAX);

  if (strcmp(method, "one") == 0 || strcmp(method, "two") == 0) {

    /* Find out type */

    RUN_get_string_parameter("lc_anchoring", type, FILENAME_MAX);

    if (strcmp(type, "normal") == 0) {
      colloids_q_tensor_anchoring_set(ANCHORING_NORMAL);
    }

    if (strcmp(type, "planar") == 0) {
      colloids_q_tensor_anchoring_set(ANCHORING_PLANAR);
    }

    if (strcmp(type, "fixed") == 0) {
      colloids_q_tensor_anchoring_set(ANCHORING_FIXED);
    }

    if (strcmp(method, "one") == 0) {
      colloids_q_anchoring_method_set(ANCHORING_METHOD_ONE);
    }

    /* Surface free energy parameter (method two only) */

    RUN_get_double_parameter("lc_anchoring_strength", &w);

    info("\n");
    info("Liquid crystal anchoring\n");
    info("Anchoring method:          = %14s\n", method);
    info("Anchoring type:            = %14s\n", type);

    if (strcmp(method, "two") == 0) {
      info("Surface free energy w:     = %14.7e\n", w);
      info("Ratio w/kappa0:            = %14.7e\n", w/kappa0);
      colloids_q_anchoring_method_set(ANCHORING_METHOD_TWO);
      colloids_q_tensor_w_set(w);
    }

  }

  return;

}
