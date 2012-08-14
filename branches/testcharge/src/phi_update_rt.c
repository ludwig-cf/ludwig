/******************************************************************************
 *
 *  phi_update_rt.c
 *
 *  Initialise the order parameter dynamics update.
 *
 *  There is one feature that is slightly awkward: for full lattice
 *  Boltzmann for binary fluid, no update is set (it's done via the
 *  appropriate collision).
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "phi.h"
#include "phi_force.h"
#include "phi_update.h"
#include "phi_fluctuations.h"
#include "phi_cahn_hilliard.h"
#include "leslie_ericksen.h"
#include "blue_phase_beris_edwards.h"

static void phi_update_rt_fe(const char * string);

/*****************************************************************************
 *
 *  phi_update_runtime
 *
 *  This could be split up if gets too ponderous.
 *
 *****************************************************************************/

void phi_update_run_time(void) {

  int n;
  char stringfe[FILENAME_MAX];

  n = RUN_get_string_parameter("free_energy", stringfe, FILENAME_MAX);

  if (n == 0 || strcmp(stringfe, "none") == 0) {
    /* No order parameter, no update, no force... */
    phi_force_required_set(0);
  }
  else {
    /* Sort out free energy */
    phi_update_rt_fe(stringfe);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_update_rt_fe
 *
 *  Filter the various free energy choices.
 *
 *****************************************************************************/

static void phi_update_rt_fe(const char * stringfe) {

  int p;
  double value;

  info("\n");
  info("Order parameter dynamics\n");
  info("------------------------\n\n");

  value = 0.0;

  if (strcmp(stringfe, "symmetric") == 0 ||
      strcmp(stringfe, "symmetric_noise") == 0) {

    info("Using Cahn-Hilliard finite difference solver:\n");

    phi_update_set(phi_cahn_hilliard);

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());

    p = 0;
    RUN_get_int_parameter("fd_phi_fluctuations", &p);
    info("Order parameter noise = %3s\n", (p == 0) ? "off" : " on");
    if (p != 0) phi_fluctuations_on_set(p);

    p = 1; /* Default is to use divergence method */
    RUN_get_int_parameter("fd_force_divergence", &p);
    info("Force calculation:      %s\n",
         (p == 0) ? "phi grad mu method" : "divergence method");
    phi_force_divergence_set(p);

  }
  else if (strcmp(stringfe, "symmetric_lb") == 0) {
  
    info("Using full lattice Boltzmann solver for Cahn-Hilliard:\n");
    phi_force_required_set(0);

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());
  }
  else if (strcmp(stringfe, "brazovskii") == 0) {

    info("Using Cahn-Hilliard solver:\n");

    phi_update_set(phi_cahn_hilliard);

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());

    p = 1;
    RUN_get_int_parameter("fd_force_divergence", &p);
    info("Force caluclation:      %s\n",
         (p == 0) ? "phi grad mu method" : "divergence method");
    phi_force_divergence_set(p);
  }
  else if (strcmp(stringfe, "polar_active") == 0) {

    info("Using Leslie-Ericksen solver:\n");
    phi_update_set(leslie_ericksen_update);

    RUN_get_double_parameter("leslie_ericksen_gamma", &value);
    leslie_ericksen_gamma_set(value);
    info("Rotational diffusion     = %12.5e\n", value);

    RUN_get_double_parameter("leslie_ericksen_swim", &value);
    leslie_ericksen_swim_set(value);
    info("Self-advection parameter = %12.5e\n", value);
  }
  else if (strcmp(stringfe, "lc_blue_phase") == 0) {

    info("Using Beris-Edwards solver:\n");
    phi_update_set(blue_phase_beris_edwards);

    p = RUN_get_double_parameter("lc_Gamma", &value);
    if (p != 0) {
      blue_phase_be_set_rotational_diffusion(value);
      info("Rotational diffusion constant = %12.5e\n", value);
    }
  }

  return;
}
