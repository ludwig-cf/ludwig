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
 *  $Id: phi_update_rt.c,v 1.1.2.3 2010-03-30 14:22:30 kevin Exp $
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
#include "phi_force.h"
#include "phi_update.h"
#include "phi_cahn_hilliard.h"
#include "leslie_ericksen.h"
#include "blue_phase_beris_edwards.h"

/*****************************************************************************
 *
 *  phi_update_runtime
 *
 *  This could be split up if gets too ponderous.
 *
 *****************************************************************************/

void phi_update_run_time(void) {

  int n, p;
  char stringfe[FILENAME_MAX];
  char string[FILENAME_MAX];
  double value;

  value = 0.0;
  n = RUN_get_string_parameter("free_energy", stringfe, FILENAME_MAX);

  if (n == 0 || strcmp(stringfe, "none") == 0) {
    /* No order parameter, no update. */
    phi_force_required_set(0);
  }
  else {

    info("\n");
    info("Order parameter dynamics\n");
    info("------------------------\n\n");

    if (strcmp(stringfe, "symmetric") == 0) {
      /* Check if we're using full LB */
      RUN_get_string_parameter("phi_finite_difference", string,
			       FILENAME_MAX);
      if (strcmp(string, "yes") == 0) {
	phi_update_set(phi_cahn_hilliard);
	info("Using Cahn-Hilliard finite difference solver:\n");
      }
      else {
	info("Using full lattice Boltzmann solver for Cahn-Hilliard:\n");
	/* Binary LB uses Swift et al method for force. */
	phi_force_required_set(0);
      }

      /* Mobility (always required) */
      RUN_get_double_parameter("mobility", &value);
      phi_ch_set_mobility(value);
      info("Mobility M            = %12.5e\n", phi_ch_get_mobility());
    }
    else if (strcmp(stringfe, "brazovskii") == 0) {

      info("Using Cahn-Hilliard solver:\n");
      phi_update_set(phi_cahn_hilliard);

      RUN_get_double_parameter("mobility", &value);
      phi_ch_set_mobility(value);
      info("Mobility M            = %12.5e\n", phi_ch_get_mobility());
    }
    else if (strcmp(stringfe, "gelx") == 0) {

      info("Using Leslie-Ericksen solver:\n");
      phi_update_set(leslie_ericksen_update);

      RUN_get_double_parameter("gelx_gamma", &value);
      leslie_ericksen_gamma_set(value);
      info("Rotational diffusion     = %12.5e\n", value);

      RUN_get_double_parameter("gelx_swim", &value);
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

  }

  return;
}
