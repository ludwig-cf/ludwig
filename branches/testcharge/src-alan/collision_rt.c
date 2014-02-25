/*****************************************************************************
 *
 *  collision_rt.c
 *
 *  $Id: collision_rt.c,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <string.h>

#include "pe.h"
#include "model.h"
#include "physics.h"
#include "runtime.h"
#include "collision.h"

/****************************************************************************
 *
 *  collision_run_time
 *
 *  Defaults are noise off and ghosts on.
 *
 *  Note that the fluid properties must be set to get sensible
 *  values out at this stage.
 *
 ****************************************************************************/

int collision_run_time(noise_t * noise) {

  int p;
  int noise_on = 0;
  int nghost;
  char tmp[128];
  double rtau[NVEL];

  p = RUN_get_string_parameter("isothermal_fluctuations", tmp, 128);

  if (p == 1 && strcmp(tmp, "on") == 0) {
    noise_on = 1;
    noise_present_set(noise, NOISE_RHO, noise_on);
  }

  /* Ghost modes */

  p = RUN_get_string_parameter("ghost_modes", tmp, 128);
  nghost = 1;
  if (p == 1 && strcmp(tmp, "off") == 0) {
    nghost = 0;
    collision_ghost_modes_off();
  }

  collision_relaxation_times_set(noise);
  collision_relaxation_times(rtau);

  info("\n");
  info("Lattice Boltzmann collision\n");
  info("---------------------------\n");
  info("Hydrodynamic modes:       on\n");
  info("Ghost modes:              %s\n", (nghost == 1) ? "on" : "off");
  info("Isothermal fluctuations:  %s\n", (noise_on == 1) ? "on" : "off");
  info("Shear relaxation time:   %12.5e\n", 1.0/rtau[1 + NDIM]);
  info("Bulk relaxation time:    %12.5e\n", 1.0/rtau[1 + NDIM + 1]);
  info("Ghost relaxation time:   %12.5e\n", 1.0/rtau[NVEL-1]);

  return 0;
}
