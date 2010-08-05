/*****************************************************************************
 *
 *  collision_rt.c
 *
 *  $Id: collision_rt.c,v 1.1.2.1 2010-08-05 17:23:10 kevin Exp $
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

void collision_run_time(void) {

  int p;
  int noise;
  int nghost;
  char tmp[128];
  double tau[NVEL];

  p = RUN_get_string_parameter("isothermal_fluctuations", tmp, 128);
  noise = 0;
  if (p == 1 && strcmp(tmp, "on") == 0) {
    noise = 1;
    collision_fluctuations_on();
  }

  /* Ghost modes */

  p = RUN_get_string_parameter("ghost_modes", tmp, 128);
  nghost = 1;
  if (p == 1 && strcmp(tmp, "off") == 0) {
    nghost = 0;
    collision_ghost_modes_off();
  }

  collision_relaxation_times_set();
  collision_relaxation_times(tau);

  info("\n");
  info("Lattice Boltzmann collision\n");
  info("---------------------------\n");
  info("Hydrodynamic modes:       on\n");
  info("Ghost modes:              %s\n", (nghost == 1) ? "on" : "off");
  info("Isothermal fluctuations:  %s\n", (noise == 1) ? "on" : "off");
  info("Shear relaxation time:   %12.5e\n", tau[1 + NDIM]);
  info("Bulk relaxation time:    %12.5e\n", tau[1 + NDIM + 1]);
  info("Ghost relaxation time:   %12.5e\n", tau[NVEL-1]);

  return;
}
