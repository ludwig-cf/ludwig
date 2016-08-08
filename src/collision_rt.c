/*****************************************************************************
 *
 *  collision_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
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

int collision_run_time(pe_t * pe, rt_t * rt, noise_t * noise) {

  int p;
  int noise_on = 0;
  int nghost;
  char tmp[BUFSIZ];
  double rtau[NVEL];

  p = rt_string_parameter(rt, "isothermal_fluctuations", tmp, BUFSIZ);

  if (p == 1 && strcmp(tmp, "on") == 0) {
    noise_on = 1;
    noise_present_set(noise, NOISE_RHO, noise_on);
  }

  /* Ghost modes */

  p = rt_string_parameter(rt, "ghost_modes", tmp, BUFSIZ);
  nghost = 1;
  if (p == 1 && strcmp(tmp, "off") == 0) {
    nghost = 0;
    collision_ghost_modes_off();
  }

  lb_collision_relaxation_times_set(noise);
  collision_relaxation_times(rtau);

  pe_info(pe, "\n");
  pe_info(pe, "Lattice Boltzmann collision\n");
  pe_info(pe, "---------------------------\n");
  pe_info(pe, "Hydrodynamic modes:       on\n");
  pe_info(pe, "Ghost modes:              %s\n", (nghost == 1) ? "on" : "off");
  pe_info(pe, "Isothermal fluctuations:  %s\n", (noise_on == 1) ? "on" : "off");
  pe_info(pe, "Shear relaxation time:   %12.5e\n", 1.0/rtau[1 + NDIM]);
  pe_info(pe, "Bulk relaxation time:    %12.5e\n", 1.0/rtau[1 + NDIM + 1]);
  pe_info(pe, "Ghost relaxation time:   %12.5e\n", 1.0/rtau[NVEL-1]);

  return 0;
}
