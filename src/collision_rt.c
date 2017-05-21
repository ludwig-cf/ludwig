/*****************************************************************************
 *
 *  collision_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
 *  TODO: Subsume into lb_distribution_rt
 *
 ****************************************************************************/

int collision_run_time(pe_t * pe, rt_t * rt, lb_t * lb, noise_t * noise) {

  int p;
  int noise_on = 0;
  int nghost;
  char relax[10];
  char tmp[BUFSIZ];
  double tau[NVEL];

  assert(pe);
  assert(rt);
  assert(lb);

  p = rt_string_parameter(rt, "isothermal_fluctuations", tmp, BUFSIZ);

  if (p == 1 && strcmp(tmp, "on") == 0) {
    noise_on = 1;
    noise_present_set(noise, NOISE_RHO, noise_on);
  }

  /* Relaxation time scheme (default M10) */
  
  strcpy(relax, "M10");
  p = rt_string_parameter(rt, "lb_relaxation_scheme", tmp, BUFSIZ);

  if (p == 1) {
    if (strcmp(tmp, "m10") == 0 || strcmp(tmp, "M10") == 0) {
      lb_collision_relaxation_set(lb, LB_RELAXATION_M10);
    }
    else if (strcmp(tmp, "trt") == 0 || strcmp(tmp, "TRT") == 0) {
      strcpy(relax, "TRT");
      lb_collision_relaxation_set(lb, LB_RELAXATION_TRT);
    }
    else if (strcmp(tmp, "bgk") == 0 || strcmp(tmp, "BGK") == 0) {
      strcpy(relax, "BGK");
      lb_collision_relaxation_set(lb, LB_RELAXATION_BGK);
    }
    else {
      pe_fatal(pe, "Unrecognised relaxation time key %s\n", tmp);
    }
  }

  /* Ghost modes */

  p = rt_string_parameter(rt, "ghost_modes", tmp, BUFSIZ);
  nghost = 1;
  if (p == 1 && strcmp(tmp, "off") == 0) {
    nghost = 0;
    lb_collision_ghost_modes_off(lb);
  }

  lb_collision_relaxation_times(lb, tau);

  pe_info(pe, "\n");
  pe_info(pe, "Lattice Boltzmann collision\n");
  pe_info(pe, "---------------------------\n");
#ifndef OLD_SHIT
  /* Need to update test references if have ... */
#else
  pe_info(pe, "Relaxation time scheme:   %s\n", relax);
#endif
  pe_info(pe, "Hydrodynamic modes:       on\n");
  pe_info(pe, "Ghost modes:              %s\n", (nghost == 1) ? "on" : "off");
  pe_info(pe, "Isothermal fluctuations:  %s\n", (noise_on == 1) ? "on" : "off");
  pe_info(pe, "Shear relaxation time:   %12.5e\n", tau[LB_TAU_SHEAR]);
  pe_info(pe, "Bulk relaxation time:    %12.5e\n", tau[LB_TAU_BULK]);
  pe_info(pe, "Ghost relaxation time:   %12.5e\n", tau[NVEL-1]);

  return 0;
}
