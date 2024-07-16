/*****************************************************************************
 *
 *  collision_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <string.h>

#include "pe.h"
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

int collision_run_time(pe_t * pe, rt_t * rt, lb_t * lb) {

  int p;
  int nghost;
  char relax[10];
  char tmp[BUFSIZ];
  double tau[NVEL];

  assert(pe);
  assert(rt);
  assert(lb);

  /* Prefer "lb_fluctuations" in the future */

  p = rt_string_parameter(rt, "isothermal_fluctuations", tmp, BUFSIZ);

  if (p == 1 && strcmp(tmp, "on") == 0) {
    lb->param->noise = 1;
    pe_exit(pe, "Please use the key lb_fluctuations instead of "
	    "isothermal_fluctations in the input\n");
  }

  p = rt_switch(rt, "lb_fluctuations");

  if (p == 1) {
    lb->param->noise = 1;
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

  /* Ghost modes (prefer "lb_ghost_modes" in future) */

  p = rt_key_required(rt, "lb_ghost_modes", RT_NONE); /* p = 0 if present */

  if (p == 0) {
    nghost = rt_switch(rt, "lb_ghost_modes");
    if (nghost == 0) lb_collision_ghost_modes_off(lb);
  }
  else {

  p = rt_string_parameter(rt, "ghost_modes", tmp, BUFSIZ);
  nghost = 1;
  if (p == 1 && strcmp(tmp, "off") == 0) {
    nghost = 0;
    lb_collision_ghost_modes_off(lb);
  }

  }
  lb_collision_relaxation_times(lb, tau);

  pe_info(pe, "\n");
  pe_info(pe, "Lattice Boltzmann collision\n");
  pe_info(pe, "---------------------------\n");
  pe_info(pe, "Relaxation time scheme:   %s\n", relax);
  pe_info(pe, "Hydrodynamic modes:       on\n");
  pe_info(pe, "Ghost modes:              %s\n", (nghost == 1) ? "on" : "off");
  pe_info(pe, "Isothermal fluctuations:  %s\n",
	  (lb->param->noise == 1) ? "on" : "off");
  pe_info(pe, "Shear relaxation time:   %12.5e\n", tau[LB_TAU_SHEAR]);
  pe_info(pe, "Bulk relaxation time:    %12.5e\n", tau[LB_TAU_BULK]);
  pe_info(pe, "Ghost relaxation time:   %12.5e\n", tau[NVEL-1]);

  return 0;
}
