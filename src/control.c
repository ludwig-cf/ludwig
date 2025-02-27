/*****************************************************************************
 *
 *  control.c
 *
 *  Model control and time stepping.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  end Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008-2025 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "physics.h"
#include "control.h"

static int freq_statistics = 100;
static int freq_measure    = 100000000;
static int freq_config     = 100000000;
static int freq_shear_io   = 100000000;
static int freq_shear_meas = 100000000;
static int freq_colloid_io = 100000000;
static int rho_nfreq       = 100000000;
static int config_at_end   = 1;
static int nsteps_         = -1;

/*****************************************************************************
 *
 *  init_control
 *
 *  Look at the user input and set time step information.
 *
 *****************************************************************************/

int init_control(pe_t * pe, rt_t * rt) {

  int n;
  int t_start;
  int t_steps;
  int is_wanted;
  char tmp[128];
  physics_t * phys = NULL;

  assert(pe);
  assert(rt);
  physics_ref(&phys);

  /* Care: t_start (in particular) might not appear in the input */
  t_start = 0;
  t_steps = 0;

  rt_int_parameter(rt, "N_start", &t_start);
  n = rt_int_parameter(rt, "N_cycles", &t_steps);
  if (n == 0) pe_fatal(pe, "Please set N_cycles in input\n");

  rt_int_parameter(rt, "freq_statistics", &freq_statistics);

  rt_int_parameter(rt, "freq_measure", &freq_measure);
  rt_int_parameter(rt, "freq_config", &freq_config);
  rt_int_parameter(rt, "freq_shear_measurement", &freq_shear_meas);
  rt_int_parameter(rt, "freq_shear_output", &freq_shear_io);
  rt_int_parameter(rt, "colloid_io_freq", &freq_colloid_io);
  rt_string_parameter(rt, "config_at_end", tmp, 128);
  if (strcmp(tmp, "no") == 0) config_at_end = 0;

  is_wanted = rt_switch(rt, "rho_io_wanted");
  if (is_wanted) rt_int_parameter(rt, "rho_io_freq", &rho_nfreq);

  physics_control_init_time(phys, t_start, t_steps);

  if (freq_statistics < 1) freq_statistics = t_start + t_steps + 1;
  if (freq_measure    < 1) freq_measure    = t_start + t_steps + 1;
  if (freq_config     < 1) freq_config     = t_start + t_steps + 1;

  if (freq_shear_io   < 1) freq_shear_io   = t_start + t_steps + 1;
  if (freq_shear_meas < 1) freq_shear_meas = t_start + t_steps + 1;

  /* This is a record of the last time step for "config_at_end" */
  nsteps_ = t_start + t_steps;

  /* All these keys are schemed for replacement by a more
   * flexible mechanism. In particular ...*/

  if (rt_key_present(rt, "freq_phi")) {
    pe_info(pe, "Input file contains key: freq_phi\n");
    pe_info(pe, "Please use phi_io_freq instead for order parameter output\n");
    pe_info(pe, "See https://ludwig.epcc.ed.ac.uk/outputs/fluid.html\n");
    pe_exit(pe, "Please check and try again\n");
  }

  if (rt_key_present(rt, "freq_psi")) {
    pe_info(pe, "Input file contains key: freq_psi\n");
    pe_info(pe, "Please use psi_io_freq instead for electrokinectic output\n");
    pe_info(pe, "See https://ludwig.epcc.ed.ac.uk/outputs/fluid.html\n");
    pe_exit(pe, "Please check and try again\n");
  }

  if (rt_key_present(rt, "freq_vel")) {
    pe_info(pe, "Input file contains key: freq_vel\n");
    pe_info(pe, "Please use vel_io_freq instead for velocity field output\n");
    pe_info(pe, "See https://ludwig.epcc.ed.ac.uk/outputs/fluid.html\n");
    pe_exit(pe, "Please check and try again\n");
  }

  if (rt_key_present(rt, "freq_fed")) {
    pe_info(pe, "Input file contains key: freq_fed\n");
    pe_info(pe, "Lattice free enegy density output is not implemented\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  is_statistics_step
 *
 *****************************************************************************/

int is_statistics_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_statistics) == 0);
}

int is_measurement_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_measure) == 0);
}

int is_config_step() {
  int t = -1;
  int isconfigatendstep = 0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  t = physics_control_timestep(phys);
  isconfigatendstep = (config_at_end && (t == nsteps_));

  return ((t % freq_config) == 0 || isconfigatendstep);
}

int is_colloid_io_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_colloid_io) == 0);
}

/*****************************************************************************
 *
 *  is_config_at_end
 *
 *****************************************************************************/

int is_config_at_end() {
  return config_at_end;
}

/*****************************************************************************
 *
 *  is_shear_measurement_step
 *
 *****************************************************************************/

int is_shear_measurement_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_shear_meas) == 0);
}

/*****************************************************************************
 *
 *  is_shear_output_step
 *
 *****************************************************************************/

int is_shear_output_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_shear_io) == 0);
}

/*****************************************************************************
 *
 *  control_freq_set
 *
 *  Control needs refactoring as object; until that time:
 *
 *****************************************************************************/

int control_freq_set(int freq) {

  freq_statistics = freq;

  return 0;
}
