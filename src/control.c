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
 *  (c) 2008-2019 The University of Edinburgh
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
static int freq_phi        = 100000000;
static int freq_psi        = 100000000;
static int freq_vel        = 100000000;
static int freq_fed        = 100000000;
static int freq_shear_io   = 100000000;
static int freq_shear_meas = 100000000;
static int freq_colloid_io = 100000000;
static int rho_nfreq       = 100000000;
static int config_at_end   = 1;

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
  rt_int_parameter(rt, "freq_phi", &freq_phi);
  rt_int_parameter(rt, "freq_psi", &freq_psi);
  rt_int_parameter(rt, "freq_vel", &freq_vel);
  rt_int_parameter(rt, "freq_fed", &freq_fed);
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
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_config) == 0);
}

int is_colloid_io_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_colloid_io) == 0);
}

/*****************************************************************************
 *
 *  is_phi_output_step
 *
 *****************************************************************************/

int is_phi_output_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_phi) == 0);
}

/*****************************************************************************
 *
 *  is_vel_output_step
 *
 *****************************************************************************/

int is_vel_output_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_vel) == 0);
}

/*****************************************************************************
 *
 *  is_psi_output_step
 *
 *****************************************************************************/

int is_psi_output_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_psi) == 0);
}

/*****************************************************************************
 *
 *  is_fed_output_step
 *
 *****************************************************************************/

int is_fed_output_step() {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % freq_fed) == 0);
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

/*****************************************************************************
 *
 *  is_rho_output_step
 *
 *****************************************************************************/

int is_rho_output_step(void) {
  physics_t * phys = NULL;
  physics_ref(&phys);
  return ((physics_control_timestep(phys) % rho_nfreq) == 0);
}
