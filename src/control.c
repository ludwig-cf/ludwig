/*****************************************************************************
 *
 *  control.c
 *
 *  Model control and time stepping.
 *
 *  $Id: control.c,v 1.10 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  end Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "control.h"

static int t_start   = 0;
static int t_steps   = 0;
static int t_current = 0;

static int freq_statistics = 100;
static int freq_measure    = 1000;
static int freq_config     = 10000;
static int freq_phi        = 100000000;
static int freq_vel        = 100000000;
static int freq_shear_io   = 100000000;
static int freq_shear_meas = 100000000;
static int config_at_end   = 1;
static int propagation_ode = 0;


/*****************************************************************************
 *
 *  init_control
 *
 *  Look at the user input and set time step information.
 *
 *****************************************************************************/

void init_control() {

  int n;
  char tmp[128];

  n = RUN_get_int_parameter("N_start", &t_start);
  n = RUN_get_int_parameter("N_cycles", &t_steps);

  n = RUN_get_string_parameter("propagation_ode", tmp, 128);
  if (n ==1 && strcmp(tmp, "on") == 0) propagation_ode = 1;

  n = RUN_get_int_parameter("freq_statistics", &freq_statistics);
  n = RUN_get_int_parameter("freq_measure", &freq_measure);
  n = RUN_get_int_parameter("freq_config", &freq_config);
  n = RUN_get_int_parameter("freq_phi", &freq_phi);
  n = RUN_get_int_parameter("freq_vel", &freq_vel);
  n = RUN_get_int_parameter("freq_shear_measurement", &freq_shear_meas);
  n = RUN_get_int_parameter("freq_shear_output", &freq_shear_io);
  n = RUN_get_string_parameter("config_at_end", tmp, 128);
  if (strcmp(tmp, "no") == 0) config_at_end = 0;

  t_current = t_start;

  if (freq_statistics < 1) freq_statistics = t_start + t_steps + 1;
  if (freq_measure    < 1) freq_measure    = t_start + t_steps + 1;
  if (freq_config     < 1) freq_config     = t_start + t_steps + 1;

  if (freq_shear_io   < 1) freq_shear_io   = t_start + t_steps + 1;
  if (freq_shear_meas < 1) freq_shear_meas = t_start + t_steps + 1;

  return;
}

/*****************************************************************************
 *
 *  get_step
 *  next step
 *
 *****************************************************************************/

int get_step() {
  return t_current;
}

int next_step() {
  ++t_current;
  return (t_start + t_steps - t_current + 1);
}

/*****************************************************************************
 *
 *  is_statistics_step
 *
 *****************************************************************************/

int is_statistics_step() {
  return ((t_current % freq_statistics) == 0);
}

int is_measurement_step() {
  return ((t_current % freq_measure) == 0);
}

int is_config_step() {
  return ((t_current % freq_config) == 0);
}

/*****************************************************************************
 *
 *  is_phi_output_step
 *
 *****************************************************************************/

int is_phi_output_step() {
  return ((t_current % freq_phi) == 0);
}

/*****************************************************************************
 *
 *  is_vel_output_step
 *
 *****************************************************************************/

int is_vel_output_step() {
  return ((t_current % freq_vel) == 0);
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
  return ((t_current % freq_shear_meas) == 0);
}

/*****************************************************************************
 *
 *  is_shear_output_step
 *
 *****************************************************************************/

int is_shear_output_step() {
  return ((t_current % freq_shear_io) == 0);
}

/*****************************************************************************
 *
 *  is_propagation_ode
 *
 *****************************************************************************/

int is_propagation_ode() {
  return propagation_ode;
}

