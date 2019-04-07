/*****************************************************************************
 *
 *  control.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008-2019 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_CONTROL_H
#define LUDWIG_CONTROL_H

#include "pe.h"
#include "runtime.h"
#include "physics.h"

int init_control(pe_t * pe, rt_t * rt);
int is_statistics_step(void);
int is_measurement_step(void);
int is_config_step(void);
int is_config_at_end(void);
int is_colloid_io_step(void);

int is_phi_output_step(void);
int is_vel_output_step(void);
int is_psi_output_step(void);
int is_rho_output_step(void);
int is_fed_output_step(void);
int is_shear_measurement_step(void);
int is_shear_output_step(void);
int control_freq_set(int freq);

#endif
