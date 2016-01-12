/*****************************************************************************
 *
 *  control.h
 *
 *  $Id: control.h,v 1.6 2009-10-08 16:29:59 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef CONTROL_H
#define CONTROL_H

#include "targetDP.h"

__targetHost__ void init_control(void);
__targetHost__ int  get_step(void);
__targetHost__ int  next_step(void);
__targetHost__ int  is_last_step(void);
__targetHost__ int  is_statistics_step(void);
__targetHost__ int  is_psi_resid_step(void);
__targetHost__ int  is_measurement_step(void);
__targetHost__ int  is_config_step(void);
__targetHost__ int  is_config_at_end(void);
__targetHost__ int  is_colloid_io_step(void);

__targetHost__ int  is_phi_output_step(void);
__targetHost__ int  is_psi_output_step(void);
__targetHost__ int  is_vel_output_step(void);
__targetHost__ int  is_fed_output_step(void);
__targetHost__ int  is_shear_measurement_step(void);
__targetHost__ int  is_shear_output_step(void);

__targetHost__ int control_freq_set(int freq);
__targetHost__ int control_time_set(int it);

#endif
