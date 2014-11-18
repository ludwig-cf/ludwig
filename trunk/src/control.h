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

HOST void init_control(void);
HOST int  get_step(void);
HOST int  next_step(void);
HOST int  is_statistics_step(void);
HOST int  is_measurement_step(void);
HOST int  is_config_step(void);
HOST int  is_config_at_end(void);
HOST int  is_colloid_io_step(void);

HOST int  is_phi_output_step(void);
HOST int  is_psi_output_step(void);
HOST int  is_vel_output_step(void);
HOST int  is_fed_output_step(void);
HOST int  is_shear_measurement_step(void);
HOST int  is_shear_output_step(void);

HOST int control_freq_set(int freq);
HOST int control_time_set(int it);

#endif
