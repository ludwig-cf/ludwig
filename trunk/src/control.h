/*****************************************************************************
 *
 *  control.h
 *
 *  $Id: control.h,v 1.5 2009-04-09 14:53:29 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _CONTROL_H
#define _CONTROL_H

void init_control(void);
int  get_step(void);
int  next_step(void);
int  is_statistics_step(void);
int  is_measurement_step(void);
int  is_config_step(void);
int  is_config_at_end(void);

int  is_phi_output_step(void);
int  is_vel_output_step(void);

#endif
