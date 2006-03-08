/*****************************************************************************
 *
 *  control.h
 *
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

#endif
