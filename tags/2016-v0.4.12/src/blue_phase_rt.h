/*****************************************************************************
 *
 *  blue_phase_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinbrugh
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_RT_H
#define BLUE_PHASE_RT_H

#include "field.h"

int blue_phase_rt_initial_conditions(field_t * q);
void blue_phase_run_time(void);

#endif
