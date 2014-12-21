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
 *  (c) 2011-2015 The University of Edinbrugh
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_RT_H
#define BLUE_PHASE_RT_H

#include "runtime.h"
#include "coords.h"
#include "field.h"

int blue_phase_run_time(rt_t * rt);
int blue_phase_rt_initial_conditions(rt_t * rt, coords_t * cs, field_t * q);

#endif
