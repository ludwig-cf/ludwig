/*****************************************************************************
 *
 *  polar_active_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef POLAR_ACTIVE_RT_H
#define POLAR_ACTIVE_RT_H

#include "pe.h"
#include "runtime.h"
#include "polar_active.h"

int polar_active_run_time(pe_t * pe, rt_t * rt, fe_polar_t * fe);
int polar_active_rt_initial_conditions(pe_t * pe, rt_t * rt, field_t * p);
int polar_active_init_aster(field_t * p);

#endif
