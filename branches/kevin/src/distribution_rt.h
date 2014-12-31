/*****************************************************************************
 *
 *  distribution_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef DISTRIBUTION_RT_H
#define DISTRIBUTION_RT_H

#include "runtime.h"
#include "coords.h"
#include "physics.h"
#include "model.h"

int lb_run_time(lb_t * lb, rt_t * rt, coords_t * cs);
int lb_rt_initial_conditions(lb_t * lb, rt_t * rt, coords_t * cs,
			     physics_t * phys);

#endif
