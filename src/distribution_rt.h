/*****************************************************************************
 *
 *  distribution_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef DISTRIBUTION_RT_H
#define DISTRIBUTION_RT_H

#include "pe.h"
#include "runtime.h"
#include "physics.h"
#include "model.h"

int lb_run_time(pe_t * pe, rt_t * rt, lb_t * lb);
int lb_rt_initial_conditions(pe_t * pe, rt_t * rt, lb_t * lb, physics_t * phys);

#endif
