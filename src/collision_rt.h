/*****************************************************************************
 *
 *  collision_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2024 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLISION_RT_H
#define LUDWIG_COLLISION_RT_H

#include "pe.h"
#include "runtime.h"
#include "lb_data.h"

int collision_run_time(pe_t * pe, rt_t * rt, lb_t * lb);

#endif
 
