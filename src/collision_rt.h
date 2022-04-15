/*****************************************************************************
 *
 *  collision_rt.h
 *
 *  $Id: collision_rt.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2022 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLISION_RT_H
#define LUDWIG_COLLISION_RT_H

#include "pe.h"
#include "runtime.h"
#include "lb_data.h"
#include "noise.h"

int collision_run_time(pe_t * pe, rt_t * rt, lb_t * lb, noise_t * noise);

#endif
 
