/****************************************************************************
 *
 *  symmetric_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef SYMMETRIC_RT_H
#define SYMMETRIC_RT_H

#include "runtime.h"
#include "coords.h"
#include "field.h"

int symmetric_run_time(rt_t * rt);
int symmetric_rt_initial_conditions(rt_t * rt, coords_t * cs, field_t * phi);

#endif
