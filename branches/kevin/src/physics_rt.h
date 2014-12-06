/****************************************************************************
 *
 *  physics_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2013 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHYSICS_RT_H
#define PHYSICS_RT_H

#include "runtime.h"
#include "physics.h"

int physics_info(physics_t * phys);
int physics_init_rt(rt_t * rt, physics_t * phys);

#endif
