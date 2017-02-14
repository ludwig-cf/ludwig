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
 *  (c) 2013-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHYSICS_RT_H
#define PHYSICS_RT_H

#include "pe.h"
#include "physics.h"
#include "runtime.h"

__host__ int physics_info(pe_t * pe, physics_t * phys);
__host__ int physics_init_rt(rt_t * rt, physics_t * phys);

#endif
