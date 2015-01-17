/*****************************************************************************
 *
 *  wall_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef WALL_RT_H
#define WALL_RT_H

#include "runtime.h"
#include "coords.h"
#include "model.h"
#include "map.h"
#include "wall.h"

int wall_rt_init(rt_t * rt, coords_t * cs, lb_t * lb, map_t * map,
		 wall_t ** wall);

#endif
