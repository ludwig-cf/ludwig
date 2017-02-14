/*****************************************************************************
 *
 *  wall_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_WALL_RT_H
#define LUDWIG_WALL_RT_H

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "model.h"
#include "map.h"
#include "wall.h"

int wall_rt_init(pe_t * pe, cs_t * cs, rt_t * rt, lb_t * lb, map_t * map,
		 wall_t ** wall);

#endif
