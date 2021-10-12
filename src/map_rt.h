/*****************************************************************************
 *
 *  map_rt.h
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_MAP_RT_H
#define LUDWIG_MAP_RT_H

#include "map.h"
#include "runtime.h"

__host__ int map_init_rt(pe_t * pe, cs_t * cs, rt_t * rt, map_t ** map);
__host__ int map_init_porous_media_from_file(pe_t * pe, cs_t * cs, rt_t * rt,
					     map_t ** map);

#endif
