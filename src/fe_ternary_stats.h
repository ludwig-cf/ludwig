/****************************************************************************
 *
 *  fe_ternary_stats.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_FE_TERNARY_STATS_H
#define LUDWIG_FE_TERNARY_STATS_H

#include "fe_ternary.h"
#include "wall.h"
#include "map.h"

__host__ int fe_ternary_stats_info(fe_ternary_t * fe, wall_t * wall,
				   map_t * map, int nt);

#endif
