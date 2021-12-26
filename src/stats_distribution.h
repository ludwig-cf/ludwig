/*****************************************************************************
 *
 *  stats_distribution.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2022 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_STATS_DISTRIBUTION_H
#define LUDWIG_STATS_DISTRIBUTION_H

#include "lb_data.h"
#include "map.h"

int stats_distribution_print(lb_t * lb, map_t * map);
int stats_distribution_momentum(lb_t * lb, map_t * map, double g[3]);

#endif
