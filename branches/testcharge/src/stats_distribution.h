/*****************************************************************************
 *
 *  stats_distribution.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_DISTRIBUTION_H
#define STATS_DISTRIBUTION_H

#include "map.h"

int stats_distribution_print(map_t * map);
int stats_distribution_momentum(map_t * map, double g[3]);

#endif
