/****************************************************************************
 *
 *  stats_velocity.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef STATS_VELOCITY_H
#define STATS_VELOCITY_H

#include "hydro.h"
#include "map.h"

int stats_velocity_minmax(hydro_t * hydro, map_t * map);

#endif
