/****************************************************************************
 *
 *  stats_velocity.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2020 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef STATS_VELOCITY_H
#define STATS_VELOCITY_H

#include "hydro.h"
#include "map.h"

typedef struct stats_vel_s stats_vel_t;

struct stats_vel_s {
  int print_vol_flux;
};

stats_vel_t stats_vel_default(void);

int stats_velocity_minmax(stats_vel_t * stat, hydro_t * hydro, map_t * map);

#endif
