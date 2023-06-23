/*****************************************************************************
 *
 *  stats_colloid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2023 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_STATS_COLLOID_H
#define LUDWIG_STATS_COLLOID_H

#include "colloids.h"

int stats_colloid_momentum(colloids_info_t * cinfo, double g[3]);
int stats_colloid_velocity_minmax(colloids_info_t * cinfo);
int stats_colloid_write_velocities(pe_t * pe, colloids_info_t * cinfo);
int stats_colloid_write_info(pe_t * pe, colloids_info_t * cinfo, const double t);

#endif
