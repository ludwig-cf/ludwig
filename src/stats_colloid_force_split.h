/*****************************************************************************
 *
 *  stats_colloid_force_split.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_STATS_COLLOID_FORCE_SPLIT_H
#define LUDWIG_STATS_COLLOID_FORCE_SPLIT_H

#include "colloids.h"
#include "free_energy.h"

__host__ int stats_colloid_force_split_update(colloids_info_t * cinfo,
					      fe_t * fe);
__host__ int stats_colloid_force_split_output(colloids_info_t * cinfo,
					      int timestep);
#endif
