/*****************************************************************************
 *
 *  stats_colloid_force_split.h
 *
 *
 *****************************************************************************/

#ifndef LUDWIG_STATS_COLLOID_FORCE_SPLIT_H
#define LUDWIG_STATS_COLLOID_FORCE_SPLIT_H

#include "colloids.h"
#include "field.h"
#include "field_grad.h"
#include "free_energy.h"
#include "map.h"

__host__ int stats_colloid_force_split_update(colloids_info_t * cinfo,
					      fe_t * fe, map_t * map,
					      field_t * q,
					      field_grad_t * q_grad);
__host__ int stats_colloid_force_split_output(colloids_info_t * cinfo,
					      int timestep);
#endif
