/*****************************************************************************
 *
 *  build_remove_replace_q.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_BUILD_REMOVE_REPLACE_Q_H
#define LUDWIG_BUILD_REMOVE_REPLACE_Q_H

#include "lb_data.h"
#include "blue_phase.h"
#include "colloids.h"
#include "map.h"

int build_remove_replace_q_driver(const lb_t * lb, const fe_lc_t * fe,
				  const colloids_info_t * info,
				  const map_t * map,
				  field_t * q);

#endif
