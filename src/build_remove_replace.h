/*****************************************************************************
 *
 *  build_remove_replace.h
 *
 *  (c) The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_BUILD_REMOVE_REPLACE_H
#define LUDWIG_BUILD_REMOVE_REPLACE_H

#include "colloids.h"
#include "field.h"
#include "lb_data.h"
#include "map.h"

int build_bbl_rebuild_flags_driver(colloids_info_t * info);
int build_remove_replace_fluid_driver(lb_t * lb, colloids_info_t * info,
                                      map_t * map);
int build_remove_replace_order_parameter_driver(lb_t *            lb,
                                                colloids_info_t * info,
                                                map_t * map, field_t * phi);
#endif
