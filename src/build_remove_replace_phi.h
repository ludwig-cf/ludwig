/*****************************************************************************
 *
 *  build_remove_replace_phi.h
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_BUILD_REMOVE_REPLACE_PHI_H
#define LUDWIG_BUILD_REMOVE_REPLACE_PHI_H

#include "colloids.h"
#include "field.h"
#include "lb_data.h"
#include "map.h"

int build_remove_replace_order_parameter_driver(lb_t *            lb,
                                                colloids_info_t * info,
                                                map_t * map,
						field_t * phi);
int build_conservation_phi_driver(const colloids_info_t * info,
				  field_t * phi);
#endif
