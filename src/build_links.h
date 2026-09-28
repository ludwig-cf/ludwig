/*****************************************************************************
 *
 *  build_links.h
 *
 *  Responsible for construction/reconstruction of link related
 *  properties for colloids in BBL.
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_BUILD_LINKS_H
#define LUDWIG_BUILD_LINKS_H

#include "colloids.h"
#include "lb_data.h"
#include "map.h"
#include "wall.h"

int build_links_update_driver(colloids_info_t * info, wall_t * wall,
                              map_t * map, const lb_t * lb);

#endif
