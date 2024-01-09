/*****************************************************************************
 *
 *  build.h
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_BUILD_H
#define LUDWIG_BUILD_H

#include "field.h"
#include "map.h"
#include "psi.h"
#include "colloids.h"
#include "lb_data.h"
#include "wall.h"
#include "free_energy.h"

int build_remove_replace(fe_t * fe, colloids_info_t * cinfo, lb_t * lb, field_t * phi,
			 field_t * p, field_t * q, psi_t * psi, map_t * map);

int build_update_links(cs_t * cs, colloids_info_t * cinfo, wall_t * wall,
		       map_t * map, const lb_model_t * model);
int build_update_map(cs_t * cs, colloids_info_t * cinfo, map_t * map);
int build_conservation(colloids_info_t * info, field_t * phi, psi_t * psi,
		       const lb_model_t * model);

int build_count_links_local(colloid_t * colloid, int * nlinks);
int build_count_faces_local(colloid_t * colloid, const lb_model_t * model,
			    double * sa, double * saf);
#endif
