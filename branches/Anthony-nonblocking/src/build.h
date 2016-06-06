/*****************************************************************************
 *
 *  build.h
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BUILD_H
#define BUILD_H

#include "field.h"
#include "map.h"
#include "psi.h"
#include "colloids.h"
#include "model.h"

int build_remove_replace(colloids_info_t * cinfo, lb_t * lb, field_t * phi,
			 field_t * p, field_t * q, psi_t * psi, map_t * map);

int build_update_links(colloids_info_t * cinfo, map_t * map);
int build_update_map(colloids_info_t * cinfo, map_t * map);
int build_conservation(colloids_info_t * info, field_t * phi, psi_t * psi);

int build_count_links_local(colloid_t * colloid, int * nlinks);
int build_count_faces_local(colloid_t * colloid, double * sa, double * saf);


#endif
