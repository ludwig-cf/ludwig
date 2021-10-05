/*****************************************************************************
 *
 *  map_init.h
 *
 *  Various map status/data initialisations.
 *
 *  Edinburgh Soft Matter and Statisical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_MAP_INIT_H
#define LUDWIG_MAP_INIT_H

#include "map.h"

__host__ int map_init_status_circle_xy(map_t * map);
__host__ int map_init_status_wall(map_t * map, int idim);
__host__ int map_init_status_simple_cubic(map_t * map, int acell);
__host__ int map_init_status_body_centred_cubic(map_t * map, int acell);
__host__ int map_init_status_face_centred_cubic(map_t * map, int acell);
__host__ int map_init_status_print_section(map_t * map, int id, int ord);
__host__ int map_init_data_uniform(map_t * map, int target, double * data);

#endif
