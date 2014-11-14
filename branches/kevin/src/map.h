/*****************************************************************************
 *
 *  map.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef MAP_H
#define MAP_H

#include "io_harness.h"
#include "targetDP.h"

enum map_status {MAP_FLUID, MAP_BOUNDARY, MAP_COLLOID, MAP_STATUS_MAX};

typedef struct map_s map_t;

HOST int map_create(int ndata, map_t ** pobj);
HOST void map_free(map_t * obj);

HOST int map_status(map_t * obj, int index, int * status);
HOST int map_status_set(map_t * obj, int index, int status);
HOST int map_pm(map_t * map, int * porous_media_flag);
HOST int map_pm_set(map_t * map, int porous_media_flag);
HOST int map_ndata(map_t * obj, int * ndata);
HOST int map_data(map_t * obj, int index, double * data);
HOST int map_data_set(map_t * obj, int index, double * data);
HOST int map_volume_local(map_t * obj, int status, int * volume);
HOST int map_volume_allreduce(map_t * obj, int status, int * volume);
HOST int map_halo(map_t * obj);
HOST int map_init_io_info(map_t * obj, int grid[3], int form_in, int form_out);
HOST int map_io_info(map_t * obj, io_info_t ** info);

#endif
