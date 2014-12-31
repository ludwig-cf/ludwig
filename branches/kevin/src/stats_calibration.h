/*****************************************************************************
 *
 *  stats_ahydro.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_AHYDRO_H
#define STATS_AHYDRO_H

#include "coords.h"
#include "colloids.h"
#include "hydro.h"
#include "map.h"

typedef struct stats_ahydro_s stats_ahydro_t;

int stats_ahydro_create(coords_t * cs, stats_ahydro_t ** p);
int stats_ahydro_free(stats_ahydro_t * ahydro);
int stats_ahydro_init(stats_ahydro_t * ahydro, colloids_info_t * cinfo);
int stats_ahydro_accumulate(stats_ahydro_t * ahydro, colloids_info_t * cinfo,
			    int ntimestep, hydro_t * hydro, map_t * map);
int stats_ahydro_finish(stats_ahydro_t * ahydro);

#endif
