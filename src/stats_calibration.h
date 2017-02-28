/*****************************************************************************
 *
 *  stats_calibration.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2017 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_STATS_CALIBRATION_H
#define LUDWIG_STATS_CALIBRATION_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "hydro.h"
#include "map.h"

typedef struct stats_ahydro_s stats_ahydro_t;

int stats_ahydro_create(pe_t * pe, cs_t * cs, colloids_info_t * cinfo,
			hydro_t * hydro, map_t * map, stats_ahydro_t ** pobj);
int stats_ahydro_accumulate(stats_ahydro_t * stat, int ntimestep);
int stats_ahydro_free(stats_ahydro_t * stat);

#endif
