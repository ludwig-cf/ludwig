/*****************************************************************************
 *
 *  stats_calibration.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_CALIBRATION_H
#define STATS_CALIBRATION_H

#include "hydro.h"
#include "map.h"

void stats_calibration_init(int nswitch);
int  stats_calibration_accumulate(int ntimestep, hydro_t * hydro, map_t * map);
void stats_calibration_finish(void);

#endif
