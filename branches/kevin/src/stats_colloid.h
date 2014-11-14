/*****************************************************************************
 *
 *  stats_colloid.h
 *
 *  $Id: stats_colloid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_COLLOID_H
#define STATS_COLLOID_H

#include "colloids.h"

int stats_colloid_momentum(colloids_info_t * cinfo, double g[3]);
int stats_colloid_velocity_minmax(colloids_info_t * cinfo);
 
#endif
