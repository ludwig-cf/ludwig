/*****************************************************************************
 *
 *  cahn_hilliard_stats_ll.h
 *  
 *  Adapted from cahn_hilliard_stats.c to acoomodate 2 order parameters
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_CAHN_HILLIARD_STATS_LL_H
#define LUDWIG_CAHN_HILLIARD_STATS_LL_H

#include "cahn_hilliard.h"

__host__ int cahn_hilliard_stats_ll(ch_t * ch, field_t * phi, map_t * map);
__host__ int cahn_hilliard_stats_ll_time0(ch_t * ch, field_t * phi, map_t * map);

#endif
