/*****************************************************************************
 *
 *  cahn_hilliard_stats.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_CAHN_HILLIARD_STATS_H
#define LUDWIG_CAHN_HILLIARD_STATS_H

#include "phi_cahn_hilliard.h"

__host__ int cahn_hilliard_stats(phi_ch_t * pch, field_t * phi, map_t * map);

#endif
