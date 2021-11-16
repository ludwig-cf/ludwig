/*****************************************************************************
 *
 *  heat_equation.h
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

#ifndef LUDWIG_HEAT_EQUATION_STATS_H
#define LUDWIG_HEAT_EQUATION_STATS_H

#include "heat_equation.h"

__host__ int heat_equation_stats(heq_t * heq, field_t * temperature, map_t * map);
__host__ int heat_equation_stats_time0(heq_t * heq, field_t * temperature,
				       map_t * map);

#endif
