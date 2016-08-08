/*****************************************************************************
 *
 *  wall.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef WALL_H
#define WALL_H

#include "runtime.h"
#include "model.h"
#include "map.h"
#include "targetDP.h"

__host__ int wall_init(rt_t * rt, lb_t * lb, map_t * map);
__host__ int wall_bounce_back(lb_t * lb, map_t * map);
__host__ int wall_set_wall_velocity(lb_t * lb);

__host__ void wall_finish(void);

__host__ void wall_accumulate_force(const double f[3]);
__host__ void wall_net_momentum(double g[3]);
__host__ int  wall_present(void);
__host__ int  wall_at_edge(const int dimension);
__host__ int wall_pm(int * present);

__host__ double wall_lubrication(const int dim, const double r[3], const double ah);

#endif
