/*****************************************************************************
 *
 *  wall.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef WALL_H
#define WALL_H

#include "model.h"
#include "map.h"
#include "targetDP.h"

__targetHost__ int wall_init(lb_t * lb, map_t * map);
__targetHost__ int wall_bounce_back(lb_t * lb, map_t * map);
__targetHost__ int wall_set_wall_velocity(lb_t * lb);

__targetHost__ void wall_finish(void);

__targetHost__ void wall_accumulate_force(const double f[3]);
__targetHost__ void wall_net_momentum(double g[3]);
__targetHost__ int  wall_present(void);
__targetHost__ int  wall_at_edge(const int dimension);
__targetHost__ int wall_pm(int * present);

__targetHost__ double wall_lubrication(const int dim, const double r[3], const double ah);

#endif
