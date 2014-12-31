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

#include "runtime.h"
#include "coords.h"
#include "model.h"
#include "map.h"
#include "targetDP.h"

HOST int wall_init(rt_t * rt, coords_t * cs, lb_t * lb, map_t * map);
HOST int wall_bounce_back(lb_t * lb, map_t * map);
HOST int wall_set_wall_velocity(lb_t * lb);

HOST void wall_finish(void);

HOST void wall_accumulate_force(const double f[3]);
HOST void wall_net_momentum(double g[3]);
HOST int  wall_present(void);
HOST int  wall_at_edge(const int dimension);
HOST int wall_pm(int * present);

HOST double wall_lubrication(coords_t * cs, const int dim, const double r[3],
			     const double ah);

#endif
