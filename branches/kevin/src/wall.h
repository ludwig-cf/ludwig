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
#include "model.h"
#include "map.h"

int wall_init(rt_t * rt, lb_t * lb, map_t * map);
int wall_bounce_back(lb_t * lb, map_t * map);
int wall_set_wall_velocity(lb_t * lb);

void wall_finish(void);

void wall_accumulate_force(const double f[3]);
void wall_net_momentum(double g[3]);
int  wall_present(void);
int  wall_at_edge(const int dimension);
int wall_pm(int * present);

double wall_lubrication(const int dim, const double r[3], const double ah);

#endif
