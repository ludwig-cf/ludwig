/*****************************************************************************
 *
 *  collision.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Alan Gray for vectorisation
 *    Ronojoy Adhikari worked out collision basis/fluctuations
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_COLLISION_H
#define LUDWIG_LB_COLLISION_H

#include "hydro.h"
#include "map.h"
#include "noise.h"
#include "model.h"
#include "free_energy.h"

__host__ int lb_collide(lb_t * lb, hydro_t * hydro, map_t * map,
			noise_t * noise, fe_t * fe);
__host__ int lb_collision_stats_kt(lb_t * lb, noise_t * noise, map_t * map);
__host__ int lb_collision_relaxation_set(lb_t * lb, lb_relaxation_enum_t nrelax);

__host__ int lb_collision_ghost_modes_on(lb_t * lb);
__host__ int lb_collision_ghost_modes_off(lb_t * lb);
__host__ int lb_collision_relaxation_times(lb_t * lb, double * tau);
__host__ int lb_collision_relaxation_times_set(lb_t * lb);

#endif
