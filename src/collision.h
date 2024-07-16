/*****************************************************************************
 *
 *  collision.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2024 The University of Edinburgh
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
#include "lb_data.h"
#include "free_energy.h"
#include "visc.h"

int lb_collide(lb_t * lb, hydro_t * hydro, map_t * map,
			noise_t * noise, fe_t * fe, visc_t * visc);
int lb_collision_stats_kt(lb_t * lb, map_t * map);
int lb_collision_relaxation_set(lb_t * lb, lb_relaxation_enum_t nrelax);

int lb_collision_ghost_modes_on(lb_t * lb);
int lb_collision_ghost_modes_off(lb_t * lb);
int lb_collision_relaxation_times(lb_t * lb, double * tau);
int lb_collision_relaxation_times_set(lb_t * lb);

#endif
