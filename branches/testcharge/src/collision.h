/*****************************************************************************
 *
 *  collision.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2014 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Alan Gray for vectorisation
 *    Ronojoy Adhikari worked out collision basis/fluctuations
 *
 *****************************************************************************/

#ifndef COLLISION_H
#define COLLISION_H

#include "hydro.h"
#include "map.h"
#include "noise.h"
#include "model.h"

typedef enum {RELAXATION_M10, RELAXATION_BGK, RELAXATION_TRT}
  lb_relaxation_enum_t;

int lb_collide(lb_t * lb, hydro_t * hydro, map_t * map, noise_t * noise);
int lb_collision_stats_kt(lb_t * lb, noise_t * noise, map_t * map);
int lb_collision_relaxation_times_set(noise_t * noise);

void lb_collision_relaxation_times(double * tau);
void lb_collision_relaxation_set(const int nrelax);

void collision_ghost_modes_on(void);
void collision_ghost_modes_off(void);
void collision_relaxation_times(double * tau);
#endif
