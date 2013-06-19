/*****************************************************************************
 *
 *  collision.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLISION_H
#define COLLISION_H

#include "hydro.h"
#include "map.h"
#include "noise.h"

typedef enum {RELAXATION_M10, RELAXATION_BGK, RELAXATION_TRT}
  lb_relaxation_enum_t;

int collide(hydro_t * hydro, map_t * map, noise_t * noise);
int collision_stats_kt(noise_t * noise, map_t * map);
int collision_relaxation_times_set(noise_t * noise);

void collision_ghost_modes_on(void);
void collision_ghost_modes_off(void);
void collision_relaxation_times(double * tau);
void collision_relaxation_set(const int nrelax);

#endif
