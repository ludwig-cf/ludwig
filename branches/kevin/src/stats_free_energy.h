/****************************************************************************
 *
 *  stats_free_energy.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef STATS_FREE_ENERGY_H
#define STATS_FREE_ENERGY_H

#include "coords.h"
#include "field.h"
#include "field_grad.h"
#include "wall.h"
#include "map.h"

int stats_free_energy_density(coords_t * cs, field_t * q, map_t * map,
			      wall_t * wall, int ncolloid);
int blue_phase_stats(coords_t * cs, field_t * q, field_grad_t * dq,
		     map_t * map, int tstep);

#endif
