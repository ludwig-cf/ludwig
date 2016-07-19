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
 *  (c) 2009 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef STATS_FREE_ENERGY_H
#define STATS_FREE_ENERGY_H

#include "free_energy.h" 
#include "field.h"
#include "field_grad.h"
#include "map.h"
#include "colloids.h"

int stats_free_energy_density(fe_t * fe, field_t * q, map_t * map,
			      colloids_info_t * cinfo);
int blue_phase_stats(field_t * q, field_grad_t * dq, map_t * map, int tstep);

#endif
