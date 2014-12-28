/*****************************************************************************
 *
 *  stats_symmetric.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2015 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *****************************************************************************/

#ifndef STATS_SYMMETRIC_H
#define STATS_SYMMETRIC_H

#include "coords.h"
#include "field_grad.h"
#include "map.h"

int stats_symmetric_length(coords_t * cs, field_grad_t * dphi, map_t * map,
			   int timestep);
int stats_symmetric_moment_inertia(coords_t * cs, field_t * phi, map_t * map,
				   int timestep);

#endif
