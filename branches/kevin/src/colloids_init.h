/*****************************************************************************
 *
 *  colloids_init.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOIDS_INIT_H

#include "coords.h"
#include "colloids.h"
#include "wall.h"

int colloids_init_random(coords_t * cs, colloids_info_t * cinfo, int n,
			 const colloid_state_t * state0, wall_t * wall,
			 double dh);

#endif
