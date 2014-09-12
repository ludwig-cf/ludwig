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

#include "colloids.h"

int colloids_init_random(colloids_info_t * cinfo, int n,
			 const colloid_state_t * state0, double dh);

#endif
