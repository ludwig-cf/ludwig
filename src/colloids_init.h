/*****************************************************************************
 *
 *  colloids_init.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2019 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOIDS_INIT_H
#define LUDWIG_COLLOIDS_INIT_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "wall.h"

int colloids_init_random(pe_t * pe, cs_t * cs, colloids_info_t * cinfo, int n,
			 const colloid_state_t * state0, wall_t * wall,
			 double dh);

#endif
