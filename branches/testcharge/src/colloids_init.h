/*****************************************************************************
 *
 *  colloids_init.h
 *
 *  $Id: colloids_init.h,v 1.3 2010-10-21 18:13:42 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOIDS_INIT_H

#include "colloids.h"

int colloids_init_random(colloids_info_t * cinfo, int n,
			 const colloid_state_t * state0, double dh);

#endif
