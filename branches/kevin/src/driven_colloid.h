/*****************************************************************************
 *
 *  driven_colloid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef DRIVEN_COLLOID_H
#define DRIVEN_COLLOID_H

#include "coords.h"
#include "colloids.h"

void driven_colloid_fmod_set(const double f0);
double driven_colloid_fmod_get(void);
void driven_colloid_force(const double s[3], double force[3]);
int driven_colloid_total_force(coords_t * cs, colloids_info_t * cinfo,
			       double ftotal[3]);
int is_driven(void);

#endif
