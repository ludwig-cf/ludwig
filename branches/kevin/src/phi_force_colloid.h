/*****************************************************************************
 *
 *  phi_force_colloid.h
 *
 *  $Id: phi_force_colloid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2015 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PHI_FORCE_COLLOID_H
#define PHI_FORCE_COLLOID_H

#include "coords.h"
#include "colloids.h"
#include "hydro.h"
#include "map.h"
#include "wall.h"

int phi_force_colloid(coords_t * cs, colloids_info_t * cinfo, hydro_t * hydro,
		      wall_t * wall, map_t * map);

#endif
