/*****************************************************************************
 *
 *  phi_force_colloid.h
 *
 *  $Id: phi_force_colloid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *  (c) 2009-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_FORCE_COLLOID_H
#define LUDWIG_PHI_FORCE_COLLOID_H

#include "phi_force_stress.h"
#include "free_energy.h"
#include "colloids.h"
#include "hydro.h"
#include "map.h"
#include "wall.h"

__host__ int phi_force_colloid(pth_t * pth, fe_t * fe, colloids_info_t * cinfo,
			       hydro_t * hydro, map_t * map, wall_t * wall);

#endif
