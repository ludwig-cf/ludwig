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

#ifndef PHI_FORCE_COLLOID_H
#define PHI_FORCE_COLLOID_H

#include "colloids.h"
#include "hydro.h"
#include "map.h"
#include "phi_force_stress.h"

__host__ int phi_force_colloid(pth_t * pth, colloids_info_t * cinfo,
			       field_t* q, field_grad_t* q_grad,
			       hydro_t * hydro, map_t * map);

#endif
