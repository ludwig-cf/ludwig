/*****************************************************************************
 *
 *  phi_force_colloid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *  (c) 2009-2026 The University of Edinburgh
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

int pth_force_fluid_driver(pth_t * pth, hydro_t * hydro);
int pth_force_fluid_wall_driver(pth_t * pth, hydro_t * hydro, map_t * map,
                                wall_t * wall);
int pth_force_colloid(pth_t * pth, fe_t * fe, colloids_info_t * cinfo,
                      hydro_t * hydro, map_t * map, wall_t * wall,
                      const lb_t * lb);
#endif
