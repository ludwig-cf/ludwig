/*****************************************************************************
 *
 *  phi_force.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2021 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_FORCE_H
#define LUDWIG_PHI_FORCE_H

#include "pe.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "field.h"
#include "hydro.h"
#include "phi_force_stress.h"
#include "wall.h"
#include "colloids.h"

__host__ int phi_force_calculation(pe_t * , cs_t * cs, lees_edw_t * le,
                                   wall_t * wall,
				   pth_t * pth, fe_t * fe, map_t * map,
				   field_t * phi, hydro_t * hydro,
					field_t * subgrid_potential, rt_t * rt, field_t * vesicle_map, field_t * phi_gradmu, field_t * psi_gradmu);

#endif
