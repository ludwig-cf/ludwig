/*****************************************************************************
 *
 *  phi_force.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_FORCE_H
#define LUDWIG_PHI_FORCE_H

#include "leesedwards.h"
#include "free_energy.h"
#include "field.h"
#include "hydro.h"
#include "field_grad_s.h"
#include "phi_force_stress.h"
#include "wall.h"

__host__ int phi_force_calculation(cs_t * cs, lees_edw_t * le, wall_t * wall,
				   pth_t * pth, fe_t * fe, map_t * map,
				   field_t * phi, hydro_t * hydro);

#endif
