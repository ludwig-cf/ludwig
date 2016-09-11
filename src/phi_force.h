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

#ifndef PHI_FORCE_H
#define PHI_FORCE_H

#include "leesedwards.h"
#include "free_energy.h"
#include "field.h"
#include "hydro.h"
#include "field_grad_s.h"
#include "phi_force_stress.h"

__host__ int phi_force_calculation(cs_t * cs, lees_edw_t * le, pth_t * pth,
				   fe_t * fe, field_t * phi,
				   hydro_t * hydro);

#endif
