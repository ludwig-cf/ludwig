/*****************************************************************************
 *
 *  phi_cahn_hilliard.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  $Id: phi_cahn_hilliard.h,v 1.5 2010-10-15 12:40:03 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_CAHN_HILLIARD_H
#define PHI_CAHN_HILLIARD_H

#include "leesedwards.h"
#include "field.h"
#include "hydro.h"
#include "map.h"
#include "noise.h"

int phi_cahn_hilliard(le_t * le, field_t * phi, hydro_t * hydro,
		      map_t * map, noise_t * noise);

#endif
