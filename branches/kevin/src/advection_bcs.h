/*****************************************************************************
 *
 *  advection_bcs.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef ADVECTION_BCS_H
#define ADVECTION_BCS_H

#include "advection.h"
#include "field.h"
#include "map.h"

int advection_bcs_no_normal_flux(int nf, advflux_t * flux, map_t * map);
int advection_bcs_wall(advflux_t * flux, field_t * phi);

#endif
