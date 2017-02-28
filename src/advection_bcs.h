/*****************************************************************************
 *
 *  advection_bcs.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2017 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_ADVECTION_BCS_H
#define LUDWIG_ADVECTION_BCS_H

#include "advection.h"
#include "field.h"
#include "map.h"

__host__ int advection_bcs_no_normal_flux(int nf, advflux_t * flux, map_t * map);
__host__ int advection_bcs_wall(field_t * phi);

__host__ int advective_bcs_no_flux_d3qx(int nf, double ** flx, map_t * map);

#endif
