/*****************************************************************************
 *
 *  advection_bcs.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2023 The University of Edinburgh
 *
 *  Contributing authors:
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
__host__ int advflux_cs_no_normal_flux(advflux_t * flux, map_t * map);

#endif
