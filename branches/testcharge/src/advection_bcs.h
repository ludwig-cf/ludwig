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
 *  (c) 2009 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef ADVECTION_BCS_H
#define ADVECTION_BCS_H

#include "advection.h"
#include "field.h"
#include "map.h"

int advection_bcs_no_normal_flux(int nf, advflux_t * flux, map_t * map);
int advection_bcs_wall(field_t * phi);
int advective_bcs_no_flux(int nf, double * fx, double * fy, double * fz,
			  map_t * map);
int advective_bcs_no_flux_d3q19(int nf, double ** flx, map_t * map);

#endif
