/*****************************************************************************
 *
 *  advection.h
 *
 *  $Id: advection.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef ADVECTION_H
#define ADVECTION_H

#include "hydro.h"
#include "field.h"

typedef struct advflux_s advflux_t;

__host__ int advflux_cs_create(cs_t * cs, int nf, advflux_t **obj);
__host__ int advflux_le_create(lees_edw_t * le, int nf, advflux_t ** pobj);
__host__ int advflux_free(advflux_t * obj);
__host__ int advflux_memcpy(advflux_t * obj);

__targetHost__  int advection_x(advflux_t * obj, hydro_t * hydro, field_t * field);

__targetHost__ int advective_fluxes(hydro_t * hydro, int nf, double * f, double * fe,
			double * fy, double * fz);
__targetHost__ int advective_fluxes_2nd(hydro_t * hydro, int nf, double * f, double * fe,
			double * fy, double * fz);
__targetHost__ int advective_fluxes_d3qx(hydro_t * hydro, int nf, double * f, 
			double ** flx);
__targetHost__ int advective_fluxes_2nd_d3qx(hydro_t * hydro, int nf, double * f, 
			double ** flx);

__targetHost__ int advection_order_set(const int order);
__targetHost__ int advection_order(int * order);
#endif
