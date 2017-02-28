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
 *  (c) 2010-2017 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_ADVECTION_H
#define LUDWIG_ADVECTION_H

#include "pe.h"
#include "coords.h"
#include "hydro.h"
#include "field.h"

typedef struct advflux_s advflux_t;

__host__ int advflux_create(pe_t * pe, cs_t * cs, lees_edw_t * le, int nf,
			    advflux_t ** pobj);
__host__ int advflux_cs_create(pe_t * pe, cs_t * cs, int nf, advflux_t **obj);
__host__ int advflux_le_create(pe_t * pe, cs_t * cs, lees_edw_t * le, int nf,
			       advflux_t ** pobj);
__host__ int advflux_free(advflux_t * obj);
__host__ int advflux_memcpy(advflux_t * obj);

__host__ int advection_x(advflux_t * obj, hydro_t * hydro, field_t * field);
__host__ int advective_fluxes_d3qx(hydro_t * hydro, int nf, double * f, 
			double ** flx);
__host__ int advective_fluxes_2nd_d3qx(hydro_t * hydro, int nf, double * f, 
			double ** flx);

__host__ int advection_order_set(const int order);
__host__ int advection_order(int * order);
#endif
