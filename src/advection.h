/*****************************************************************************
 *
 *  advection.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
__host__ int advflux_zero(advflux_t * obj);
__host__ int advflux_memcpy(advflux_t * obj, tdpMemcpyKind flag);


__host__ int advflux_cs_compute(advflux_t * flux, hydro_t * h, field_t * f);
__host__ int advflux_cs_zero(advflux_t * flux);

__host__ int advection_x(advflux_t * obj, hydro_t * hydro, field_t * field);

__host__ int advection_order_set(const int order);
__host__ int advection_order(int * order);
#endif
