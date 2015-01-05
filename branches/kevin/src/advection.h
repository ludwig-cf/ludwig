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
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef ADVECTION_H
#define ADVECTION_H

#include "leesedwards.h"
#include "hydro.h"
#include "field.h"

typedef struct advflux_s advflux_t;

/* For Lees Edwards */

int advflux_create(le_t * le, int nf, advflux_t ** pobj);
int advflux_free(advflux_t * flux);
int advflux_compute(advflux_t * flux, hydro_t * hydro, field_t * field);
int advflux_free(advflux_t * flux);

int advection_order_set(const int order);
int advection_order(int * order);

#endif
