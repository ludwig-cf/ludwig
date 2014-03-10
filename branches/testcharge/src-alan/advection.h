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
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef ADVECTION_H
#define ADVECTION_H

#include "hydro.h"
#include "field.h"

typedef struct advflux_s advflux_t;

int advflux_create(int nf, advflux_t ** pobj);
void advflux_free(advflux_t * obj);
int advection_x(advflux_t * obj, hydro_t * hydro, field_t * field);

int advective_fluxes(hydro_t * hydro, int nf, double * f, double * fe,
		     double * fy, double * fz);
int advective_fluxes_2nd(hydro_t * hydro, int nf, double * f, double * fe,
			 double * fy, double * fz);

int advection_order_set(const int order);
int advection_order(int * order);
#endif
