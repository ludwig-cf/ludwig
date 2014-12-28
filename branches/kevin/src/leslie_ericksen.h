/*****************************************************************************
 *
 *  leslie_ericksen.h
 *
 *  $Id: leslie_ericksen.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LESLIE_ERICKSEN_H
#define LESLIE_ERICKSEN_H

#include "coords.h"
#include "field.h"
#include "hydro.h"

int leslie_ericksen_update(field_t * p, coords_t * cs, hydro_t * hydro);
int leslie_ericksen_gamma_set(const double gamma);
int leslie_ericksen_swim_set(const double gamma);

#endif
