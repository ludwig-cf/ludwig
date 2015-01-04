/*****************************************************************************
 *
 *  phi_lb_coupler.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2015 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PHI_LB_COUPLER_H
#define PHI_LB_COUPLER_H

#include "coords.h"
#include "field.h"
#include "model.h"

int phi_lb_to_field(coords_t * cs, field_t * phi, lb_t * lb);
int phi_lb_from_field(coords_t * cs, field_t * phi, lb_t * lb);

#endif
