/*****************************************************************************
 *
 *  gradient_ternary_solid.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_TERNARY_SOLID_H
#define GRADIENT_TERNARY_SOLID_H

#include "map.h"
#include "ternary.h"
#include "field_grad.h"

__host__ int grad_ternary_solid_map_set(map_t * map);
__host__ int grad_ternary_solid_d2(field_grad_t * fg);
__host__ int grad_ternary_solid_dab(field_grad_t * fg);
__host__ int grad_ternary_solid_fe_set(fe_ternary_t * fe);

#endif
