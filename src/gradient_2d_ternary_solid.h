/*****************************************************************************
 *
 *  gradient_2d_ternary_solid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_GRADIENT_2D_TERNARY_SOLID_H
#define LUDWIG_GRADIENT_2D_TERNARY_SOLID_H

#include "fe_ternary.h"
#include "field_grad.h"
#include "map.h"

__host__ int grad_2d_ternary_solid_fe_set(fe_ternary_t * fe);
__host__ int grad_2d_ternary_solid_d2(field_grad_t * fgrad);
__host__ int grad_2d_ternary_solid_set(map_t * map);

#endif
