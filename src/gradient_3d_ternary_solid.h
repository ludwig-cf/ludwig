/*****************************************************************************
 *
 *  gradient_3d_ternary_solid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *
 *****************************************************************************/

#ifndef LUDWIG_GRAD_3D_TERNARY_SOLID_H
#define LUDWIG_GRAD_3D_TERNARY_SOLID_H

#include "map.h"
#include "fe_ternary.h"
#include "field_grad.h"

__host__ int grad_3d_ternary_solid_map_set(map_t * map);
__host__ int grad_3d_ternary_solid_fe_set(fe_ternary_t * fe);
__host__ int grad_3d_ternary_solid_d2(field_grad_t * fgrad);

#endif
