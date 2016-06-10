/*****************************************************************************
 *
 *  gradient_3d_27pt_solid.h
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

#ifndef GRADIENT_3D_27PT_SOLID_H
#define GRADIENT_3D_27PT_SOLID_H

#include "map.h"
#include "field_grad.h" 

__host__ int grad_3d_27pt_solid_map_set(map_t * map);
__host__ int grad_3d_27pt_solid_d2(field_grad_t * fg);

#endif
