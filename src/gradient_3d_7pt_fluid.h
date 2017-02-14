/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid.h
 *
 *  $Id: gradient_3d_7pt_fluid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_3D_7PT_FLUID_H
#define GRADIENT_3D_7PT_FLUID_H

#include "targetDP.h"
#include "field_grad.h"

__host__ int grad_3d_7pt_fluid_d2(field_grad_t * fg);
__host__ int grad_3d_7pt_fluid_d4(field_grad_t * fg);
__host__ int grad_3d_7pt_fluid_dab(field_grad_t * fg);

#endif
