/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid.h
 *
 *  $Id: gradient_2d_5pt_fluid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_2D_5PT_FLUID_H
#define GRADIENT_2D_5PT_FLUID_H

#include "field_grad.h"

__host__ int grad_2d_5pt_fluid_d2(field_grad_t * fgrad);
__host__ int grad_2d_5pt_fluid_d4(field_grad_t * fgrad);

#endif
