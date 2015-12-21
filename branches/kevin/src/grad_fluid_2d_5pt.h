/*****************************************************************************
 *
 *  grad_fluid_2d_5pt.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2015 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef GRAD_FLUID_2D_5PT_H
#define GRAD_FLUID_2D_5PT_H

#include "leesedwards.h"
#include "grad_compute.h"

typedef struct grad_fluid_2d_5pt_s grad_fluid_2d_5pt_t;

__host__ int grad_fluid_2d_5pt_create(le_t * le, grad_fluid_2d_5pt_t ** gc);
__host__ int grad_fluid_2d_5pt_free(grad_fluid_2d_5pt_t * gc);
__host__ int grad_fluid_2d_5pt_computer(grad_fluid_2d_5pt_t * gc,
					field_t * field, field_grad_t * grad);

#endif
