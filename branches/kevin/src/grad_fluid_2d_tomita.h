/*****************************************************************************
 *
 *  grad_fluid_2d_tomita.h
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2015 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef GRAD_FLUID_2D_TOMITA_H
#define GRAD_FLUID_2D_TOMITA_H

#include "leesedwards.h"
#include "grad_compute.h"

typedef struct grad_fluid_2d_tomita_s grad_fluid_2d_tomita_t;

__host__ int grad_fluid_2d_tomita_create(le_t * le,
					 grad_fluid_2d_tomita_t ** pobj);
__host__ int grad_fluid_2d_tomita_free(grad_fluid_2d_tomita_t * gc);
__host__ int grad_fluid_2d_tomita_computer(grad_fluid_2d_tomita_t * gc,
					   field_t * field,
					   field_grad_t * grad);
#endif
