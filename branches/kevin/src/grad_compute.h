/*****************************************************************************
 *
 *  grad_compute.h
 *
 *  Abstract gradient compute.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2015 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef GRAD_COMPUTE_H
#define GRAD_COMPUTE_H

#include "pe.h"
#include "field.h"
#include "field_grad.h"

typedef struct grad_compute_s grad_compute_t;

__host__ int grad_compute_free(grad_compute_t * gc);
__host__ int grad_computer(grad_compute_t * gc, field_t * field,
			   field_grad_t * grad);

#endif
