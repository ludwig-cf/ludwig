/*****************************************************************************
 *
 *  grad_compute.c
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

#include "grad_compute_s.h"

/*****************************************************************************
 *
 *  grad_compute_free
 *
 *****************************************************************************/

__host__ int grad_compute_free(grad_compute_t * gc) {

  assert(gc);
  assert(gc->vtable);
  assert(gc->vtable->free);

  gc->vtable->free(gc);

  return 0;
}

/*****************************************************************************
 *
 *  grad_computer
 *
 *****************************************************************************/

__host__ int grad_computer(grad_compute_t * gc, field_t * field,
			   field_grad_t * grad) {

  assert(gc);
  assert(gc->vtable);
  assert(gc->vtable->compute);

  gc->vtable->compute(gc, field, grad);

  return 0;
}
