/*****************************************************************************
 *
 *  grad_compute_s.h
 *
 *****************************************************************************/

#ifndef GRAD_COMPUTE_S_H
#define GRAD_COMPUTE_S_H

#include "grad_compute.h"

typedef int (* grad_compute_free_ft)(grad_compute_t * gc);
typedef int (* grad_computer_ft)(grad_compute_t * gc, field_t * field,
				 field_grad_t * grad);

/* Function table. */

typedef struct gc_vtable_s gc_vtable_t;

struct gc_vtable_s {
  grad_compute_free_ft free;       /* Destructor */
  grad_computer_ft     compute;    /* Does work */
};

struct grad_compute_s {
  gc_vtable_t * vtable;
};

#endif
