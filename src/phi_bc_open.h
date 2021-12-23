/*****************************************************************************
 *
 *  phi_bc_open.h
 *
 *  Compositional order parameter boundary condition (abstract type).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_BC_OPEN_H
#define LUDWIG_PHI_BC_OPEN_H

#include "field.h"

typedef struct phi_bc_open_vtable_s phi_bc_open_vtable_t;
typedef struct phi_bc_open_s        phi_bc_open_t;

typedef int (* phi_bc_open_free_ft)   (phi_bc_open_t * bc);
typedef int (* phi_bc_open_update_ft) (phi_bc_open_t * bc, field_t * phi);

struct phi_bc_open_vtable_s {
  phi_bc_open_free_ft   free;      /* Desctructor */
  phi_bc_open_update_ft update;    /* Update */
};

/* Implementations */

typedef enum phi_bc_open_enum {PHI_BC_INVALID,
                               PHI_BC_INFLOW_FIXED,
                               PHI_BC_OUTFLOW_FREE,
                               PHI_BC_MAX} phi_bc_open_enum_t;
/* Superclass block */

struct phi_bc_open_s {
  const phi_bc_open_vtable_t * func;   /* function table */
  phi_bc_open_enum_t             id;   /* unique type identifier */
};

#endif
