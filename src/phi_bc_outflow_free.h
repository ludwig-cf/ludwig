/*****************************************************************************
 *
 *  phi_bc_outflow_free.h
 *
 *  An outflow boundary conditions.
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

#ifndef LUDWIG_PHI_BC_OUTFLOW_FREE_H
#define LUDWIG_PHI_BC_OUTFLOW_FREE_H

#include "phi_bc_open.h"
#include "phi_bc_outflow_opts.h"

typedef struct phi_bc_outflow_free_s phi_bc_outflow_free_t;

struct phi_bc_outflow_free_s {
  phi_bc_open_t super;                /* Superclass block */
  pe_t * pe;                          /* Parallel environment */
  cs_t * cs;                          /* Coordinate system */
  phi_bc_outflow_opts_t options;      /* Parameters */

  phi_bc_outflow_free_t * target;     /* Device copy */
};

__host__ int phi_bc_outflow_free_create(pe_t * pe, cs_t * cs,
					const phi_bc_outflow_opts_t * options,
					phi_bc_outflow_free_t ** outflow);
__host__ int phi_bc_outflow_free_update(phi_bc_outflow_free_t * outflow,
					field_t * phi);
__host__ int phi_bc_outflow_free_free(phi_bc_outflow_free_t * outflow);

#endif
