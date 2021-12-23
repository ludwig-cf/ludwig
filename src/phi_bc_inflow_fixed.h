/*****************************************************************************
 *
 *  phi_bc_inflow_fixed.h
 *
 *  Composition inflow boundary condition.
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

#ifndef LUDWIG_PHI_BC_INFLOW_FIXED_H
#define LUDWIG_PHI_BC_INFLOW_FIXED_H

#include "phi_bc_open.h"
#include "phi_bc_inflow_opts.h"

typedef struct phi_bc_inflow_fixed_s phi_bc_inflow_fixed_t;

/* Inflow boundary condition */

struct phi_bc_inflow_fixed_s {

  phi_bc_open_t super;             /* Superclass block */
  pe_t * pe;                       /* Parallel environment */
  cs_t * cs;                       /* Coordinate system */
  phi_bc_inflow_opts_t options;    /* Parameters */
  phi_bc_inflow_fixed_t * target;  /* Device pointer */ 

};

__host__ int phi_bc_inflow_fixed_create(pe_t * pe, cs_t * cs,
					const phi_bc_inflow_opts_t * options,
					phi_bc_inflow_fixed_t ** inflow);
__host__ int phi_bc_inflow_fixed_free(phi_bc_inflow_fixed_t * inflow);
__host__ int phi_bc_inflow_fixed_update(phi_bc_inflow_fixed_t * inflow,
					field_t * phi);

#endif
