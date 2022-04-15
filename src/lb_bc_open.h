/*****************************************************************************
 *
 *  lb_bc_open.h
 *
 *  Lattice fluid open boundary condition (abstract type).
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

#ifndef LUDWIG_LB_BC_OPEN_H
#define LUDWIG_LB_BC_OPEN_H

#include "hydro.h"
#include "lb_data.h"

typedef struct lb_bc_open_vtable_s lb_bc_open_vtable_t;
typedef struct lb_bc_open_s        lb_bc_open_t;

/* Abstract function table */
/* lb_bc_open_free_ft         is destructor */
/* lb_bc_open_update_ft       update driver to set (rho, u) */
/* lb_bc_open_impose_ft       apply conditions (set relevant f_i) */
/* lb_bc_open_stats_ft        general purpose stats with optional extra info */

typedef int (* lb_bc_open_free_ft)   (lb_bc_open_t * bc);
typedef int (* lb_bc_open_update_ft) (lb_bc_open_t * bc, hydro_t * hydro);
typedef int (* lb_bc_open_impose_ft) (lb_bc_open_t * bc, hydro_t * hydro,
				      lb_t * lb);
typedef int (* lb_bc_open_stats_ft)  (lb_bc_open_t * bc);

struct lb_bc_open_vtable_s {
  lb_bc_open_free_ft   free;      /* Desctructor */
  lb_bc_open_update_ft update;    /* Update */
  lb_bc_open_impose_ft impose;    /* Apply update to distributions */
  lb_bc_open_stats_ft  stats;     /* General information */
};

/* Implementations */

typedef enum lb_bc_open_enum {LB_BC_INVALID,
                              LB_BC_INFLOW_RHOU,
			      LB_BC_OUTFLOW_RHOU,
			      LB_BC_MAX} lb_bc_open_enum_t;

/* Superclass block */

struct lb_bc_open_s {
  const lb_bc_open_vtable_t * func;   /* function table */
  lb_bc_open_enum_t             id;   /* unique type identifier */
};

#endif
