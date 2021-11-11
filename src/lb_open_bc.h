/*****************************************************************************
 *
 *  lb_open_bc.h
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

#ifndef LUDWIG_LB_OPEN_BC_H
#define LUDWIG_LB_OPEN_BC_H

#include "hydro.h"
#include "model.h"

/* Implementations available */

typedef enum {LB_OPEN_BC_INFLOW_RHOU} lb_open_bc_enum_t;

typedef struct lb_open_bc_vtable_s lb_open_bc_vtable_t;
typedef struct lb_open_bc_s lb_open_bc_t;

/* Abstract function table */
/* lb_open_bc_free_ft         is destructor */
/* lb_open_bc_update_ft       update driver to set (rho, u) */
/* lb_open_bc_impose_ft       apply conditions (set relevant f_i) */
/* lb_open_bc_stats_ft        general purpose stats with optional extra info */

typedef int (* lb_open_bc_free_ft)   (lb_open_bc_t * bc);
typedef int (* lb_open_bc_update_ft) (lb_open_bc_t * bc, hydro_t * hydro);
typedef int (* lb_open_bc_impose_ft) (lb_open_bc_t * bc, hydro_t * hydro,
				      lb_t * lb);
typedef int (* lb_open_bc_stats_ft)  (lb_open_bc_t * bc); 

struct lb_open_bc_vtable_s {
  lb_open_bc_free_ft   free;      /* Desctructor */
  lb_open_bc_update_ft update;    /* Update */
  lb_open_bc_impose_ft impose;    /* Apply update to distributions */
  lb_open_bc_stats_ft  stats;     /* General information */
};

struct lb_open_bc_s {
  const lb_open_bc_vtable_t * func;   /* function table */
  lb_open_bc_enum_t             id;   /* unique type identifier */
};

#endif
