/*****************************************************************************
 *
 *  lb_bc_inflow_rhou.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_BC_INFLOW_RHOU_H
#define LUDWIG_LB_BC_INFLOW_RHOU_H

#include "lb_bc_open.h"
#include "lb_bc_inflow_opts.h"

typedef struct lb_bc_inflow_rhou_s lb_bc_inflow_rhou_t;

/* Inflow boundary condition. */

struct lb_bc_inflow_rhou_s {
  lb_bc_open_t super;              /* Superclass block */
  pe_t * pe;                       /* Parallel environment */
  cs_t * cs;                       /* Coordinate system */
  lb_bc_inflow_opts_t options;     /* Options/parameters */
  lb_bc_inflow_rhou_t * target;    /* Target pointer */
  
  /* Boundary links */
  int nlink;                    /* Number of links (local) */
  int * linki;                  /* Fluid site in boundary (halo) region */
  int * linkj;                  /* Fluid site in domain proper */
  int8_t * linkp;               /* Velocity index in lb basis (i->j) */
};

__host__ int lb_bc_inflow_rhou_create(pe_t * pe, cs_t * cs,
				      const lb_bc_inflow_opts_t * options,
				      lb_bc_inflow_rhou_t ** inflow);

__host__ int lb_bc_inflow_rhou_free(lb_bc_inflow_rhou_t * inflow);
__host__ int lb_bc_inflow_rhou_update(lb_bc_inflow_rhou_t * inflow,
				      hydro_t * hydro);
__host__ int lb_bc_inflow_rhou_impose(lb_bc_inflow_rhou_t * inflow,
				      hydro_t * hydro,
				      lb_t * lb);
__host__ int lb_bc_inflow_rhou_stats(lb_bc_inflow_rhou_t * inflow);

#endif

