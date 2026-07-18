/*****************************************************************************
 *
 *  lb_bc_inflow_u.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_BC_INFLOW_U_H
#define LUDWIG_LB_BC_INFLOW_U_H

#include "lb_bc_open.h"
#include "lb_bc_inflow_opts.h"

typedef struct lb_bc_inflow_u_s lb_bc_inflow_u_t;

/* Inflow boundary condition. */

struct lb_bc_inflow_u_s {
  lb_bc_open_t super;              /* Superclass block */
  pe_t * pe;                       /* Parallel environment */
  cs_t * cs;                       /* Coordinate system */
  lb_bc_inflow_opts_t options;     /* Options/parameters */
  lb_bc_inflow_u_t * target;       /* Target pointer */
  
  /* Boundary links */
  int nlink;                    /* Number of links (local) */
  int * linki;                  /* Fluid site in boundary (halo) region */
  int * linkj;                  /* Fluid site in domain proper */
  int8_t * linkp;               /* Velocity index in lb basis (i->j) */
};

int lb_bc_inflow_u_create(pe_t * pe, cs_t * cs,
			  const lb_bc_inflow_opts_t * options,
			  lb_bc_inflow_u_t ** inflow);

int lb_bc_inflow_u_free(lb_bc_inflow_u_t * inflow);
int lb_bc_inflow_u_update(lb_bc_inflow_u_t * inflow, hydro_t * hydro);
int lb_bc_inflow_u_impose(lb_bc_inflow_u_t * inflow, hydro_t * hydro,
			  lb_t * lb);
int lb_bc_inflow_u_stats(lb_bc_inflow_u_t * inflow);

#endif

