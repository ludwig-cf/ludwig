/*****************************************************************************
 *
 *  lb_bc_outflow_rhou.h
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

#ifndef LB_BC_OUTFLOW_RHOU_H
#define LB_BC_OUTFLOW_RHOU_H

#include "lb_bc_open.h"
#include "lb_bc_outflow_opts.h"

typedef struct lb_bc_outflow_rhou_s lb_bc_outflow_rhou_t;

struct lb_bc_outflow_rhou_s {
  lb_bc_open_t super;            /* Superclass block "abstract class" */
  pe_t * pe;                     /* Parallel environment */
  cs_t * cs;                     /* Coordinate system */
  lb_bc_outflow_opts_t options;  /* Parameters/options */
  
  /* Boundary links */
  int      nlink;                /* Number of links (local) */
  int    * linki;                /* Fluid site boundary region */
  int    * linkj;                /* Fluid site in domain proper */
  int8_t * linkp;                /* Velocity index in LB basis (i->j) */
};

__host__ int lb_bc_outflow_rhou_create(pe_t * pe, cs_t * cs,
				       const lb_bc_outflow_opts_t * options,
				       lb_bc_outflow_rhou_t ** outflow);
__host__ int lb_bc_outflow_rhou_free(lb_bc_outflow_rhou_t * outflow);

__host__ int lb_bc_outflow_rhou_update(lb_bc_outflow_rhou_t * outflow,
				       hydro_t * hydro);
__host__ int lb_bc_outflow_rhou_impose(lb_bc_outflow_rhou_t * outflow,
				       hydro_t * hydro, lb_t * lb);
__host__ int lb_bc_outflow_rhou_stats(lb_bc_outflow_rhou_t * outflow);

#endif
