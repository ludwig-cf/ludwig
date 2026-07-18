/*****************************************************************************
 *
 *  lb_bc_outflow_p.h
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

#ifndef LB_BC_OUTFLOW_P_H
#define LB_BC_OUTFLOW_P_H

#include "lb_bc_open.h"
#include "lb_bc_outflow_opts.h"

typedef struct lb_bc_outflow_p_s lb_bc_outflow_p_t;

struct lb_bc_outflow_p_s {
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

int lb_bc_outflow_p_create(pe_t * pe, cs_t * cs,
			   const lb_bc_outflow_opts_t * options,
			   lb_bc_outflow_p_t ** outflow);
int lb_bc_outflow_p_free(lb_bc_outflow_p_t * outflow);

int lb_bc_outflow_p_update(lb_bc_outflow_p_t * outflow, hydro_t * hydro);
int lb_bc_outflow_p_impose(lb_bc_outflow_p_t * outflow,
			   hydro_t * hydro, lb_t * lb);
int lb_bc_outflow_p_stats(lb_bc_outflow_p_t * outflow);

#endif
