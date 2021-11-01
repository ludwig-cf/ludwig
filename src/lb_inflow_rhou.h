/*****************************************************************************
 *
 *  lb_inflow_rhou.h
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_INFLOW_RHOU_H
#define LUDWIG_LB_INFLOW_RHOU_H

#include "lb_open_bc.h"
#include "lb_openbc_options.h"

typedef struct lb_inflow_rhou_s lb_inflow_rhou_t;

/* Inflow boundary condition. */

struct lb_inflow_rhou_s {
  lb_open_bc_t super;           /* Superclass block */
  pe_t * pe;                    /* Parallel environment */
  cs_t * cs;                    /* Coordinate system */
  lb_openbc_options_t options;  /* Options/parameters */
  lb_inflow_rhou_t * target;    /* Target pointer */
  
  /* Boundary links */
  int nlink;                    /* Number of links (local) */
  int8_t * linkp;               /* Velocity index in LB basis (i->j) */
  int * linki;                  /* Fluid site in system */
  int * linkj;                  /* Fluid site in inflow halo boundary */
};

__host__ int lb_inflow_rhou_create(pe_t * pe, cs_t * cs,
				    const lb_openbc_options_t * options,
				    lb_inflow_rhou_t ** inflow);

__host__ int lb_inflow_rhou_free(lb_inflow_rhou_t * inflow);
__host__ int lb_inflow_rhou_update(lb_inflow_rhou_t * inflow,
				    hydro_t * hydro, lb_t * lb);
__host__ int lb_inflow_rhou_stats(lb_inflow_rhou_t * inflow);

__host__ int lb_inflow_rhou_info(lb_inflow_rhou_t * inflow);

#endif

