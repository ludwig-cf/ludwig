/*****************************************************************************
 *
 *  lb_bc_inflow_opts.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Comuting Centre
 *
 *  (c) 2021 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_BC_INFLOW_OPTS_H
#define LUDWIG_LB_BC_INFLOW_OPTS_H

#include "lb_model.h"

/* TYPES */

typedef struct lb_bc_inflow_opts_s lb_bc_inflow_opts_t;

struct lb_bc_inflow_opts_s {
  int nvel;                    /* Model */
  int flow[3];                 /* Flow coordinate direction (exactly 1) */ 
  double u0[3];                /* A velocity for the boundary */
};

lb_bc_inflow_opts_t lb_bc_inflow_opts_default(void);

int lb_bc_inflow_opts_valid(lb_bc_inflow_opts_t options);
int lb_bc_inflow_opts_flow_valid(const int flow[3]);

#endif
