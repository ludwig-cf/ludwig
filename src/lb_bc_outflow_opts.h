/*****************************************************************************
 *
 *  lb_bc_outflow_opts.h
 *
 *  Container for outflow boundary condition options.
 *
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LB_BC_OUTFLOW_OPTS_H
#define LB_BC_OUTFLOW_OPTS_H

#include "lb_model.h"

typedef struct lb_bc_outflow_opts_s lb_bc_outflow_opts_t;

struct lb_bc_outflow_opts_s {
  int nvel;
  int flow[3];
  double rho0;
  double u0[3];
};

lb_bc_outflow_opts_t lb_bc_outflow_opts_default(void);
int lb_bc_outflow_opts_valid(lb_bc_outflow_opts_t options);

#endif
