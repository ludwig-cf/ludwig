/*****************************************************************************
 *
 *  lb_bc_inflow_opts.c
 *
 *  Options container for inflow open boundary conditions.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

#include "lb_bc_inflow_opts.h"


/*****************************************************************************
 *
 *  lb_bc_inflow_opts_default
 *
 *****************************************************************************/

lb_bc_inflow_opts_t lb_bc_inflow_opts_default(void) {

  lb_bc_inflow_opts_t opts = {.nvel = 19, .flow = {0}, .u0 = {0}};

  return opts;
}

/*****************************************************************************
 *
 *  lb_bc_inflow_opts_valid
 *
 *****************************************************************************/

int lb_bc_inflow_opts_valid(lb_bc_inflow_opts_t options) {

  int valid = 0; /* 0 = invalid */

  int oknvel = lb_model_is_available(options.nvel);
  int okflow = lb_bc_inflow_opts_flow_valid(options.flow);

  /* No conditions on u0 at the moment. */

  valid = (oknvel && okflow);

  return valid;
}

/*****************************************************************************
 *
 *  lb_bc_inflow_opts_flow_valid
 *
 *****************************************************************************/

int lb_bc_inflow_opts_flow_valid(const int flow[3]) {

  int isvalid = 0;
  int sum = flow[0] + flow[1] + flow[2];

  /* No flow is ok; otherwise only one direction. */
  isvalid = (sum == 0) || (sum == 1);

  return isvalid;
}
