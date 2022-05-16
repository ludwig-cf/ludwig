/*****************************************************************************
 *
 *  lb_bc_outflow_opts.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "lb_bc_outflow_opts.h"

/*****************************************************************************
 *
 *  lb_bc_outflow_opts_default
 *
 *****************************************************************************/

lb_bc_outflow_opts_t lb_bc_outflow_opts_default(void) {

  lb_bc_outflow_opts_t opts = {.nvel = 19, .flow = {0}, .rho0 = 1.0};

  return opts;
}

/*****************************************************************************
 *
 *  lb_bc_outflow_options_valid
 *
 *****************************************************************************/

int lb_bc_outflow_opts_valid(lb_bc_outflow_opts_t options) {

  int isvalid = 0; /* 0 = invalid */

  int sum = options.flow[X] + options.flow[Y] + options.flow[Z];

  isvalid = lb_model_is_available(options.nvel); /* Available */
  isvalid = (isvalid && (sum == 0 || sum == 1)); /* One direction, or none */

  /* We're not making any stipulations about rho. */

  return isvalid;
}
