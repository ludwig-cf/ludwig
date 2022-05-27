/*****************************************************************************
 *
 *  phi_bc_inflow_opts.c
 *
 *  Inflow options for composition open boundary condition.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcced.ac.uk)
 *
 *****************************************************************************/

#include "phi_bc_inflow_opts.h"

/*****************************************************************************
 *
 *  phi_bc_inflow_opts_default
 *
 *****************************************************************************/

phi_bc_inflow_opts_t phi_bc_inflow_opts_default(void) {

  phi_bc_inflow_opts_t options = {0}; /* Uninterestingly, NULL */

  return options;
}

/*****************************************************************************
 *
 *  phi_bc_inflow_opts_valid
 *
 *****************************************************************************/

int phi_bc_inflow_opts_valid(phi_bc_inflow_opts_t options) {

  int valid = 0; /* Invalid */

  /* No condition on phi0 */
  valid = phi_bc_inflow_opts_flow_valid(options.flow);
  
  return valid;
}

/*****************************************************************************
 *
 *  phi_bc_inflow_opts_flow_valid
 *
 *****************************************************************************/

int phi_bc_inflow_opts_flow_valid(const int flow[3]) {

  int valid = 0;                           /* Invalid */
  int sum = flow[0] + flow[1] + flow[2];

  /* No flow is ok; otherwise one direction only. */
  valid = (sum == 0 || sum == 1);

  return valid;
}
