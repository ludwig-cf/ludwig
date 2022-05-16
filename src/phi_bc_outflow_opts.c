/*****************************************************************************
 *
 *  phi_bc_outflow_opts.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "phi_bc_outflow_opts.h"

/*****************************************************************************
 *
 *  phi_bc_outflow_opts_default
 *
 *****************************************************************************/

phi_bc_outflow_opts_t phi_bc_outflow_opts_default(void) {

  phi_bc_outflow_opts_t options = {0};

  return options;
}

/*****************************************************************************
 *
 *  phi_bc_outflow_opts_valid
 *
 *****************************************************************************/

int phi_bc_outflow_opts_valid(phi_bc_outflow_opts_t options) {

  int isvalid = 0; /* Invalid */

  isvalid = phi_bc_outflow_opts_flow_valid(options.flow);

  return isvalid;
}

/*****************************************************************************
 *
 *  phi_bc_outflow_opts_flow_valid
 *
 *****************************************************************************/

int phi_bc_outflow_opts_flow_valid(const int flow[3]) {

  int isvalid = 0; /* Invalid */
  int sum = flow[0] + flow[1] + flow[2];

  isvalid = (sum == 0) || (sum == 1);

  return isvalid;
}
