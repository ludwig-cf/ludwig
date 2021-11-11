/*****************************************************************************
 *
 *  lb_openbc_options.c
 *
 *  Options container for fluid open boundary conditions.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

#include "lb_openbc_options.h"

/* Defaults */

#define LB_INFLOW_DEFAULT()         LB_INFLOW_NONE
#define LB_OUTFLOW_DEFAULT()        LB_OUTFLOW_NONE
#define LB_FLOW_DEFAULT()           {0,0,0}
#define LB_FLOW_U0_DEFAULT()        {0.0,0.0,0.0}

/*****************************************************************************
 *
 *  lb_openbc_options_inflow_default
 *
 *****************************************************************************/

__host__ lb_inflow_enum_t lb_openbc_options_inflow_default(void) {

  return LB_INFLOW_DEFAULT();
}

/*****************************************************************************
 *
 *  lb_options_openbc_outflow_default
 *
 *****************************************************************************/

__host__ lb_outflow_enum_t lb_openbc_options_outflow_default(void) {

  return LB_OUTFLOW_DEFAULT();
}

/*****************************************************************************
 *
 *  lb_openbc_options_default
 *
 *****************************************************************************/

__host__ lb_openbc_options_t lb_openbc_options_default(void) {

  lb_openbc_options_t options = {.bctype  = 0,
                                 .inflow  = LB_INFLOW_DEFAULT(),
                                 .outflow = LB_OUTFLOW_DEFAULT(),
                                 .flow    = LB_FLOW_DEFAULT(),
                                 .u0      = LB_FLOW_U0_DEFAULT()};

  return options;
}

/*****************************************************************************
 *
 *  lb_openbc_options_valid
 *
 *****************************************************************************/

__host__ int lb_openbc_options_valid(const lb_openbc_options_t * options) {

  int valid = 0; /* 0 = invalid */

  valid += lb_openbc_options_inflow_valid(options->inflow);
  valid += lb_openbc_options_outflow_valid(options->outflow);
  valid += lb_openbc_options_flow_valid(options->flow);

  return valid;
}

/*****************************************************************************
 *
 *  lb_openbc_options_inflow_valid
 *
 *****************************************************************************/

__host__ int lb_openbc_options_inflow_valid(lb_inflow_enum_t inflow) {

  int isvalid = 0; /* 0 = invalid */

  isvalid += (inflow == LB_INFLOW_NONE);

  return isvalid;
}

/*****************************************************************************
 *
 *  lb_openbc_options_outflow_valid
 *
 *****************************************************************************/

__host__ int lb_openbc_options_outflow_valid(lb_outflow_enum_t outflow) {

  int isvalid = 0; /* 0 = invalid */

  isvalid += (outflow == LB_OUTFLOW_NONE);

  return isvalid;
}

/*****************************************************************************
 *
 *  lb_openbc_options_flow_valid
 *
 *****************************************************************************/

__host__ int lb_openbc_options_flow_valid(const int flow[3]) {

  int isvalid = 0;
  int sum = flow[0] + flow[1] + flow[2];

  /* No flow is ok; otherwise only one direction. */
  isvalid = (sum == 0) || (sum == 1);

  return isvalid;
}

/*****************************************************************************
 *
 *  lb_openbc_options_info
 *
 *****************************************************************************/

__host__ int lb_openbc_options_info(pe_t * pe,
				    const lb_openbc_options_t * options) {

  assert(pe);
  assert(options);

  pe_info(pe, "bctype:    %d\n", options->bctype);
  pe_info(pe, "inflow:    %d\n", options->inflow);
  pe_info(pe, "outflow:   %d\n", options->outflow);
  pe_info(pe, "flow:      %1d %1d %1d\n", options->flow[0], options->flow[1],
	  options->flow[2]);

  return 0;
}
