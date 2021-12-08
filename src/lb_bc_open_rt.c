/*****************************************************************************
 *
 *  lb_bc_open_rt.c
 *
 *  LB open boundary condition initialisation.
 *
 *  Currently acceptable positions:
 *    We must have wall in two coordinate directions (3d)
 *    We must have no periodic conditions at all "periodicity 0_0_0"
 *    The inflow/outflow direction must be in the non-wall direction.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contibuting authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "lb_bc_open_rt.h"

/* Available implementations */

#include "lb_bc_inflow_rhou.h"
#include "lb_bc_outflow_rhou.h"

/*****************************************************************************
 *
 *  lb_bc_open_rt
 *
 *  As we are not generally going to have an inflow without an outflow
 *  (and vice-versa) these are together...
 *
 *****************************************************************************/

__host__ int lb_bc_open_rt(pe_t * pe, rt_t * rt, cs_t * cs, lb_t * lb,
			   lb_bc_open_t ** inflow,
			   lb_bc_open_t ** outflow) {

  int have_bc = 0;   /* inflow/outflow required */
  int flow[3] = {};  /* flow direction */
  
  assert(pe);
  assert(rt);
  assert(cs);
  assert(lb);

  have_bc = rt_switch(rt, "lb_bc_open");

  if (have_bc) {
    int wall[3] = {};
    int periodic[3] = {};

    /* Take flow direction from non-wall direction. */
    /* There must be D-1 walls in D dimensions */

    rt_int_parameter_vector(rt, "boundary_walls", wall);
    flow[X] = 1 - wall[X];
    flow[Y] = 1 - wall[Y];
    flow[Z] = 1 - wall[Z];

    if (lb_bc_inflow_opts_flow_valid(flow) == 0) {
      pe_fatal(pe, "Inflow/outflow requires exactly one-non wall direction\n");
    }
    
    /* Test periodicity. Must be (0,0,0). */

    cs_periodic(cs, periodic);
    if (periodic[X] || periodic[Y] || periodic[Z]) {
      pe_fatal(pe, "Inflow/outflow requires fully non-periodic system\n");
    }

    /* Any  other checks are deferred ... */

    rt_key_required(rt, "lb_bc_inflow_type",  RT_INFO);
    rt_key_required(rt, "lb_bc_outflow_type", RT_INFO);
  }
  
  /* Inflow */

  if (have_bc) {
 
    char intype[BUFSIZ] = {};
    double u0[3] = {};

    rt_string_parameter(rt, "lb_bc_inflow_type", intype, BUFSIZ/2);
    rt_double_parameter_vector(rt, "lb_bc_inflow_rhou_u0", u0);

    pe_info(pe, "\n");
    pe_info(pe, "Hydrodynamic open boundary condition for inflow\n");

    if (strncmp(intype, "rhou", BUFSIZ) == 0) {
      /* Give me a rhou inflow */
      lb_bc_inflow_rhou_t * rhou = NULL;
      lb_bc_inflow_opts_t options = {.nvel = lb->model.nvel,
	                             .flow = {flow[X], flow[Y], flow[Z]},
	                             .u0   = {u0[X],u0[Y],u0[Z]}};

      /* Check options are valid */
      if (lb_bc_inflow_opts_valid(options) == 0) {
	/* Print them out */
	pe_fatal(pe, "Please check inflow options and try again\n.");
      }

      lb_bc_inflow_rhou_create(pe, cs, &options, &rhou);
      *inflow = (lb_bc_open_t *) rhou;

      /* Might be nice to delegate this elsewhere ... */
      pe_info(pe, "Inflow type:              %s\n", "rhou");
      pe_info(pe, "Inflow flow profile:      %s\n", "uniform");
      pe_info(pe, "Inflow flow direction:    %d %d %d\n",
	      flow[X], flow[Y], flow[Z]);
      pe_info(pe, "Inflow flow value u0:    %14.7e %14.7e %14.7e\n",
	      u0[X], u0[Y], u0[Z]);
    }
    else {
      /* Not recognised */
      pe_fatal(pe, "lb_bc_inflow_type not recognised\n");
    }
  }

  /* Outflow */
  if (have_bc) {

    char outtype[BUFSIZ] = {};
    double rho0 = 1.0;

    rt_string_parameter(rt, "lb_bc_outflow_type", outtype, BUFSIZ/2);
    rt_double_parameter(rt, "rho0", &rho0);
    rt_double_parameter(rt, "lb_bc_outflow_rhou_rho0", &rho0);

    pe_info(pe, "\n");
    pe_info(pe, "Hydrodynamic open boundary condition at outflow\n");
    
    if (strncmp(outtype, "rhou", BUFSIZ) == 0) {
      lb_bc_outflow_rhou_t * rhou = NULL;
      lb_bc_outflow_opts_t options = {.nvel = lb->model.nvel,
	                              .flow = {flow[X], flow[Y], flow[Z]},
	                              .rho0 = rho0};
      /* Check options valid */
      if (lb_bc_outflow_opts_valid(options) == 0) {
	/* Print out? */
	pe_fatal(pe, "Please check outflow options and try again\n");
      }

      lb_bc_outflow_rhou_create(pe, cs, &options, &rhou);
      *outflow = (lb_bc_open_t *) rhou;

      pe_info(pe, "Outflow type:             %s\n", "rhou");
      pe_info(pe, "Outflow flow direction:   %d %d %d\n",
	      flow[X], flow[Y], flow[Z]);
      pe_info(pe, "Outflow flow rho0:       %14.7e\n", rho0);
    }
    else {
      pe_fatal(pe, "lb_bc_outflow_type not recognised\n");
    }
  }

  return 0;
}
