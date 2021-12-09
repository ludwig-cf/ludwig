/*****************************************************************************
 *
 *  phi_bc_open_rt.c
 *
 *  Generate paired instance of inflow and outflow boundary conditions
 *  for composition.
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
#include <string.h>

#include "phi_bc_open_rt.h"


/* Available implementations */

#include "phi_bc_inflow_fixed.h"
#include "phi_bc_outflow_free.h"

/*****************************************************************************
 *
 *  phi_bc_open_rt
 *
 *  A factory method to generate both inflow and outflow (there must
 *  be either none or both) from user input.
 *
 *****************************************************************************/

__host__ int phi_bc_open_rt(pe_t * pe, rt_t * rt, cs_t * cs,
			    phi_bc_open_t ** inflow,
			    phi_bc_open_t ** outflow) {

  int have_bc = 0;  /* inflow/outflow required? */
  int flow[3] = {}; /* flow direction (one only, or none) */

  assert(pe);
  assert(cs);
  assert(rt);
  assert(inflow);
  assert(outflow);

  have_bc = rt_switch(rt, "phi_bc_open");

  if (have_bc) {
    int wall[3] = {};
    int periodic[3] = {};

    /* Take flow direction from non-wall direction */

    rt_int_parameter_vector(rt, "boundary_walls", wall);
    flow[X] = 1 - wall[X];
    flow[Y] = 1 - wall[Y];
    flow[Z] = 1 - wall[Z];

    if (phi_bc_inflow_opts_flow_valid(flow) == 0) {
      pe_fatal(pe, "Inflow/outflow requires exactly one open direction\n");
    }

    cs_periodic(cs, periodic);
    if (periodic[X] || periodic[Y] || periodic[Z]) {
      pe_fatal(pe, "Inflow/outflow requires fully non-periodic system\n");
    }

    rt_key_required(rt, "phi_bc_inflow_type",  RT_INFO);
    rt_key_required(rt, "phi_bc_outflow_type", RT_INFO);
  }

  /* Inflow */

  if (have_bc) {
    char intype[BUFSIZ] = {};
    double phib = -999.999;

    rt_string_parameter(rt, "phi_bc_inflow_type", intype, BUFSIZ);
    rt_double_parameter(rt, "phi_bc_inflow_fixed_phib", &phib);

    pe_info(pe, "\n");
    pe_info(pe, "Order parameter\n");
    pe_info(pe, "---------------\n\n");
    pe_info(pe, "Inflow open boundary for composition (phi)\n");

    if (strncmp(intype, "fixed", BUFSIZ) == 0) {
      /* A fixed-type inflow */
      phi_bc_inflow_fixed_t * bc = NULL;
      phi_bc_inflow_opts_t options = {.phi0 = phib,
	                              .flow = {flow[X], flow[Y], flow[Z]}};

      if (phi_bc_inflow_opts_valid(options) == 0) {
	/* Further diagnostic information. */
	pe_fatal(pe, "Please check phi_bc_inflow options\n");
      }

      phi_bc_inflow_fixed_create(pe, cs, &options, &bc);
      *inflow = (phi_bc_open_t *) bc;

      pe_info(pe, "Composition inflow condition:   %s\n", "fixed");
      pe_info(pe, "Composition inflow direction:   %d %d %d\n",
	      flow[X], flow[Y], flow[Z]);
      pe_info(pe, "Composition inflow phi_b:      %14.7e\n", phib);
    }
    else {
      pe_info(pe, "phi_bc_inflow_type not recognised: %s\n", intype);
      pe_fatal(pe, "Please check and try again.\n");
    }
  }

  /* Outflow */

  if (have_bc) {

    char outtype[BUFSIZ] = {};

    rt_string_parameter(rt, "phi_bc_outflow_type", outtype, BUFSIZ);

    pe_info(pe, "\n");
    pe_info(pe, "Outflow open boundary for composition (phi)\n");

    if (strncmp(outtype, "free", BUFSIZ) == 0) {
      phi_bc_outflow_free_t * bc = NULL;
      phi_bc_outflow_opts_t options = {.flow = {flow[X], flow[Y], flow[Z]}};

      /* Check options valid */
      if (phi_bc_outflow_opts_valid(options) == 0) {
	pe_fatal(pe, "phi_bc_outflow_fixed options not valid\n");
      }

      phi_bc_outflow_free_create(pe, cs, &options, &bc);
      *outflow = (phi_bc_open_t *) bc;

      pe_info(pe, "Composition outflow condition:  %s\n", "free");
    }
    else {
      pe_fatal(pe, "phi_bc_outflow_type not recognised: %s\n", outtype);
    }
  }

  return 0;
}
