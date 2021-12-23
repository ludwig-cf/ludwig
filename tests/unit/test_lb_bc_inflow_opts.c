/*****************************************************************************
 *
 *  test_lb_bc_inflow_opts.c
 *
 *
 *  Edinburgh Solft Matter and Statistical Physics Group and
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

#include "pe.h"
#include "lb_bc_inflow_opts.h"

__host__ int test_lb_bc_inflow_opts_default(void);
__host__ int test_lb_bc_inflow_opts_flow_valid(void);
__host__ int test_lb_bc_inflow_opts_valid(void);

/*****************************************************************************
 *
 *  test_lb_bc_inflow_opts_suite
 *
 *****************************************************************************/

__host__ int test_lb_bc_inflow_opts_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_bc_inflow_opts_default();
  test_lb_bc_inflow_opts_flow_valid();
  test_lb_bc_inflow_opts_valid();

  pe_info(pe, "PASS     ./unit/test_lb_bc_inflow_opts\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_bc_inflow_opts_default
 *
 *****************************************************************************/

__host__ int test_lb_bc_inflow_opts_default(void) {

  int isvalid = 0;

  {
    lb_bc_inflow_opts_t options = lb_bc_inflow_opts_default();

    assert(options.nvel == 19);
    assert(options.flow[X] == 0);
    assert(options.flow[Y] == 0);
    assert(options.flow[Z] == 0);

    isvalid = lb_bc_inflow_opts_valid(options);
  }

  return isvalid;
}

/*****************************************************************************
 *
 *  test_lb_bc_inflow_opts_flow_valid
 *
 *****************************************************************************/

__host__ int test_lb_bc_inflow_opts_flow_valid(void) {

  int isvalid = 0;

  {
    int flow[3] = {0, 0, 0};

    isvalid = lb_bc_inflow_opts_flow_valid(flow);
    assert(isvalid);
  }

  {
    int flow[3] = {0, 1, 0};

    isvalid = lb_bc_inflow_opts_flow_valid(flow);
    assert(isvalid);
  }

  {
    int flow[3] = {1, 1, 0};

    isvalid = lb_bc_inflow_opts_flow_valid(flow);
    assert(!isvalid);
  }

  {
    int flow[3] = {1, 1, 1};

    isvalid = lb_bc_inflow_opts_flow_valid(flow);
    assert(!isvalid);
  }

  return isvalid;
}

/*****************************************************************************
 *
 *  test_lb_bc_inflow_opts_valid
 *
 *****************************************************************************/

__host__ int test_lb_bc_inflow_opts_valid(void) {

  int ierr = 0;

  {
    lb_bc_inflow_opts_t options = {.nvel = 9};

    ierr = lb_bc_inflow_opts_valid(options);
    assert(ierr);
  }
  
  {
    lb_bc_inflow_opts_t options = {}; /* Not valid */

    ierr = lb_bc_inflow_opts_valid(options);
    assert(ierr == 0);
  }

  return ierr;
}
					     
