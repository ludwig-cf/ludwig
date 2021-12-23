/*****************************************************************************
 *
 *  test_phi_bc_inflow_opts.c
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
#include <float.h>
#include <math.h>

#include "pe.h"
#include "phi_bc_inflow_opts.h"

__host__ int test_phi_bc_inflow_opts_default(void);
__host__ int test_phi_bc_inflow_opts_valid(void);
__host__ int test_phi_bc_inflow_opts_flow_valid(void);

/*****************************************************************************
 *
 *  test_phi_bc_inflow_opts_suite
 *
 *****************************************************************************/

__host__ int test_phi_bc_inflow_opts_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_phi_bc_inflow_opts_default();
  test_phi_bc_inflow_opts_valid();
  test_phi_bc_inflow_opts_flow_valid();

  pe_info(pe, "PASS     ./unit/test_phi_bc_inflow_opts\n");
  pe_free(pe);
  
  return 0;
}

/*****************************************************************************
 *
 *  test_phi_bc_inflow_opts_default
 *
 *****************************************************************************/

__host__ int test_phi_bc_inflow_opts_default(void) {

  int isvalid = 0;
  phi_bc_inflow_opts_t options = phi_bc_inflow_opts_default();

  assert(fabs(options.phi0 - 0.0) < DBL_EPSILON);
  assert(options.flow[0] == 0);
  assert(options.flow[1] == 0);
  assert(options.flow[2] == 0);

  isvalid = phi_bc_inflow_opts_valid(options);
  assert(isvalid);
  
  return isvalid;
}

/*****************************************************************************
 *
 *  test_phi_bc_inflow_opts_valid
 *
 *****************************************************************************/

__host__ int test_phi_bc_inflow_opts_valid(void) {

  int isvalid = 0;

  {
    phi_bc_inflow_opts_t options = {.phi0 = 0.0, .flow = {1,0,0}}; /* valid */
    isvalid = phi_bc_inflow_opts_valid(options);
    assert(isvalid);
  }

  {
    phi_bc_inflow_opts_t options = {.phi0 = 0.0, .flow = {1,1,1}}; /* not */
    isvalid = phi_bc_inflow_opts_valid(options);
    assert(isvalid == 0);
  }
    
  return isvalid;
}

/*****************************************************************************
 *
 *  test_phi_bc_inflow_opts_flow_valid
 *
 *****************************************************************************/

__host__ int test_phi_bc_inflow_opts_flow_valid(void) {

  int isvalid = 0;

  {
    int flow[3] = {0, 0, 0};
    isvalid = phi_bc_inflow_opts_flow_valid(flow);
    assert(isvalid);
  }

  {
    int flow[3] = {1, 0, 0};
    isvalid = phi_bc_inflow_opts_flow_valid(flow);
    assert(isvalid);
  }

  {
    int flow[3] = {0, 1, 0};
    isvalid = phi_bc_inflow_opts_flow_valid(flow);
    assert(isvalid);
  }

  {
    int flow[3] = {0, 0, 1};
    isvalid = phi_bc_inflow_opts_flow_valid(flow);
    assert(isvalid);
  }

  {
    int flow[3] = {1, 0, 1};
    isvalid = phi_bc_inflow_opts_flow_valid(flow);
    assert(isvalid == 0);
  }

  return isvalid;
}
