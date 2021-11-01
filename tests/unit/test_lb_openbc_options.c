/*****************************************************************************
 *
 *  test_lb_openbc_options.c
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

#include "lb_openbc_options.h"

__host__ int test_lb_openbc_options_inflow_default(void);
__host__ int test_lb_openbc_options_outflow_default(void);
__host__ int test_lb_openbc_options_inflow_valid(void);
__host__ int test_lb_openbc_options_outflow_valid(void);
__host__ int test_lb_openbc_options_flow_valid(void);

/*****************************************************************************
 *
 *  test_lb_openbc_options_suite
 *
 *****************************************************************************/

__host__ int test_lb_openbc_options_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_openbc_options_inflow_default();
  test_lb_openbc_options_outflow_default();
  test_lb_openbc_options_inflow_valid();
  test_lb_openbc_options_outflow_valid();
  test_lb_openbc_options_flow_valid();

  pe_info(pe, "PASS     ./unit/test_lb_openbc_options\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_openbc_options_inflow_default
 *
 *****************************************************************************/

__host__ int test_lb_openbc_options_inflow_default(void) {

  int isvalid = 0;

  {
    lb_inflow_enum_t bctype = lb_openbc_options_inflow_default();
    assert(bctype == LB_INFLOW_NONE);
    isvalid = lb_openbc_options_inflow_valid(bctype);
  }

  return isvalid;
} 

/*****************************************************************************
 *
 *  test_lb_openbc_options_outflow_default
 *
 *****************************************************************************/

__host__ int test_lb_openbc_options_outflow_default(void) {

  int isvalid = 0;

  {
    lb_outflow_enum_t bctype = lb_openbc_options_outflow_default();

    assert(bctype == LB_OUTFLOW_NONE);
    isvalid = lb_openbc_options_outflow_valid(bctype);
  }

  return isvalid;
}

/*****************************************************************************
 *
 *  test_lb_openbc_options_default
 *
 *****************************************************************************/

__host__ int test_lb_openbc_options_default(void) {

  int isvalid = 0;

  {
    lb_openbc_options_t options = lb_openbc_options_default();

    assert(options.bctype == 0);
    assert(options.inflow == LB_INFLOW_NONE);
    assert(options.outflow == LB_OUTFLOW_NONE);

    isvalid = lb_openbc_options_valid(&options);
  }

  return isvalid;
}



/*****************************************************************************
 *
 *  test_lb_openbc_options_inflow_valid
 *
 *****************************************************************************/

__host__ int test_lb_openbc_options_inflow_valid(void) {

  int isvalid = 0;

  {
    lb_inflow_enum_t bctype = LB_INFLOW_NONE;

    isvalid = lb_openbc_options_inflow_valid(bctype);
    assert(isvalid);
  }

  {
    lb_inflow_enum_t bctype = LB_INFLOW_MAX;

    isvalid = lb_openbc_options_inflow_valid(bctype);
    assert(!isvalid);
  }

  return isvalid;
}


/*****************************************************************************
 *
 *  test_lb_openbc_options_outflow_valid
 *
 *****************************************************************************/

__host__ int test_lb_openbc_options_outflow_valid(void) {

  int isvalid = 0;

  {
    lb_outflow_enum_t bctype = LB_OUTFLOW_NONE;

    isvalid = lb_openbc_options_outflow_valid(bctype);
    assert(isvalid);
  }

  {
    lb_outflow_enum_t bctype = LB_OUTFLOW_MAX;

    isvalid = lb_openbc_options_outflow_valid(bctype);
    assert(!isvalid);
  }

  return isvalid;
}

/*****************************************************************************
 *
 *  test_lb_openbc_options_flow_valid
 *
 *****************************************************************************/

__host__ int test_lb_openbc_options_flow_valid(void) {

  int isvalid = 0;

  {
    int flow[3] = {0, 0, 0};

    isvalid = lb_openbc_options_flow_valid(flow);
    assert(isvalid);
  }

  {
    int flow[3] = {0, 1, 0};

    isvalid = lb_openbc_options_flow_valid(flow);
    assert(isvalid);
  }

  {
    int flow[3] = {1, 1, 0};

    isvalid = lb_openbc_options_flow_valid(flow);
    assert(!isvalid);
  }

  {
    int flow[3] = {1, 1, 1};

    isvalid = lb_openbc_options_flow_valid(flow);
    assert(!isvalid);
  }

  return isvalid;
}

