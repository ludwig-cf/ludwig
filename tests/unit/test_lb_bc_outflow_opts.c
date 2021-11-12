/*****************************************************************************
 *
 *  test_lb_bc_outflow_opts.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "lb_bc_outflow_opts.h"

__host__ int test_lb_bc_outflow_opts_default(void);
__host__ int test_lb_bc_outflow_opts_valid(void);

/*****************************************************************************
 *
 *  test_lb_bc_outflow_opts_suite
 *
 *****************************************************************************/

__host__ int test_lb_bc_outflow_opts_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_bc_outflow_opts_default();
  test_lb_bc_outflow_opts_valid();

  pe_info(pe, "PASS     ./unit/test_lb_bc_outflow_opts\n");
  pe_free(pe);
  
  return 0;
}

/*****************************************************************************
 *
 *  test_lb_bc_outflo_opts_default
 *
 *****************************************************************************/

__host__ int test_lb_bc_outflow_opts_default(void) {

  lb_bc_outflow_opts_t defaults = lb_bc_outflow_opts_default();

  assert(defaults.nvel    == 0);
  assert(defaults.flow[X] == 0);
  assert(defaults.flow[Y] == 0);
  assert(defaults.flow[Z] == 0);

  return defaults.nvel;
}

/*****************************************************************************
 *
 *  test_lb_bc_outflow_opts_valid
 *
 *****************************************************************************/

__host__ int test_lb_bc_outflow_opts_valid(void) {

  int ierr = 0;

  {
    lb_bc_outflow_opts_t options = lb_bc_outflow_opts_default();

    ierr = lb_bc_outflow_opts_valid(options);
    assert(ierr == 0);
  }

  {
    lb_bc_outflow_opts_t options = {.nvel = 9, .flow = {1, 0, 0}};

    ierr = lb_bc_outflow_opts_valid(options);
    assert(ierr != 0);
  }

  {
    lb_bc_outflow_opts_t options = {.nvel = 9, .flow = {1, 1, 0}};

    ierr = lb_bc_outflow_opts_valid(options);
    assert(ierr == 0);
  }
  
  return ierr;
}
