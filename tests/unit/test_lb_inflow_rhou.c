/*****************************************************************************
 *
 *  test_lb_inflow_rhou.c
 *
 *
 *****************************************************************************/

#include <assert.h>

#include "lb_inflow_rhou.c"

/* Tests */

__host__ int test_lb_inflow_rhou_create(pe_t * pe, cs_t * cs);
__host__ int test_lb_inflow_rhou_update(pe_t * pe, cs_t * cs);

/*****************************************************************************
 *
 *  test_lb_inflow_rhou_suite
 *
 *****************************************************************************/

__host__ int test_lb_inflow_rhou_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_lb_inflow_rhou_create(pe, cs);
  test_lb_inflow_rhou_update(pe, cs);

  pe_info(pe, "PASS     ./unit/test_lb_inflow_rhou\n");

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_inflow_rhou_create
 *
 *****************************************************************************/

__host__ int test_lb_inflow_rhou_create(pe_t * pe, cs_t * cs) {

  lb_openbc_options_t options = lb_openbc_options_default();
  lb_inflow_rhou_t * inflow = NULL;

  assert(pe);
  assert(cs);

  lb_inflow_rhou_create(pe, cs, &options, &inflow);

  assert(inflow);
  assert(inflow->pe == pe);
  assert(inflow->cs = cs);

  assert(inflow->super.func);
  assert(inflow->super.id == LB_OPEN_BC_INFLOW_RHOU);

  /* Check options */

  assert(inflow->options.inflow);

  /* Default options given no links */

  assert(inflow->nlink == 0);
  assert(inflow->linkp);
  assert(inflow->linki);
  assert(inflow->linkj);

  lb_inflow_rhou_free(inflow);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_inflow_rhou_update
 *
 *****************************************************************************/

__host__ int test_lb_inflow_rhou_update(pe_t * pe, cs_t * cs) {

  assert(pe);
  assert(cs);


  return 0;
}
