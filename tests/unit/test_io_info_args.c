/*****************************************************************************
 *
 *  test_io_info_args.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2025 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "io_info_args.h"

int test_io_info_args_default(void);
int test_io_info_args_iogrid_valid(void);

/*****************************************************************************
 *
 *  test_io_info_args_suite
 *
 *****************************************************************************/

int test_io_info_args_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_io_info_args_default();
  test_io_info_args_iogrid_valid();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_info_args_default
 *
 *****************************************************************************/

int test_io_info_args_default(void) {

  int ifail = 0;
  io_info_args_t args = io_info_args_default();

  /* If the size of the struct changes, the tests need updating */

  assert(sizeof(io_info_args_t) == (2*sizeof(io_options_t) + 4*sizeof(int)));

  assert(args.input.mode  == io_mode_default());
  assert(args.output.mode == io_mode_default());
  assert(args.grid[0]     == 1);
  assert(args.grid[1]     == 1);
  assert(args.grid[2]     == 1);
  assert(args.iofreq      == 0);

  if (args.input.mode != 0) ifail += 1;

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_info_args_iogrid_valid
 *
 *****************************************************************************/

int test_io_info_args_iogrid_valid(void) {

  int ifail = 0;

  /* Wrong. Zero not allowed. */
  {
    int iogrid[3] = {0, 1, 1};
    assert(io_info_args_iogrid_valid(iogrid) == 0);
    if (io_info_args_iogrid_valid(iogrid)) ifail = -1;
  }

  /* Right */
  {
    int iogrid[3] = {1, 1, 1};
    assert(io_info_args_iogrid_valid(iogrid) == 1);
    if (io_info_args_iogrid_valid(iogrid) == 0) ifail = -1;
  }

  return ifail;
}
