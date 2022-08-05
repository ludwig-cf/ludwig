/*****************************************************************************
 *
 *  test_io_aggr_buf.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "io_aggr_buf.h"

int test_io_aggr_buf_create(void);

/*****************************************************************************
 *
 *  test_io_aggr_buf_suite
 *
 *****************************************************************************/

int test_io_aggr_buf_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* If the size of the struct has changed, tests need to be changed... */
  assert(sizeof(io_aggr_buf_t) == 48);

  test_io_aggr_buf_create();

  pe_info(pe, "PASS     ./unit/testr_io_aggr_buf\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_aggr_buf_create
 *
 *****************************************************************************/

int test_io_aggr_buf_create(void) {

  int ifail = 0;

  {
    /* Create and free. */
    size_t lsz = 99;
    cs_limits_t lim = {-2, 18, 1, 8, 1, 4};
    io_aggr_buf_t aggr = {0};

    io_aggr_buf_create(lsz, lim, &aggr);

    assert(aggr.szelement == lsz);
    assert(aggr.szbuf     == lsz*cs_limits_size(lim));
    assert(aggr.lim.imin  == lim.imin); /* Assume sufficient */
    assert(aggr.buf);

    io_aggr_buf_free(&aggr);
    assert(aggr.szelement == 0);
    assert(aggr.szbuf == 0);
    assert(aggr.lim.imin == 0);
    assert(aggr.buf == NULL);
  }

  return ifail;
}
