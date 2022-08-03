/*****************************************************************************
 *
 *  io_aggr_buf.c
 *
 *  Temporary buffer for lattice quantity i/o. A minimal container.
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

#include "io_aggr_buf.h"

/*****************************************************************************
 *
 *  io_aggr_buf_create
 *
 *****************************************************************************/

int io_aggr_buf_create(size_t lsz, cs_limits_t lim, io_aggr_buf_t * aggr) {

  assert(aggr);
  assert(cs_limits_size(lim) > 0); /* No zero-size buffers */

  *aggr = (io_aggr_buf_t) {0};

  aggr->szelement = lsz;
  aggr->szbuf = lsz*sizeof(char)*cs_limits_size(lim);
  aggr->lim = lim;

  aggr->buf = (char *) malloc(aggr->szbuf*sizeof(char));
  assert(aggr->buf);

  return 0;
}

/*****************************************************************************
 *
 * io_aggr_buf_free
 *
 *****************************************************************************/

int io_aggr_buf_free(io_aggr_buf_t * aggr) {

  assert(aggr);
  assert(aggr->buf);

  free(aggr->buf);

  *aggr = (io_aggr_buf_t) {0};

  return 0;
}

