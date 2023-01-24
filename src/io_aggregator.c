/*****************************************************************************
 *
 *  io_aggregator.c
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

#include "io_aggregator.h"

/*****************************************************************************
 *
 *  io_aggregator_create
 *
 *  Allocate memory and initialise.
 *
 *****************************************************************************/

int io_aggregator_create(io_element_t el, cs_limits_t lim,
			 io_aggregator_t ** aggr) {

  io_aggregator_t * newaggr = NULL;

  assert(aggr);

  newaggr = (io_aggregator_t *) calloc(1, sizeof(io_aggregator_t));
  if (newaggr == NULL) goto err;

  if (0 != io_aggregator_initialise(el, lim, newaggr)) goto err;

  *aggr = newaggr;

  return 0;

 err:
  if (newaggr) free(newaggr);
  return -1;
}

/*****************************************************************************
 *
 *  io_aggregator_free
 *
 *****************************************************************************/

int io_aggregator_free(io_aggregator_t ** aggr) {

  assert(aggr);

  io_aggregator_finalise(*aggr);
  free(*aggr);

  *aggr = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  io_aggregator_initialise
 *
 *****************************************************************************/

int io_aggregator_initialise(io_element_t e, cs_limits_t lim,
			     io_aggregator_t * aggr) {

  assert(aggr);

  *aggr = (io_aggregator_t) {0};

  aggr->element = e;
  aggr->szelement = e.count*e.datasize;
  aggr->szbuf = aggr->szelement*cs_limits_size(lim);
  aggr->lim = lim;

  if (aggr->szbuf == 0) goto err;

  aggr->buf = (char *) malloc(aggr->szbuf*sizeof(char));

  if (aggr->buf == NULL) goto err;

  return 0;

 err:

  *aggr = (io_aggregator_t) {0};
  return -1;
}

/*****************************************************************************
 *
 * io_aggregator_finalise
 *
 *****************************************************************************/

int io_aggregator_finalise(io_aggregator_t * aggr) {

  assert(aggr);
  assert(aggr->buf);

  free(aggr->buf);

  *aggr = (io_aggregator_t) {0};

  return 0;
}
