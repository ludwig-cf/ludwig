/*****************************************************************************
 *
 *  io_aggregator.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_AGGREGATOR_H
#define LUDWIG_IO_AGGREGATOR_H

#include <stdlib.h>
#include "cs_limits.h"
#include "io_element.h"

typedef struct io_aggregator_s io_aggregator_t;

struct io_aggregator_s {
  io_element_t element;     /* Element information */
  cs_limits_t lim;          /* 3-d limits of buffer */
  size_t szelement;         /* bytes per record */
  size_t szbuf;             /* total size of buffer (bytes) */
  char * buf;               /* Storage space */
};

int io_aggregator_create(io_element_t el, cs_limits_t lim,
			 io_aggregator_t ** aggr);
int io_aggregator_free(io_aggregator_t ** aggr);

int io_aggregator_initialise(io_element_t el, cs_limits_t lim,
			     io_aggregator_t * aggr);
int io_aggregator_finalise(io_aggregator_t * aggr);

#endif
