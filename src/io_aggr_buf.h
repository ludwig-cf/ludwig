/*****************************************************************************
 *
 *  io_aggr_buf.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_AGGR_BUF_H
#define LUDWIG_IO_AGGR_BUF_H

#include "cs_limits.h"

typedef struct io_aggr_buf_s io_aggr_buf_t;

struct io_aggr_buf_s {
  size_t szelement;    /* element sz in bytes */
  size_t szbuf;        /* total sz */
  cs_limits_t lim;     /* 3-d limits of buffer */
  char * buf;
};

int io_aggr_buf_create(size_t lsz, cs_limits_t lim, io_aggr_buf_t * aggr);
int io_aggr_buf_free(io_aggr_buf_t * aggr);

#endif
