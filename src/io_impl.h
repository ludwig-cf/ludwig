/*****************************************************************************
 *
 *  io_impl.h
 *
 *  Abstraction of i/o implementations.
 *
 *  An implementation is expected  to support synchronous i/o,
 *  but may not support asynchronous i/o. In addtion, only
 *  asynchronous write is considered at the moment.
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

#ifndef LUDWIG_IO_IMPL_H
#define LUDWIG_IO_IMPL_H

#include "io_metadata.h"
#include "io_aggregator.h"

typedef struct io_impl_vt_s io_impl_vt_t;
typedef struct io_impl_s io_impl_t;

/* General */

typedef int (* io_impl_free_ft) (io_impl_t ** io);

/* Synchronous implmentations */

typedef int (* io_impl_read_ft) (io_impl_t * io, const char * filename);
typedef int (* io_impl_write_ft) (io_impl_t * io, const char * filename);

/* Asynchronous implementation may also supply, for writing ... */

typedef int (* io_impl_write_begin_ft) (io_impl_t * io, const char * filename);
typedef int (* io_impl_write_end_ft) (io_impl_t * io);

struct io_impl_vt_s {
  io_impl_free_ft         free;         /* Destructor */
  io_impl_read_ft         read;         /* Synchronous read */
  io_impl_write_ft        write;        /* Synchronous write */

  io_impl_write_begin_ft  write_begin;  /* Asynchronous start */
  io_impl_write_end_ft    write_end;    /* Asynchronous end */
};

struct io_impl_s {
  const io_impl_vt_t * impl;            /* Implementation */
  io_aggregator_t    * aggr;            /* Implementation has an aggregator */
};

/* Factory method to instantiate a concrete object ... */

int io_impl_create(const io_metadata_t * metadata, io_impl_t ** io);

#endif
