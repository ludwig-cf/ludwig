/*****************************************************************************
 *
 *  io_impl_mpio.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_IMPL_MPIO_H
#define LUDWIG_IO_IMPL_MPIO_H

#include "io_impl.h"
#include "io_metadata.h"

typedef struct io_impl_mpio_s io_impl_mpio_t;

struct io_impl_mpio_s {
  io_impl_t super;                       /* superclass block */
  const io_metadata_t * metadata;        /* options, element type, ... */

  /* MPIO implementation state ... */
  MPI_File fh;                           /* file handle */
  MPI_Status status;                     /* last status */
  MPI_Datatype element;                  /* element type */
  MPI_Datatype array;                    /* subarray type */
  MPI_Datatype file;                     /* file type */
};

int io_impl_mpio_create(const io_metadata_t * meta, io_impl_mpio_t ** io);
int io_impl_mpio_free(io_impl_mpio_t ** io);

int io_impl_mpio_initialise(const io_metadata_t * meta, io_impl_mpio_t * io);
int io_impl_mpio_finalise(io_impl_mpio_t * io);

int io_impl_mpio_write(io_impl_mpio_t * io, const char * filename);
int io_impl_mpio_read(io_impl_mpio_t * io, const char * filename);
int io_impl_mpio_write_begin(io_impl_mpio_t * io, const char * filename);
int io_impl_mpio_write_end(io_impl_mpio_t * io);

#endif
