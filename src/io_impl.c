/*****************************************************************************
 *
 *  io_impl.c
 *
 *  A factory method to choose a real implementation.
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

#include "io_impl.h"
#include "io_impl_mpio.h"

int io_impl_create(const io_metadata_t * metadata, io_impl_t ** io) {

  *io = NULL;

  switch (metadata->options.mode) {
  case IO_MODE_MPIIO:
    {
      io_impl_mpio_t * mpio = NULL;
      io_impl_mpio_create(metadata, &mpio);
      *io = (io_impl_t *) mpio;
    }
    break;
  default:
    ;
  }

  return 0;
}
