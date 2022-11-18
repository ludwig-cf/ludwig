/*****************************************************************************
 *
 *  io_metadata.c
 *
 *  Lattice quantity i/o metadata. 
 *
 *  Aggregate information on i/o, that is the run time options, the data
 *  element type, and the lattice structure, and provide a communicator.
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
#include <stdlib.h>

#include "io_metadata.h"

/*****************************************************************************
 *
 *  io_metadata_create
 *
 *  Generate once only per run as the cost of the MPI_Comm_split()
 *  should not be repeated.
 *
 *****************************************************************************/

int io_metadata_create(cs_t * cs,
		       const io_options_t * options,
		       const io_element_t * element,
		       io_metadata_t * meta) {
  assert(cs);
  assert(options);
  assert(element);
  assert(meta);

  *meta = (io_metadata_t) {0};

  meta->cs = cs;
  cs_cart_comm(cs, &meta->parent);

  {
    /* Store the limits as a convenience */
    int nlocal[3] = {0};
    cs_nlocal(cs, nlocal);
    meta->limits.imin = 1; meta->limits.imax = nlocal[X];
    meta->limits.jmin = 1; meta->limits.jmax = nlocal[Y];
    meta->limits.kmin = 1; meta->limits.kmax = nlocal[Z];
  }

  meta->options = *options;
  meta->element = *element;

  {
    /* Must have a decomposition... */
    int ifail = io_subfile_create(cs, options->iogrid, &meta->subfile);
    if (ifail != 0) return -1;
  }

  /* Split communicator for the file. */
  /* One could revisit assigning the rank ... */
  {
    int rank = -1;
    MPI_Comm_rank(meta->parent, &rank);
    MPI_Comm_split(meta->parent, meta->subfile.index, rank, &meta->comm);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_metadata_free
 *
 *****************************************************************************/

int io_metadata_free(io_metadata_t * meta) {

  assert(meta);

  MPI_Comm_free(&meta->comm);
  *meta = (io_metadata_t) {0};

  return 0;
}
