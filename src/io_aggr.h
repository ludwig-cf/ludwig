/*****************************************************************************
 *
 *  io_aggr.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_AGGR_H
#define LUDWIG_IO_AGGR_H

#include <mpi.h>

typedef struct io_aggr_s io_aggr_t;

struct io_aggr_s {
  MPI_Datatype bin_etype;   /* Data type for element (binary) */
  MPI_Datatype asc_etype;   /* Data type for element (ascii) usu. MPI_CHAR */
  size_t       bin_esize;   /* number bytes per lattice site */
  size_t       asc_esize;   /* fixed character size per line of output */
};

#endif
