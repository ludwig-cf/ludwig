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
  MPI_Datatype etype;   /* Data type for element */
  size_t       esize;   /* Number bytes per element */
};

#endif
