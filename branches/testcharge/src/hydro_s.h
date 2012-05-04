/*****************************************************************************
 *
 *  hydro_s.h
 *
 *  Structure.
 *
 *****************************************************************************/

#ifndef HYDRO_S_H
#define HYDRO_S_H

#include <mpi.h>

#include "io_harness.h"
#include "hydro.h"

struct hydro_s {
  int nf;                  /* Extent of fields = 3 for vectors */
  int nhcomm;              /* Width of halo region for u field */
  double * u;              /* Velocity field */
  double * f;              /* Body force field */
  MPI_Datatype uhalo[3];   /* Halo exchange datatypes for velocity */
  struct io_info_t * info;        /* I/O handler. */
};

#endif
