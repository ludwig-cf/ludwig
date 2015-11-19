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

/* Data storage */

#ifdef LB_DATA_SOA
#define HYADR ADDR_VECSITE_R
#else
#define HYADR ADDR_VECSITE
#endif


struct hydro_s {
  int nf;                  /* Extent of fields = 3 for vectors */
  int nhcomm;              /* Width of halo region for u field */
  double * u;              /* Velocity field (on host)*/
  double * f;              /* Body force field (on host) */
  double * t_u;              /* Velocity field (on target) */
  double * t_f;              /* Body force field (on target) */
  MPI_Datatype uhalo[3];   /* Halo exchange datatypes for velocity */
  io_info_t * info;        /* I/O handler. */

  hydro_t * tcopy;              /* copy of this structure on target */ 

};

#endif
