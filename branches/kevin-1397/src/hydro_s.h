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
#include "memory.h"

#include "leesedwards.h"
#include "io_harness.h"
#include "hydro.h"

/* Data storage: Always a 3-vector NHDIM */

#define NHDIM 3

struct hydro_s {
  int nsite;               /* Allocated sites (local) */
  int nhcomm;              /* Width of halo region for u field */
  double * u;              /* Velocity field (on host)*/
  double * f;              /* Body force field (on host) */
  MPI_Datatype uhalo[3];   /* Halo exchange datatypes for velocity */
  io_info_t * info;        /* I/O handler. */

  hydro_t * tcopy;         /* structure on target */ 
};

#define addr_hydro(index, ia) addr_rank1(le_nsites(), NHDIM, index, ia)
#define vaddr_hydro(index, ia, iv) vaddr_rank1(le_nsites(), NHDIM, index, ia, iv)

#endif
