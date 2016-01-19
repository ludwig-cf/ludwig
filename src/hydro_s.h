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

#include "io_harness.h"
#include "hydro.h"

/* Data storage */

struct hydro_s {
  int nsite;
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

#ifndef OLD_SHIT

/* Remove nsite in struct when finished */

#include "leesedwards.h"

#define addr_hydro(index, ia) addr_rank1(le_nsites(), 3, index, ia)
#define vaddr_hydro(index, ia, iv) vaddr_rank1(le_nsites(), 3, index, ia, iv)

#else

#ifdef LB_DATA_SOA
#define HYADR ADDR_VECSITE_R
#else
#define HYADR ADDR_VECSITE
#endif
#endif

#endif
