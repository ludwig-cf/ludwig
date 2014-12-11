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
/* A preprocessor macro is provided to switch between two options
 * for the arrangement of grid-based field objects in memory:
 *
 * The following macros allow the objects to be addressed in
 * terms of:
 *
 *  lattice spatial index = coords_index(ic, jc, kc) 0 ... nsite
 *  field index ifield                               0 ... nfield
 */


/* array of structures */
#define ADDR_HYDRO(nsite, nfield, index, ifield)	\
  ((nfield)*(index) + (ifield))


/* structure of arrays */
#define ADDR_HYDRO_R(nsite, nfield, index, ifield)	\
  ((nsite)*(ifield) + (index))

#ifdef LB_DATA_SOA
#define HYADR ADDR_HYDRO_R
#else
#define HYADR ADDR_HYDRO
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
};

#endif
