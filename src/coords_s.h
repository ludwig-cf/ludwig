
#ifndef LUDWIG_COORDS_S_H
#define LUDWIG_COORDS_S_H

#include "coords.h"

typedef struct coords_param_s cs_param_t;

struct coords_param_s {
  int nhalo;                       /* Width of halo region */
  int nsites;                      /* Total sites (incl. halo) */
  int ntotal[3];                   /* System (physical) size */
  int nlocal[3];                   /* Local system size */
  int noffset[3];                  /* Local system offset */
  int str[3];                      /* Memory strides */
  int periodic[3];                 /* Periodic boundary (non-periodic = 0) */

  int mpi_cartsz[3];               /* Cartesian size */
  int mpi_cartcoords[3];           /* Cartesian coordinates lookup */

  double lmin[3];                  /* System L_min */
};

struct coords_s {
  pe_t * pe;                       /* Retain a reference to pe */
  int nref;                        /* Reference count */

  cs_param_t * param;              /* Constants */

  /* Host data */
  int mpi_cartrank;                /* MPI Cartesian rank */
  int reorder;                     /* MPI reorder flag */
  int mpi_cart_neighbours[2][3];   /* Ranks of Cartesian neighbours lookup */
  int * listnlocal[3];             /* Rectilinear decomposition */
  int * listnoffset[3];            /* Rectilinear offsets */

  MPI_Comm commcart;               /* Cartesian communicator */
  MPI_Comm commperiodic;           /* Cartesian periodic communicator */

  cs_t * target;                   /* Host pointer to target memory */
};

#endif
