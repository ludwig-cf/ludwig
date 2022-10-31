/*****************************************************************************
 *
 *  io_cart_sub.c
 *
 *  A 3-dimensional Cartesian partitioning of the existing global
 *  Cartesian communictor into blocks for the purpose of i/o to
 *  file. One file for each Cartesian sub-block.
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

#include "coords_s.h"
#include "io_cart_sub.h"

/*****************************************************************************
 *
 *  io_cart_sub_create
 *
 *  The requested iogrid[3] must exactly devide the existing Cartesian
 *  communicator.
 *
 *  Returns 0 on success with a newly created communicator.
 *
 *****************************************************************************/

int io_cart_sub_create(cs_t * cs, int iogrid[3], io_cart_sub_t * iosub) {

  assert(cs);
  assert(iosub);

  *iosub = (io_cart_sub_t) {0};

  /* Check we can make a decomposition ... */

  for (int i = 0; i < 3; i++) {
    if (cs->param->mpi_cartsz[i] % iogrid[i] != 0) goto err;
  }

  cs_cart_comm(cs, &iosub->parent);

  /* Some integer arithmetic to form blocks */

  for (int i = 0; i < 3; i++) {
    int isz = cs->param->mpi_cartsz[i];
    int icoord = cs->param->mpi_cartcoords[i];
    int ioffset = icoord / (isz/iogrid[i]);

    iosub->size[i]   = iogrid[i];
    iosub->coords[i] = iogrid[i]*icoord/isz;

    /* Offset and size, local size allowing for non-uniform decomposition */

    iosub->ntotal[i] = cs->param->ntotal[i];
    iosub->nlocal[i] = 0;
    iosub->offset[i] = cs->listnoffset[i][ioffset];

    for (int j = ioffset; j < ioffset + (isz/iogrid[i]); j++) {
      iosub->nlocal[i] += cs->listnlocal[i][j];
    }
  }

  /* We can now split the communicator from the coordinates */

  iosub->nfile = iogrid[X]*iogrid[Y]*iogrid[Z];
  iosub->index = iosub->coords[X]
               + iosub->coords[Y]*iogrid[X]
               + iosub->coords[Z]*iogrid[X]*iogrid[Y];
  {
    int rank = -1;
    MPI_Comm_rank(iosub->parent, &rank);
    MPI_Comm_split(iosub->parent, iosub->index, rank, &iosub->comm);
  }

  return 0;

 err:
  return -1;
}

/*****************************************************************************
 *
 *  io_cart_sub_free
 *
 *****************************************************************************/

int io_cart_sub_free(io_cart_sub_t * iosub) {

  assert(iosub);

  MPI_Comm_free(&iosub->comm);

  *iosub = (io_cart_sub_t) {0};
  iosub->comm = MPI_COMM_NULL;

  return 0;
}

/*****************************************************************************
 *
 *  io_cart_sub_printf
 *
 *****************************************************************************/

int io_cart_sub_printf(const io_cart_sub_t * iosub) {

  int sz   = -1;
  int rank = -1;
  char msg[BUFSIZ] = {0};

  assert(iosub);

  /* Commnuication is in the Cartesian communicator */

  MPI_Comm_size(iosub->parent, &sz);
  MPI_Comm_rank(iosub->parent, &rank);


  {
    int coords[3] = {0};  /* Cartesian communicator coordinates */
    int iorank = -1;
    MPI_Comm_rank(iosub->comm, &iorank);
    MPI_Cart_coords(iosub->parent, rank, 3, coords);
    sprintf(msg, "%4d %3d %3d %3d   %2d %4d   %3d %3d %3d  %4d %4d %4d",
	    rank, coords[X], coords[Y], coords[Z],
	    iosub->index, iorank,
	    iosub->coords[X], iosub->coords[Y], iosub->coords[Z],
	    iosub->offset[X], iosub->offset[Y], iosub->offset[Z]);
  }

  if (rank > 0) {
    MPI_Send(msg, BUFSIZ, MPI_CHAR, 0, 271022, iosub->parent);
  }
  else {
    printf("Cartesian           io_cart_sub\n");
    printf("rank   x   y   z    file rank  x   y   z    ox   oy   oz\n");
    printf("%s\n", msg);
    for (int ir = 1; ir < sz; ir++) {
      MPI_Recv(msg, BUFSIZ, MPI_CHAR, ir, 271022, iosub->parent,
	       MPI_STATUS_IGNORE);
      printf("%s\n", msg);
    }
  }

  return 0;
}
