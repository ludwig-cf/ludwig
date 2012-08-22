/*****************************************************************************
 *
 *  coords_field.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "coords_field.h"

/*****************************************************************************
 *
 *  coords_field_init_mpi_indexed
 *
 *  Initialise the MPI_Type_indexed structures associated with a
 *  field with nf components in the current coordinate system.
 *  These datatypes allow a halo swap used in conjunction with
 *  coords_field_halo() below.
 *
 *  The definition of the structures may be understood by comparing
 *  the extents and strides with the loops in the 'serial' halo swap
 *  in coords_field_halo(). The datatype of the message is MPI_DOUBLE.
 *
 *  Assumes the field storage is f[index][nf] where index is the
 *  usual spatial index returned by coords_index(), and nf is the
 *  number of fields.
 *
 *  The indexed structures are used so that the recieves in the
 *  different coordinate directions do not overlap anywhere. The
 *  receives may then all be posted independently.
 *
 *  The caller must arrange for the MPI_Datatypes to be released
 *  at the end of use.
 *
 *****************************************************************************/

int coords_field_init_mpi_indexed(int nf, MPI_Datatype halo[3]) {

  int nhalo;
  int nlocal[3], nh[3];
  int ic, jc, n;

  int ncount;        /* Count for the indexed type */
  int * blocklen;    /* Array of block lengths */
  int * displace;    /* Array of displacements */

  assert(halo);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nh[X] = nlocal[X] + 2*nhalo;
  nh[Y] = nlocal[Y] + 2*nhalo;
  nh[Z] = nlocal[Z] + 2*nhalo;

  /* X direction */
  /* We require nlocal[Y] contiguous strips of length nlocal[Z],
   * repeated for each halo layer. The strides start at zero, and
   * increment nh[Z] for each strip. */

  ncount = nhalo*nlocal[Y];

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nf*nlocal[Z];
  }

  for (n = 0; n < nhalo; n++) {
    for (jc = 0; jc < nlocal[Y]; jc++) {
      displace[n*nlocal[Y] + jc] = nf*(n*nh[Y]*nh[Z] + jc*nh[Z]);
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &halo[X]);
  MPI_Type_commit(&halo[X]);

  free(displace);
  free(blocklen);

  /* Y direction */
  /* We can use nh[X] contiguous strips of nlocal[Z], repeated for
   * each halo region. The strides start at zero, and increment by
   * nh[Y]*nh[Z] for each strip. */

  ncount = nhalo*nh[X];

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nf*nlocal[Z];
  }

  for (n = 0; n < nhalo; n++) {
    for (ic = 0; ic < nh[X] ; ic++) {
      displace[n*nh[X] + ic] = nf*(n*nh[Z] + ic*nh[Y]*nh[Z]);
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &halo[Y]);
  MPI_Type_commit(&halo[Y]);

  free(displace);
  free(blocklen);

  /* Z direction */
  /* Here, we need nh[X]*nh[Y] small contiguous strips of nhalo, with
   * a stride between each of nh[Z]. */

  ncount = nh[X]*nh[Y];

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nf*nhalo;
  }

  for (ic = 0; ic < nh[X]; ic++) {
    for (jc = 0; jc < nh[Y]; jc++) {
      displace[ic*nh[Y] + jc] = nf*(ic*nh[Y]*nh[Z]+ jc*nh[Z]);
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &halo[Z]);
  MPI_Type_commit(&halo[Z]);

  free(displace);
  free(blocklen);

  return 0;
}

/*****************************************************************************
 *
 *  coord_field_halo
 *
 *  A general halo swap where:
 *    nf is the number of 3-D fields
 *    f is the base address of the field
 *    halo are thre X, Y, Z MPI indexed datatypes for the swap
 *
 *****************************************************************************/

int coords_field_halo(int nf, double * f, MPI_Datatype halo[3]) {

  int n, nh;
  int nhalo;
  int nlocal[3];
  int ic, jc, kc;
  int pback, pforw;          /* MPI ranks of 'left' and 'right' neighbours */
  int ihalo, ireal;          /* Indices of halo and 'real' lattice regions */
  MPI_Comm comm;             /* Cartesian communicator */
  MPI_Request req_recv[6];
  MPI_Request req_send[6];
  MPI_Status  status[6];

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;

  assert(f);
  assert(halo);

  comm = cart_comm();
  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  for (n = 0; n < 6; n++) {
    req_send[n] = MPI_REQUEST_NULL;
    req_recv[n] = MPI_REQUEST_NULL;
  }

  /* Post all recieves */

  if (cart_size(X) > 1) {
    pback = cart_neighb(BACKWARD, X);
    pforw = cart_neighb(FORWARD, X);
    ihalo = nf*coords_index(nlocal[X] + 1, 1, 1);
    MPI_Irecv(f + ihalo,  1, halo[X], pforw, btagx, comm, req_recv);
    ihalo = nf*coords_index(1 - nhalo, 1, 1);
    MPI_Irecv(f + ihalo,  1, halo[X], pback, ftagx, comm, req_recv + 1);
  }

  if (cart_size(Y) > 1) {
    pback = cart_neighb(BACKWARD, Y);
    pforw = cart_neighb(FORWARD, Y);
    ihalo = nf*coords_index(1 - nhalo, nlocal[Y] + 1, 1);
    MPI_Irecv(f + ihalo,  1, halo[Y], pforw, btagy, comm, req_recv + 2);
    ihalo = nf*coords_index(1 - nhalo, 1 - nhalo, 1);
    MPI_Irecv(f + ihalo,  1, halo[Y], pback, ftagy, comm, req_recv + 3);
  }

  if (cart_size(Z) > 1) {
    pback = cart_neighb(BACKWARD, Z);
    pforw = cart_neighb(FORWARD, Z);
    ihalo = nf*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z] + 1);
    MPI_Irecv(f + ihalo,  1, halo[Z], pforw, btagz, comm, req_recv + 4);
    ihalo = nf*coords_index(1 - nhalo, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(f + ihalo,  1, halo[Z], pback, ftagz, comm, req_recv + 5);
  }

  /* Now the sends */

  if (cart_size(X) > 1) {
    pback = cart_neighb(BACKWARD, X);
    pforw = cart_neighb(FORWARD, X);
    ireal = nf*coords_index(1, 1, 1);
    MPI_Issend(f + ireal, 1, halo[X], pback, btagx, comm, req_send);
    ireal = nf*coords_index(nlocal[X] - nhalo + 1, 1, 1);
    MPI_Issend(f + ireal, 1, halo[X], pforw, ftagx, comm, req_send + 1);
  }
  else {
    for (nh = 0; nh < nhalo; nh++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1 ; kc <= nlocal[Z]; kc++) {
          for (n = 0; n < nf; n++) {
            ihalo = n + nf*coords_index(0 - nh, jc, kc);
            ireal = n + nf*coords_index(nlocal[X] - nh, jc, kc);
            f[ihalo] = f[ireal];
            ihalo = n + nf*coords_index(nlocal[X] + 1 + nh, jc,kc);
            ireal = n + nf*coords_index(1 + nh, jc, kc);
            f[ihalo] = f[ireal];
          }
        }
      }
    }
  }

  /* X recvs to be complete before Y sends */
  MPI_Waitall(2, req_recv, status);

  if (cart_size(Y) > 1) {
    pback = cart_neighb(BACKWARD, Y);
    pforw = cart_neighb(FORWARD, Y);
    ireal = nf*coords_index(1 - nhalo, 1, 1);
    MPI_Issend(f + ireal, 1, halo[Y], pback, btagy, comm, req_send + 2);
    ireal = nf*coords_index(1 - nhalo, nlocal[Y] - nhalo + 1, 1);
    MPI_Issend(f + ireal, 1, halo[Y], pforw, ftagy, comm, req_send + 3);
  }
  else {
    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1-nhalo; ic <= nlocal[X] + nhalo; ic++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {
          for (n = 0; n < nf; n++) {
            ihalo = n + nf*coords_index(ic, 0 - nh, kc);
            ireal = n + nf*coords_index(ic, nlocal[Y] - nh, kc);
            f[ihalo] = f[ireal];
            ihalo = n + nf*coords_index(ic, nlocal[Y] + 1 + nh, kc);
            ireal = n + nf*coords_index(ic, 1 + nh, kc);
            f[ihalo] = f[ireal];
          }
        }
      }
    }
  }

  /* Y recvs to be complete before Z sends */
  MPI_Waitall(2, req_recv + 2, status);

  if (cart_size(Z) > 1) {
    pback = cart_neighb(BACKWARD, Z);
    pforw = cart_neighb(FORWARD, Z);
    ireal = nf*coords_index(1 - nhalo, 1 - nhalo, 1);
    MPI_Issend(f + ireal, 1, halo[Z], pback, btagz, comm, req_send + 4);
    ireal = nf*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z] - nhalo + 1);
    MPI_Issend(f + ireal, 1, halo[Z], pforw, ftagz, comm, req_send + 5);
  }
  else {
    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
        for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
          for (n = 0; n < nf; n++) {
            ihalo = n + nf*coords_index(ic, jc, 0 - nh);
            ireal = n + nf*coords_index(ic, jc, nlocal[Z] - nh);
            f[ihalo] = f[ireal];
            ihalo = n + nf*coords_index(ic, jc, nlocal[Z] + 1 + nh);
            ireal = n + nf*coords_index(ic, jc, 1 + nh);
            f[ihalo] = f[ireal];
          }
        }
      }
    }
  }

  /* Finish */
  MPI_Waitall(2, req_recv + 4, status);
  MPI_Waitall(6, req_send, status);

  return 0;
}
