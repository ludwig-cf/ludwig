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
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "memory.h"
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

/*****************************************************************************
 *
 *  coords_field_init_mpi_indexed
 *
 *  Here we define MPI_Type_indexed structures to take care of halo
 *  swaps on the lattice. These structures may be understood by
 *  comparing the extents and strides with the loops in the 'serial'
 *  halo swap in field_halo().
 *
 *  This is the most general case where the coords object has a given
 *  halo extent nhalo, but we only require a swap of width nhcomm,
 *  where 0 < nhcomm <= nhalo. We use nh[3] for the extent of the full
 *  lattice locally in memory.
 *
 *  The indexed structures are used so that the receives in the different
 *  coordinate direction do not overlap anywhere. The receives may then
 *  all be posted independently.
 *
 *  We assume the field storage is contiguous per lattice site, i.e.,
 *  f[index][nf], where nf is the number of fields. E.g., a vector
 *  will be expected to have nf = 3.
 *
 *  Three newly commited MPI_Datatypes are returned. These datatypes
 *  must be used in conjunction with the correct starting indices as
 *  seen in the field_halo() routine.
 *
 *****************************************************************************/

int coords_field_init_mpi_indexed(int nhcomm, int nf, MPI_Datatype mpidata,
				  MPI_Datatype halo[3]) {

  int ic, jc, n;
  int nhalo;
  int nlocal[3];
  int nh[3];             /* Length of full system in memory nlocal + 2*nhalo */
  int nstripx, nstripy;  /* Length of strips nlocal + 2*nhcomm */
  int ncount;            /* Count for the indexed type */
  int * blocklen;        /* Array of block lengths */
  int * displace;        /* Array of displacements */


  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  nh[X] = nlocal[X] + 2*nhalo;
  nh[Y] = nlocal[Y] + 2*nhalo;
  nh[Z] = nlocal[Z] + 2*nhalo;

  /* X direction */
  /* We may use nlocal[Y] contiguous strips of nlocal[Z]  (each site of which
   * will have nf elements). This is repeated for each communicated halo
   * layer. The strides start at zero, and increment by nf*nh[Z] for each
   * strip */

  nstripy = nlocal[Y];
  ncount = nhcomm*nstripy;

  blocklen = (int*) calloc(ncount, sizeof(int));
  displace = (int*) calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nf*nlocal[Z];
  }

  for (n = 0; n < nhcomm; n++) {
    for (jc = 0; jc < nstripy; jc++) {
      displace[n*nstripy + jc] = nf*(n*nh[Y]*nh[Z] + jc*nh[Z]);
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, mpidata, &halo[X]);
  MPI_Type_commit(&halo[X]);

  free(displace);
  free(blocklen);

  /* Y direction */
  /* We can use (nlocal[X] + 2*nhcomm) contiguous strips of nlocal[Z],
   * repeated for each halo layer required. The strides start at zero,
   * and increment by the full nf*nh[Y]*nh[Z] for each strip. */

  nstripx = nlocal[X] + 2*nhcomm;
  ncount = nhcomm*nstripx;

  blocklen = (int*) calloc(ncount, sizeof(int));
  displace = (int*) calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nf*nlocal[Z];
  }

  for (n = 0; n < nhcomm; n++) {
    for (ic = 0; ic < nstripx; ic++) {
      displace[n*nstripx + ic] = nf*(n*nh[Z] + ic*nh[Y]*nh[Z]);
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, mpidata, &halo[Y]);
  MPI_Type_commit(&halo[Y]);

  free(displace);
  free(blocklen);

  /* Z direction */
  /* Here, we need (nlocal[X] + 2*nhcomm)*(nlocal[Y] + 2*nhcomm) short
   * contiguous strips each of nf*nhcomm. The strides start at zero,
   * and are just one full strip nf*nh[Z]. */

  nstripx = (nlocal[X] + 2*nhcomm);
  nstripy = (nlocal[Y] + 2*nhcomm);
  ncount = nstripx*nstripy;

  blocklen = (int*) calloc(ncount, sizeof(int));
  displace = (int*) calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nf*nhcomm;
  }

  for (ic = 0; ic < nstripx; ic++) {
    for (jc = 0; jc < nstripy; jc++) {
      displace[ic*nstripy + jc] = nf*(ic*nh[Y]*nh[Z] + jc*nh[Z]);
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, mpidata, &halo[Z]);
  MPI_Type_commit(&halo[Z]);

  free(displace);
  free(blocklen);

  return 0;
}

/*****************************************************************************
 *
 *  coords_field_halo_d
 *
 *  General halo swap for lattice field quantities (double).
 *
 *  We expect a coords object appropriate for local system nlocal with
 *  full halo extent nhalo. The field requires nhcomm halo layers to
 *  be communicated with 0 < nhcomm <= nhalo. The field haas nf elements
 *  per lattice site and the ordering is field[index][nf] with index
 *  the spatial index returned by coords_index().
 *
 *  The three coordinate directions are transfered in turn with
 *  coorsponding halo data types halo[]. The field has starting
 *  address f.
 *
 *****************************************************************************/

int coords_field_halo_d(int nhcomm, int nf, double * f, MPI_Datatype halo[3]) {

  int nlocal[3];
  int ic, jc, kc, ihalo, ireal;
  int pforw, pback;
  int n, nh;

  MPI_Request req_send[6];
  MPI_Request req_recv[6];
  MPI_Status  status[6];
  MPI_Comm    comm;

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;

  assert(f);
  assert(halo);

  coords_nlocal(nlocal);
  comm = cart_comm();

  for (n = 0; n < 6; n++) {
    req_send[n] = MPI_REQUEST_NULL;
    req_recv[n] = MPI_REQUEST_NULL;
  }

  /* Post all receives */

  if (cart_size(X) > 1) {
    pback = cart_neighb(BACKWARD, X);
    pforw = cart_neighb(FORWARD, X);
    ihalo = nf*coords_index(nlocal[X] + 1, 1, 1);
    MPI_Irecv(f + ihalo, 1, halo[X], pforw, btagx, comm, req_recv);
    ihalo = nf*coords_index(1 - nhcomm, 1, 1);
    MPI_Irecv(f + ihalo, 1, halo[X], pback, ftagx, comm, req_recv + 1);
  }

  if (cart_size(Y) > 1) {
    pback = cart_neighb(BACKWARD, Y);
    pforw = cart_neighb(FORWARD, Y);
    ihalo = nf*coords_index(1 - nhcomm, nlocal[Y] + 1, 1);
    MPI_Irecv(f + ihalo, 1, halo[Y], pforw, btagy, comm, req_recv + 2);
    ihalo = nf*coords_index(1 - nhcomm, 1 - nhcomm, 1);
    MPI_Irecv(f + ihalo, 1, halo[Y], pback, ftagy, comm, req_recv + 3);
  }

  if (cart_size(Z) > 1) {
    pback = cart_neighb(BACKWARD, Z);
    pforw = cart_neighb(FORWARD, Z);
    ihalo = nf*coords_index(1 - nhcomm, 1 - nhcomm, nlocal[Z] + 1);
    MPI_Irecv(f + ihalo, 1, halo[Z], pforw, btagz, comm, req_recv + 4);
    ihalo = nf*coords_index(1 - nhcomm, 1 - nhcomm, 1 - nhcomm);
    MPI_Irecv(f + ihalo, 1, halo[Z], pback, ftagz, comm, req_recv + 5);
  }


  /* X sends */

  if (cart_size(X) > 1) {
    pback = cart_neighb(BACKWARD, X);
    pforw = cart_neighb(FORWARD, X);
    ireal = nf*coords_index(1, 1, 1);
    MPI_Issend(f + ireal, 1, halo[X], pback, btagx, comm, req_send);
    ireal = nf*coords_index(nlocal[X] - nhcomm + 1, 1, 1);
    MPI_Issend(f + ireal, 1, halo[X], pforw, ftagx, comm, req_send + 1);
  }
  else {
    for (nh = 0; nh < nhcomm; nh++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    ihalo = n + nf*coords_index(0 - nh, jc, kc);
	    ireal = n + nf*coords_index(nlocal[X] - nh, jc, kc);
	    f[ihalo] = f[ireal];
	    ihalo = n + nf*coords_index(nlocal[X] + 1 + nh, jc, kc);
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
    ireal = nf*coords_index(1 - nhcomm, 1, 1);
    MPI_Issend(f + ireal, 1, halo[Y], pback, btagy, comm, req_send + 2);
    ireal = nf*coords_index(1 - nhcomm, nlocal[Y] - nhcomm + 1, 1);
    MPI_Issend(f + ireal, 1, halo[Y], pforw, ftagy, comm, req_send + 3);
  }
  else {
    for (nh = 0; nh < nhcomm; nh++) {
      for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
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
    ireal = nf*coords_index(1 - nhcomm, 1 - nhcomm, 1);
    MPI_Issend(f + ireal, 1, halo[Z], pback, btagz, comm, req_send + 4);
    ireal = nf*coords_index(1 - nhcomm, 1 - nhcomm, nlocal[Z] - nhcomm + 1);
    MPI_Issend(f + ireal, 1, halo[Z], pforw, ftagz, comm, req_send + 5);
  }
  else {
    for (nh = 0; nh < nhcomm; nh++) {
      for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
	for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
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

/*****************************************************************************
 *
 *  coords_field_halo
 *
 *  Here is a general case allowing MPI_CHAR or MPI_DOUBLE
 *
 *****************************************************************************/

int coords_field_halo(int nhcomm, int nf, void * buf, MPI_Datatype mpidata,
		      MPI_Datatype halo[3]) {

  int nlocal[3];
  int ic, jc, kc, ihalo, ireal;
  int pforw, pback;
  int n, nh;
  size_t sz;
  unsigned char * mbuf = (unsigned char*) buf;

  MPI_Request req_send[6];
  MPI_Request req_recv[6];
  MPI_Status  status[6];
  MPI_Comm    comm;

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;

  assert(mbuf);
  assert(halo);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);

  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);

  coords_nlocal(nlocal);
  comm = cart_comm();

  for (n = 0; n < 6; n++) {
    req_send[n] = MPI_REQUEST_NULL;
    req_recv[n] = MPI_REQUEST_NULL;
  }

  /* Post all receives */

  if (cart_size(X) > 1) {
    pback = cart_neighb(BACKWARD, X);
    pforw = cart_neighb(FORWARD, X);
    ihalo = sz*nf*coords_index(nlocal[X] + 1, 1, 1);
    MPI_Irecv(mbuf + ihalo, 1, halo[X], pforw, btagx, comm, req_recv);
    ihalo = sz*nf*coords_index(1 - nhcomm, 1, 1);
    MPI_Irecv(mbuf + ihalo, 1, halo[X], pback, ftagx, comm, req_recv + 1);
  }

  if (cart_size(Y) > 1) {
    pback = cart_neighb(BACKWARD, Y);
    pforw = cart_neighb(FORWARD, Y);
    ihalo = sz*nf*coords_index(1 - nhcomm, nlocal[Y] + 1, 1);
    MPI_Irecv(mbuf + ihalo, 1, halo[Y], pforw, btagy, comm, req_recv + 2);
    ihalo = sz*nf*coords_index(1 - nhcomm, 1 - nhcomm, 1);
    MPI_Irecv(mbuf + ihalo, 1, halo[Y], pback, ftagy, comm, req_recv + 3);
  }

  if (cart_size(Z) > 1) {
    pback = cart_neighb(BACKWARD, Z);
    pforw = cart_neighb(FORWARD, Z);
    ihalo = sz*nf*coords_index(1 - nhcomm, 1 - nhcomm, nlocal[Z] + 1);
    MPI_Irecv(mbuf + ihalo, 1, halo[Z], pforw, btagz, comm, req_recv + 4);
    ihalo = sz*nf*coords_index(1 - nhcomm, 1 - nhcomm, 1 - nhcomm);
    MPI_Irecv(mbuf + ihalo, 1, halo[Z], pback, ftagz, comm, req_recv + 5);
  }


  /* X sends */

  if (cart_size(X) > 1) {
    pback = cart_neighb(BACKWARD, X);
    pforw = cart_neighb(FORWARD, X);
    ireal = sz*nf*coords_index(1, 1, 1);
    MPI_Issend(mbuf + ireal, 1, halo[X], pback, btagx, comm, req_send);
    ireal = sz*nf*coords_index(nlocal[X] - nhcomm + 1, 1, 1);
    MPI_Issend(mbuf + ireal, 1, halo[X], pforw, ftagx, comm, req_send + 1);
  }
  else {
    for (nh = 0; nh < nhcomm; nh++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    ihalo = n + nf*coords_index(0 - nh, jc, kc);
	    ireal = n + nf*coords_index(nlocal[X] - nh, jc, kc);
	    memcpy(mbuf + sz*ihalo, mbuf + sz*ireal, sz);
	    ihalo = n + nf*coords_index(nlocal[X] + 1 + nh, jc, kc);
	    ireal = n + nf*coords_index(1 + nh, jc, kc);
	    memcpy(mbuf + sz*ihalo, mbuf + sz*ireal, sz);
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
    ireal = sz*nf*coords_index(1 - nhcomm, 1, 1);
    MPI_Issend(mbuf + ireal, 1, halo[Y], pback, btagy, comm, req_send + 2);
    ireal = sz*nf*coords_index(1 - nhcomm, nlocal[Y] - nhcomm + 1, 1);
    MPI_Issend(mbuf + ireal, 1, halo[Y], pforw, ftagy, comm, req_send + 3);
  }
  else {
    for (nh = 0; nh < nhcomm; nh++) {
      for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    ihalo = n + nf*coords_index(ic, 0 - nh, kc);
	    ireal = n + nf*coords_index(ic, nlocal[Y] - nh, kc);
	    memcpy(mbuf + sz*ihalo, mbuf + sz*ireal, sz);
	    ihalo = n + nf*coords_index(ic, nlocal[Y] + 1 + nh, kc);
	    ireal = n + nf*coords_index(ic, 1 + nh, kc);
	    memcpy(mbuf + sz*ihalo, mbuf + sz*ireal, sz);
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
    ireal = sz*nf*coords_index(1 - nhcomm, 1 - nhcomm, 1);
    MPI_Issend(mbuf + ireal, 1, halo[Z], pback, btagz, comm, req_send + 4);
    ireal = sz*nf*coords_index(1 - nhcomm, 1 - nhcomm, nlocal[Z] - nhcomm + 1);
    MPI_Issend(mbuf + ireal, 1, halo[Z], pforw, ftagz, comm, req_send + 5);
  }
  else {
    for (nh = 0; nh < nhcomm; nh++) {
      for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
	for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
	  for (n = 0; n < nf; n++) {
	    ihalo = n + nf*coords_index(ic, jc, 0 - nh);
	    ireal = n + nf*coords_index(ic, jc, nlocal[Z] - nh);
	    memcpy(mbuf + sz*ihalo, mbuf + sz*ireal, sz);
	    ihalo = n + nf*coords_index(ic, jc, nlocal[Z] + 1 + nh);
	    ireal = n + nf*coords_index(ic, jc, 1 + nh);
	    memcpy(mbuf + sz*ihalo, mbuf + sz*ireal, sz);
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

/*****************************************************************************
 *
 *  coords_field_halo_rank1
 *
 *****************************************************************************/

int coords_field_halo_rank1(int nall, int nhcomm, int na, void * buf,
			    MPI_Datatype mpidata) {
  int sz;
  int ic, jc, kc;
  int ia, index;
  int nh;
  int ireal, ihalo;
  int icount, nsend;
  int pforw, pback;
  int nlocal[3];

  void * sendforw;
  void * sendback;
  void * recvforw;
  void * recvback;

  MPI_Comm comm;
  MPI_Request req[4];
  MPI_Status status[2];

  const int tagf = 2015;
  const int tagb = 2016;

  assert(buf);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);

  comm = cart_comm();
  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);

  coords_nlocal(nlocal);

  /* X-direction */

  nsend = nhcomm*na*nlocal[Y]*nlocal[Z];
  sendforw = (void *) malloc(nsend*sz);
  sendback = (void *) malloc(nsend*sz);
  recvforw = (void *) malloc(nsend*sz);
  recvback = (void *) malloc(nsend*sz);
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvback == NULL) fatal("malloc(recvback) failed\n");

  /* Load send buffers */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  /* Backward going... */
	  index = coords_index(1 + nh, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  /* ...and forward going. */
	  index = coords_index(nlocal[X] - nh, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cart_size(X) == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
  }
  else {
    pforw = cart_neighb(FORWARD, X);
    pback = cart_neighb(BACKWARD, X);
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait for receives */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  index = coords_index(nlocal[X] + 1 + nh, jc, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  index = coords_index(0 - nh, jc, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  /* Y direction */

  nsend = nhcomm*na*(nlocal[X] + 2*nhcomm)*nlocal[Z];
  sendforw = (void *) malloc(nsend*sz);
  sendback = (void *) malloc(nsend*sz);
  recvforw = (void *) malloc(nsend*sz);
  recvback = (void *) malloc(nsend*sz);
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvback == NULL) fatal("malloc(recvback) failed\n");

  /* Load buffers */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  index = coords_index(ic, 1 + nh, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  index = coords_index(ic, nlocal[Y] - nh, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cart_size(Y) == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
  }
  else {
    pforw = cart_neighb(FORWARD, Y);
    pback = cart_neighb(BACKWARD, Y);
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait for receives */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  index = coords_index(ic, 0 - nh, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  index = coords_index(ic, nlocal[Y] + 1 + nh, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  /* Z direction */

  nsend = nhcomm*na*(nlocal[X] + 2*nhcomm)*(nlocal[Y] + 2*nhcomm);
  sendforw = (void *) malloc(nsend*sz);
  sendback = (void *) malloc(nsend*sz);
  recvforw = (void *) malloc(nsend*sz);
  recvback = (void *) malloc(nsend*sz);
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvback == NULL) fatal("malloc(recvback) failed\n");

  /* Load */
  /* Some adjust required for 2d systems ? */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
	for (ia = 0; ia < na; ia++) {
	  kc = imin(1 + nh, nlocal[Z]);
	  index = coords_index(ic, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  kc = imax(nlocal[Z] - nh, 1);
	  index = coords_index(ic, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cart_size(Z) == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
  }
  else {
    pforw = cart_neighb(FORWARD, Y);
    pback = cart_neighb(BACKWARD, Y);
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait before unloading */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
	for (ia = 0; ia < na; ia++) {
	  double val1, val2;
	  index = coords_index(ic, jc, 0 - nh);
	  ihalo = addr_rank1(nall, na, index, ia);
	  val1 = *((double*)(recvback +icount*sz));
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  index = coords_index(ic, jc, nlocal[Z] + 1 + nh);
	  ihalo = addr_rank1(nall, na, index, ia);
	  val2 = *((double *)(recvforw+icount*sz));
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  /*	  if (ia == 0) printf("H %2d %2d %2d %2d %10.2e %10.2e\n", nh, ic, jc, ia, val1, val2);*/
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  return 0;
}
