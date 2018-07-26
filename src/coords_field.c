/*****************************************************************************
 *
 *  coords_field.c
 *
 *  Additional routines for halo swaps.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2018 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "coords_s.h"
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

int coords_field_init_mpi_indexed(cs_t * cs, int nhcomm, int nf,
				  MPI_Datatype mpidata,
				  MPI_Datatype halo[3]) {

  int ic, jc, n;
  int nhalo;
  int nlocal[3];
  int nh[3];             /* Length of full system in memory nlocal + 2*nhalo */
  int nstripx, nstripy;  /* Length of strips nlocal + 2*nhcomm */
  int ncount;            /* Count for the indexed type */
  int * blocklen;        /* Array of block lengths */
  int * displace;        /* Array of displacements */

  assert(cs);

  cs_nhalo(cs, &nhalo);
  cs_nlocal(cs, nlocal);

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
  assert(blocklen);
  assert(displace);
  if (blocklen == NULL) pe_fatal(cs->pe, "calloc(blocklen) failed\n");
  if (displace == NULL) pe_fatal(cs->pe, "calloc(displace) failed\n");

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
  assert(blocklen);
  assert(displace);
  if (blocklen == NULL) pe_fatal(cs->pe, "calloc(blocklen) failed\n");
  if (displace == NULL) pe_fatal(cs->pe, "calloc(displace) failed\n");

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
  assert(blocklen);
  assert(displace);
  if (blocklen == NULL) pe_fatal(cs->pe, "calloc(blocklen) failed\n");
  if (displace == NULL) pe_fatal(cs->pe, "calloc(displace) failed\n");

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
 *  coords_field_halo_rank1
 *
 *****************************************************************************/

int coords_field_halo_rank1(cs_t * cs, int nall, int nhcomm, int na,
			    void * mbuf,
			    MPI_Datatype mpidata) {
  int sz;
  int ic, jc, kc;
  int ia, index;
  int nh;
  int ireal, ihalo;
  int icount, nsend;
  int pforw, pback;
  int nlocal[3];

  unsigned char * buf;
  unsigned char * sendforw;
  unsigned char * sendback;
  unsigned char * recvforw;
  unsigned char * recvback;

  MPI_Comm comm;
  MPI_Request req[4];
  MPI_Status status[2];

  const int tagf = 2015;
  const int tagb = 2016;

  assert(cs);
  assert(mbuf);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);

  buf = (unsigned char *) mbuf;

  comm = cs->commcart;
  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);

  cs_nlocal(cs, nlocal);

  /* X-direction */

  nsend = nhcomm*na*nlocal[Y]*nlocal[Z];
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(cs->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(cs->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(cs->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(cs->pe, "malloc(recvback) failed\n");

  /* Load send buffers */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  /* Backward going... */
	  index = cs_index(cs, 1 + nh, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  /* ...and forward going. */
	  index = cs_index(cs, nlocal[X] - nh, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cs->param->mpi_cartsz[X] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs->mpi_cart_neighbours[CS_FORW][X];
    pback = cs->mpi_cart_neighbours[CS_BACK][X];
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
	  index = cs_index(cs, nlocal[X] + 1 + nh, jc, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  index = cs_index(cs, 0 - nh, jc, kc);
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
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(cs->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(cs->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(cs->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(cs->pe, "malloc(recvback) failed\n");

  /* Load buffers */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  index = cs_index(cs, ic, 1 + nh, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  index = cs_index(cs, ic, nlocal[Y] - nh, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cs->param->mpi_cartsz[Y] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs->mpi_cart_neighbours[CS_FORW][Y];
    pback = cs->mpi_cart_neighbours[CS_BACK][Y];
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
	  index = cs_index(cs, ic, 0 - nh, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  index = cs_index(cs, ic, nlocal[Y] + 1 + nh, kc);
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
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(cs->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(cs->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(cs->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(cs->pe, "malloc(recvback) failed\n");

  /* Load */
  /* Some adjustment in the load required for 2d systems (X-Y) */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
	for (ia = 0; ia < na; ia++) {
	  kc = imin(1 + nh, nlocal[Z]);
	  index = cs_index(cs, ic, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  kc = imax(nlocal[Z] - nh, 1);
	  index = cs_index(cs, ic, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cs->param->mpi_cartsz[Z] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs->mpi_cart_neighbours[CS_FORW][Z];
    pback = cs->mpi_cart_neighbours[CS_BACK][Z];
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
	  index = cs_index(cs, ic, jc, 0 - nh);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  index = cs_index(cs, ic, jc, nlocal[Z] + 1 + nh);
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

  return 0;
}
