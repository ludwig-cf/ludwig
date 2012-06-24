/*****************************************************************************
 *
 *  hydro.c
 *
 *  Hydrodynamic quantities: velocity, body force on fluid.
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
#include <math.h>
#include <stdlib.h> 

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "io_harness.h"
#include "util.h"
#include "control.h" /* Can we move this into LE please */
#include "hydro_s.h"

int field_init_mpi_indexed(int nlocal[3], int nhalo, int nhcomm, int nf,
			   MPI_Datatype halo[3]);
int field_halo(int nlocal[3], int nhalo, int nhcomm, int nf, double * f,
	       MPI_Datatype halo[3], MPI_Comm comm);

static int hydro_lees_edwards_parallel(hydro_t * obj);
static int hydro_u_write(FILE * fp, int index, void * self);
static int hydro_u_write_ascii(FILE * fp, int index, void * self);

/*****************************************************************************
 *
 *  hydro_create
 *
 *  We typically require a halo region for the velocity which is only
 *  one lattice site in width, i.e., nhcomm = 1. This is independent
 *  of the width of the halo region specified for coords object.
 *
 *****************************************************************************/

int hydro_create(int nhcomm, hydro_t ** pobj) {

  int nsites;
  hydro_t * obj = NULL;

  int nhalo;
  int nlocal[3];

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  assert(pobj);

  obj = calloc(1, sizeof(hydro_t));
  if (obj == NULL) fatal("calloc(hydro) failed\n");

  obj->nf = 3; /* always for velocity, force */
  obj->nhcomm = nhcomm;

  nsites = le_nsites();
  obj->u = calloc(obj->nf*nsites, sizeof(double));
  if (obj->u == NULL) fatal("calloc(hydro->u) failed\n");

  obj->f = calloc(obj->nf*nsites, sizeof(double));
  if (obj->f == NULL) fatal("calloc(hydro->f) failed\n");

  field_init_mpi_indexed(nlocal, nhalo, nhcomm, obj->nf, obj->uhalo);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_free
 *
 *****************************************************************************/

void hydro_free(hydro_t * obj) {

  assert(obj);

  MPI_Type_free(&obj->uhalo[Z]);
  MPI_Type_free(&obj->uhalo[Y]);
  MPI_Type_free(&obj->uhalo[X]);
  free(obj->f);
  free(obj->u);
  free(obj);
  obj = NULL;

  return;
}

/*****************************************************************************
 *
 *  hydro_u_halo
 *
 *****************************************************************************/

int hydro_u_halo(hydro_t * obj) {

  int nhalo;
  int nlocal[3];

  assert(obj);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  field_halo(nlocal, nhalo, obj->nhcomm, obj->nf, obj->u, obj->uhalo,
	     cart_comm());

  return 0;
}

/*****************************************************************************
 *
 *  field_init_mpi_indexed
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

int field_init_mpi_indexed(int nlocal[3], int nhalo, int nhcomm, int nf,
			   MPI_Datatype halo[3]) {

  int ic, jc, n;
  int nh[3];             /* Length of full system in memory nlocal + 2*nhalo */
  int nstripx, nstripy;  /* Length of strips nlocal + 2*nhcomm */
  int ncount;            /* Count for the indexed type */
  int * blocklen;        /* Array of block lengths */
  int * displace;        /* Array of displacements */

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

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
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

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &halo[X]);
  MPI_Type_commit(&halo[X]);

  free(displace);
  free(blocklen);

  /* Y direction */
  /* We can use (nlocal[X] + 2*nhcomm) contiguous strips of nlocal[Z],
   * repeated for each halo layer required. The strides start at zero,
   * and increment by the full nf*nh[Y]*nh[Z] for each strip. */

  nstripx = nlocal[X] + 2*nhcomm;
  ncount = nhcomm*nstripx;

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
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

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &halo[Y]);
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

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
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

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &halo[Z]);
  MPI_Type_commit(&halo[Z]);

  free(displace);
  free(blocklen);

  return 0;
}

/*****************************************************************************
 *
 *  field_halo
 *
 *  General halo swap for lattice field quantities.
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

int field_halo(int nlocal[3], int nhalo, int nhcomm, int nf, double * f,
	       MPI_Datatype halo[3], MPI_Comm comm) {

  int ic, jc, kc, ihalo, ireal;
  int pforw, pback;
  int n, nh;

  MPI_Request req_send[6];
  MPI_Request req_recv[6];
  MPI_Status  status[6];

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;


  assert(f);

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
 *  hydro_init_io_info
 *
 *  There is no read for the velocity; this should come from the
 *  distribution.
 *
 *****************************************************************************/

int hydro_init_io_info(hydro_t * obj, int grid[3], int form_in, int form_out) {

  assert(obj);
  assert(grid);
  assert(obj->info == NULL);

  obj->info = io_info_create_with_grid(grid);
  if (obj->info == NULL) fatal("io_info_create(hydro) failed\n");

  io_info_set_name(obj->info, "Velocity field");
  io_info_write_set(obj->info, IO_FORMAT_BINARY, hydro_u_write);
  io_info_write_set(obj->info, IO_FORMAT_ASCII, hydro_u_write_ascii);
  io_info_set_bytesize(obj->info, obj->nf*sizeof(double));

  io_info_format_set(obj->info, form_in, form_out);
  io_write_metadata("vel", obj->info);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_io_info
 *
 *****************************************************************************/

int hydro_io_info(hydro_t * obj, io_info_t ** info) {

  assert(obj);
  assert(obj->info); /* Should have been initialised */

  *info = obj->info;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local_set
 *
 *****************************************************************************/

int hydro_f_local_set(hydro_t * obj, int index, const double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->f[obj->nf*index + ia] = force[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local
 *
 *****************************************************************************/

int hydro_f_local(hydro_t * obj, int index, double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    force[ia] = obj->f[obj->nf*index + ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local_add
 *
 *  Accumulate (repeat, accumulate) the fluid force at site index.
 *
 *****************************************************************************/

int hydro_f_local_add(hydro_t * obj, int index, const double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->f[obj->nf*index + ia] += force[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_set
 *
 *****************************************************************************/

int hydro_u_set(hydro_t * obj, int index, const double u[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->u[obj->nf*index + ia] = u[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u
 *
 *****************************************************************************/

int hydro_u(hydro_t * obj, int index, double u[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    u[ia] = obj->u[obj->nf*index + ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_zero
 *
 *****************************************************************************/

int hydro_f_zero(hydro_t * obj, const double uzero[3]) {

  int ic, jc, kc, index;
  int nlocal[3];

  assert(obj);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	hydro_f_local_set(obj, index, uzero);

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_lees_edwards
 *
 *  Compute the 'look-across-the-boundary' values of the velocity field,
 *  and update the velocity buffer region accordingly.
 *
 *  The communication might be improved:
 *  - only one buffer either side of the planes needs to be set?
 *  - only one communication per y sub domain if more than one buffer?
 *
 *****************************************************************************/

int hydro_lees_edwards(hydro_t * obj) {

  int nhalo;
  int nlocal[3]; /* Local system size */
  int ib;        /* Index in buffer region */
  int ib0;       /* buffer region offset */
  int ic;        /* Index corresponding x location in real system */
  int nf;        /* Number of fields */

  int jc, kc, ia, index0, index1, index2;

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  double t;      /* Time */
  int jdy;       /* Integral part of displacement */
  int j1, j2;    /* j values in real system to interpolate between */

  double ule[3]; /* +/- velocity jump at plane */

  assert(obj);

  if (cart_size(Y) > 1) {
    hydro_lees_edwards_parallel(obj);
  }
  else {

    ule[X] = 0.0;
    ule[Y] = 0.0;  /* Only y component will be non-zero */
    ule[Z] = 0.0;

    nf = obj->nf;

    nhalo = coords_nhalo();
    coords_nlocal(nlocal);
    ib0 = nlocal[X] + nhalo + 1;

    t = 1.0*get_step();

    for (ib = 0; ib < le_get_nxbuffer(); ib++) {

      ic = le_index_buffer_to_real(ib);
      dy = le_buffer_displacement(ib, t);

      /* This is a slightly awkward way to compute the velocity
       * jump: just the (+/-) displacement devided by time. */

      ule[Y] = dy/t; /* STEADY SHEAR ONLY */

      dy = fmod(dy, L(Y));
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

	/* Actually required here is j1 = jc - jdy - 1, but there's
	 * horrible modular arithmetic for the periodic boundaries
	 * to ensure 1 <= j1,j2 <= nlocal[Y] */

	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];

	/* If nhcomm < nhalo, we could use nhcomm here in the kc loop.
	 * (As j1 and j2 are always in the domain proper, jc can use nhalo.) */

	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	  index0 = le_site_index(ib0 + ib, jc, kc);
	  index1 = le_site_index(ic, j1, kc);
	  index2 = le_site_index(ic, j2, kc);
	  for (ia = 0; ia < 3; ia++) {
	    obj->u[nf*index0 + ia] = ule[ia] +
	      fr*obj->u[nf*index1 + ia] + (1.0 - fr)*obj->u[nf*index2 + ia];
	  }
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_lees_edwards_parallel
 *
 *  The Lees Edwards transformation for the velocity field in parallel.
 *  This is a linear interpolation.
 *
 *  Note that we communicate with up to 3 processors in each direction;
 *  this avoids having to update the halos completely.
 *
 *****************************************************************************/

static int hydro_lees_edwards_parallel(hydro_t * obj) {

  int      nlocal[3];      /* Local system size */
  int      noffset[3];     /* Local starting offset */
  int ib;                  /* Index in buffer region */
  int ib0;                 /* buffer region offset */
  int ic;                  /* Index corresponding x location in real system */
  int jc, kc, j1, j2;
  int n, n1, n2, n3;
  double dy;               /* Displacement for current ic->ib pair */
  double fr;               /* Fractional displacement */
  double t;                /* time */
  double * buffer;         /* Interpolation buffer */
  int jdy;                 /* Integral part of displacement */
  int index, ia;
  int nf, nhalo;
  double ule[3];

  int      nrank_s[3];     /* send ranks */
  int      nrank_r[3];     /* recv ranks */
  const int tag0 = 1256;
  const int tag1 = 1257;
  const int tag2 = 1258;

  MPI_Comm    le_comm;
  MPI_Request request[6];
  MPI_Status  status[3];

  assert(obj);
  nf = obj->nf;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  ib0 = nlocal[X] + nhalo + 1;

  le_comm = le_communicator();

  /* Allocate the temporary buffer */

  n = (nlocal[Y] + 2*nhalo + 1)*(nlocal[Z] + 2*nhalo);
  buffer = calloc(nf*n, sizeof(double));
  if (buffer == NULL) fatal("hydrodynamics: malloc(le buffer) failed\n");

  t = 1.0*get_step();

  ule[X] = 0.0;
  ule[Z] = 0.0;

  /* One round of communication for each buffer plane */

  for (ib = 0; ib < le_get_nxbuffer(); ib++) {

    ic = le_index_buffer_to_real(ib);
    kc = 1 - nhalo;

    /* Work out the displacement-dependent quantities */

    dy = le_buffer_displacement(ib, t);
    ule[Y] = dy/t; /* STEADY SHEAR ONLY */
    dy = fmod(dy, L(Y));
    jdy = floor(dy);
    fr  = dy - jdy;

    /* First j1 required is j1 = jc - jdy - 1 with jc = 1 - nhalo.
     * Modular arithmetic ensures 1 <= j1 <= N_total(Y). */

    jc = noffset[Y] + 1 - nhalo;
    j1 = 1 + (jc - jdy - 2 + 2*N_total(Y)) % N_total(Y);

    le_jstart_to_ranks(j1, nrank_s, nrank_r);

    /* Local quantities: given a local starting index j2, we receive
     * n1 + n2 sites into the buffer, and send n1 sites starting with
     * j2, and the remaining n2 sites from starting position nhalo. */

    j2 = 1 + (j1 - 1) % nlocal[Y];

    n1 = (nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = imin(nlocal[Y], j2 + 2*nhalo)*(nlocal[Z] + 2*nhalo);
    n3 = imax(0, j2 - nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

    assert((n1+n2+n3) == (nlocal[Y] + 2*nhalo + 1)*(nlocal[Z] + 2*nhalo));

    /* Post receives, sends and wait for receives. */

    MPI_Irecv(buffer, nf*n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);
    MPI_Irecv(buffer + nf*n1, nf*n2, MPI_DOUBLE, nrank_r[1], tag1,
	      le_comm, request + 1);
    MPI_Irecv(buffer + nf*(n1 + n2), nf*n3, MPI_DOUBLE, nrank_r[2], tag2,
	      le_comm, request + 2);

    index = le_site_index(ic, j2, kc);
    MPI_Issend(&obj->u[nf*index], nf*n1, MPI_DOUBLE, nrank_s[0], tag0,
	       le_comm, request + 3);

    index = le_site_index(ic, 1, kc);
    MPI_Issend(&obj->u[nf*index], nf*n2, MPI_DOUBLE, nrank_s[1], tag1,
	       le_comm, request + 4);
    MPI_Issend(&obj->u[nf*index], nf*n3, MPI_DOUBLE, nrank_s[2], tag2,
	       le_comm, request + 5);

    MPI_Waitall(3, request, status);

    /* Perform the actual interpolation from temporary buffer to
     * buffer region. */

    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

      j1 = (jc + nhalo - 1    )*(nlocal[Z] + 2*nhalo);
      j2 = (jc + nhalo - 1 + 1)*(nlocal[Z] + 2*nhalo);

      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	index = le_site_index(ib0 + ib, jc, kc);
	for (ia = 0; ia < 3; ia++) {
	  obj->u[nf*index + ia] = ule[ia]
	    + fr*buffer[nf*(j1 + kc + nhalo - 1) + ia]
	    + (1.0 - fr)*buffer[nf*(j2 + kc + nhalo - 1) + ia];
	}
      }
    }

    MPI_Waitall(3, request + 3, status);
  }

  free(buffer);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_write
 *
 *****************************************************************************/

static int hydro_u_write(FILE * fp, int index, void * arg) {

  int n;
  hydro_t * obj = arg;

  assert(fp);
  assert(obj);

  n = fwrite(&obj->u[obj->nf*index], sizeof(double), obj->nf, fp);
  if (n != obj->nf) fatal("fwrite(hydro->u) failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_write_ascii
 *
 *****************************************************************************/

static int hydro_u_write_ascii(FILE * fp, int index, void * arg) {

  int n;
  hydro_t * obj = arg;

  assert(fp);
  assert(obj);

  n = fprintf(fp, "%22.15e %22.15e %22.15e\n", obj->u[obj->nf*index + X],
	      obj->u[obj->nf*index + Y], obj->u[obj->nf*index + Z]);

  /* Expect total of 69 characters ... */
  if (n != 69) fatal("fprintf(hydro->u) failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_gradient_tensor
 *
 *  Return the velocity gradient tensor w_ab = d_b u_a at
 *  the site (ic, jc, kc).
 *
 *  The differencing is 2nd order centred.
 *
 *  This must take account of the Lees Edwards planes in  the x-direction.
 *
 *****************************************************************************/

int hydro_u_gradient_tensor(hydro_t * obj, int ic, int jc, int kc,
			    double w[3][3]) {
  int im1, ip1;

  assert(obj);

  im1 = le_index_real_to_buffer(ic, -1);
  im1 = obj->nf*le_site_index(im1, jc, kc);
  ip1 = le_index_real_to_buffer(ic, +1);
  ip1 = obj->nf*le_site_index(ip1, jc, kc);

  w[X][X] = 0.5*(obj->u[ip1 + X] - obj->u[im1 + X]);
  w[Y][X] = 0.5*(obj->u[ip1 + Y] - obj->u[im1 + Y]);
  w[Z][X] = 0.5*(obj->u[ip1 + Z] - obj->u[im1 + Z]);

  im1 = obj->nf*le_site_index(ic, jc - 1, kc);
  ip1 = obj->nf*le_site_index(ic, jc + 1, kc);

  w[X][Y] = 0.5*(obj->u[ip1 + X] - obj->u[im1 + X]);
  w[Y][Y] = 0.5*(obj->u[ip1 + Y] - obj->u[im1 + Y]);
  w[Z][Y] = 0.5*(obj->u[ip1 + Z] - obj->u[im1 + Z]);

  im1 = obj->nf*le_site_index(ic, jc, kc - 1);
  ip1 = obj->nf*le_site_index(ic, jc, kc + 1);

  w[X][Z] = 0.5*(obj->u[ip1 + X] - obj->u[im1 + X]);
  w[Y][Z] = 0.5*(obj->u[ip1 + Y] - obj->u[im1 + Y]);
  w[Z][Z] = 0.5*(obj->u[ip1 + Z] - obj->u[im1 + Z]);

  return 0;
}
