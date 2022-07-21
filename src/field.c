/*****************************************************************************
 *
 *  field.c
 *
 *  Rank 1 objects: scalar fields, vector fields, and compressed tensor Q_ab.
 *
 *  The data storage order is determined in memory.h.
 *
 *  Lees-Edwards transformation is supported provided the lees_edw_t
 *  object is supplied at initialisation time. Otherwise, the normal
 *  cs_t coordinate system applies.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Aln Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "io_harness.h"

#include "timer.h"
#include "util.h"
#include "field.h"

static int field_write(FILE * fp, int index, void * self);
static int field_write_ascii(FILE * fp, int index, void * self);
static int field_read(FILE * fp, int index, void * self);
static int field_read_ascii(FILE * fp, int index, void * self);

static int field_leesedwards_parallel(field_t * obj);

__host__ int field_init(field_t * obj, int nhcomm, lees_edw_t * le);

/*****************************************************************************
 *
 *  field_create
 *
 *  le_t * le may be NULL if no Lees Edwards planes are present.
 *
 *  This just sets the type of field; often order parameter, e.g.,:
 *     nf = 1 for scalar "phi"
 *     nf = 3 for vector "p"
 *     nf = 5 for tensor "q" (compressed traceless, symmetric)
 *
 *****************************************************************************/

__host__ int field_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			  const char * name,
			  const field_options_t * opts,
			  field_t ** pobj) {

  field_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(name);
  assert(opts);
  assert(pobj);

  if (field_options_valid(opts) == 0) {
    pe_fatal(pe, "Internal error: invalid field options\n");
  }

  obj = (field_t *) calloc(1, sizeof(field_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(obj) failed\n");

  obj->nf = opts->ndata;

  obj->name = (char *) calloc(strlen(name) + 1, sizeof(char));
  assert(obj->name);
  if (obj->name == NULL) pe_fatal(pe, "calloc(name) failed\n");

  strncpy(obj->name, name, imin(strlen(name), BUFSIZ));
  obj->name[strlen(name)] = '\0';

  obj->pe = pe;
  obj->cs = cs;
  pe_retain(pe);
  cs_retain(cs);

  field_init(obj, opts->nhcomm, le);
  field_halo_create(obj, &obj->h);
  obj->opts = *opts;

  if (obj->opts.haloverbose) field_halo_info(obj);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  field_free
 *
 *****************************************************************************/

__host__ int field_free(field_t * obj) {

  int ndevice;
  double * tmp;

  assert(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpMemcpy(&tmp, &obj->target->data, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpFree(obj->target);
  }

  if (obj->data) free(obj->data);
  if (obj->name) free(obj->name);
  if (obj->halo) halo_swap_free(obj->halo);
  if (obj->info) io_info_free(obj->info);

  field_halo_free(&obj->h);

  cs_free(obj->cs);
  pe_free(obj->pe);
  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  field_init
 *
 *  Initialise the lattice data, MPI halo information.
 *
 *  The le_t may be NULL, in which case create an instance with
 *  no planes.
 *
 *  TODO:
 *  The behaviour with no planes (cs_t only) could be refactored
 *  into two separate classes. 
 *
 *****************************************************************************/

__host__ int field_init(field_t * obj, int nhcomm, lees_edw_t * le) {

  int ndevice;
  int nsites;
  size_t nfsz;
  double * tmp;

  assert(obj);
  assert(obj->data == NULL);

  cs_nsites(obj->cs, &nsites);
  if (le) lees_edw_nsites(le, &nsites);

  obj->le = le;
  obj->nhcomm = nhcomm;
  obj->nsites = nsites;
  nfsz = (size_t) obj->nf*nsites;

#ifndef OLD_DATA
  obj->data = (double *) calloc(nfsz, sizeof(double));
  if (obj->data == NULL) pe_fatal(obj->pe, "calloc(obj->data) failed\n");
#else
  obj->data = (double *) mem_aligned_malloc(MEM_PAGESIZE, nfsz*sizeof(double));
  if (obj->data == NULL) pe_fatal(obj->pe, "calloc(obj->data) failed\n");
#endif

  /* Allocate target copy of structure (or alias) */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    cs_t * cstarget = NULL;
    lees_edw_t * letarget = NULL;
    tdpMalloc((void **) &obj->target, sizeof(field_t));
    tdpMalloc((void **) &tmp, nfsz*sizeof(double));
    tdpMemcpy(&obj->target->data, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);

    cs_target(obj->cs, &cstarget);
    if (le) lees_edw_target(obj->le, &letarget);
    tdpMemcpy(&obj->target->cs, &cstarget, sizeof(cs_t *),
	      tdpMemcpyHostToDevice);
    tdpMemcpy(&obj->target->le, &letarget, sizeof(lees_edw_t *),
	      tdpMemcpyHostToDevice);
    field_memcpy(obj, tdpMemcpyHostToDevice);
  }

  /* MPI datatypes for halo */

  halo_swap_create_r1(obj->pe, obj->cs, nhcomm, nsites, obj->nf, &obj->halo);
  assert(obj->halo);

  halo_swap_handlers_set(obj->halo, halo_swap_pack_rank1, halo_swap_unpack_rank1);

  return 0;
}

/*****************************************************************************
 *
 *  field_memcpy
 *
 *****************************************************************************/

__host__ int field_memcpy(field_t * obj, tdpMemcpyKind flag) {

  int ndevice;
  size_t nfsz;
  double * tmp;

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {

    nfsz = (size_t) obj->nf*obj->nsites;
    tdpMemcpy(&tmp, &obj->target->data, sizeof(double *),
	      tdpMemcpyDeviceToHost);

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpMemcpy(&obj->target->nf, &obj->nf, sizeof(int), flag);
      tdpMemcpy(&obj->target->nhcomm, &obj->nhcomm, sizeof(int), flag);
      tdpMemcpy(&obj->target->nsites, &obj->nsites, sizeof(int), flag);
      tdpMemcpy(tmp, obj->data, nfsz*sizeof(double), flag);
      break;
    case tdpMemcpyDeviceToHost:
      tdpMemcpy(obj->data, tmp, nfsz*sizeof(double), flag);
      break;
    default:
      pe_fatal(obj->pe, "Bad flag in field_memcpy\n");
      break;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_nf
 *
 *****************************************************************************/

__host__ __device__ int field_nf(field_t * obj, int * nf) {

  assert(obj);
  assert(nf);

  *nf = obj->nf;

  return 0;
}

/*****************************************************************************
 *
 *  field_init_io_info
 *
 *****************************************************************************/

__host__ int field_init_io_info(field_t * obj, int grid[3], int form_in,
				int form_out) {

  io_info_args_t args = io_info_args_default();

  assert(obj);
  assert(obj->info == NULL);

  args.grid[X] = grid[X];
  args.grid[Y] = grid[Y];
  args.grid[Z] = grid[Z];

  io_info_create(obj->pe, obj->cs, &args, &obj->info);
  if (obj->info == NULL) pe_fatal(obj->pe, "io_info_create(field) failed\n");

  io_info_set_name(obj->info, obj->name);
  io_info_write_set(obj->info, IO_FORMAT_BINARY, field_write);
  io_info_write_set(obj->info, IO_FORMAT_ASCII, field_write_ascii);
  io_info_read_set(obj->info, IO_FORMAT_BINARY, field_read);
  io_info_read_set(obj->info, IO_FORMAT_ASCII, field_read_ascii);

  /* ASCII format size is 23 bytes per element plus a '\n' */
  io_info_set_bytesize(obj->info, IO_FORMAT_BINARY, obj->nf*sizeof(double));
  io_info_set_bytesize(obj->info, IO_FORMAT_ASCII, (obj->nf*23 + 1));

  io_info_format_set(obj->info, form_in, form_out);
  io_info_metadata_filestub_set(obj->info, obj->name);

  return 0;
}

/*****************************************************************************
 *
 *  field_io_info
 *
 *****************************************************************************/

__host__ int field_io_info(field_t * obj, io_info_t ** info) {

  assert(obj);
  assert(obj->info);
  assert(info);

  *info = obj->info;

  return 0;
}

/*****************************************************************************
 *
 *  field_halo
 *
 *****************************************************************************/

__host__ int field_halo(field_t * obj) {

  int nlocal[3];
  assert(obj);

  cs_nlocal(obj->cs, nlocal);

  if (nlocal[Z] < obj->nhcomm) {
    /* This constraint means can't use target method;
     * this also requires a copy if the address spaces are distinct. */
    field_memcpy(obj, tdpMemcpyDeviceToHost);
    field_halo_swap(obj, FIELD_HALO_HOST);
    field_memcpy(obj, tdpMemcpyHostToDevice);
  }
  else {
    /* Default to ... */
    field_halo_swap(obj, obj->opts.haloscheme);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_swap
 *
 *****************************************************************************/

__host__ int field_halo_swap(field_t * obj, field_halo_enum_t flag) {

  double * data;

  assert(obj);

  switch (flag) {
  case FIELD_HALO_HOST:
    halo_swap_host_rank1(obj->halo, obj->data, MPI_DOUBLE);
    break;
  case FIELD_HALO_TARGET:
    // tdpMemcpy(&data, &obj->target->data, sizeof(double *),
	  //     tdpMemcpyDeviceToHost);
    // halo_swap_packed(obj->halo, data);
    // break;
  case FIELD_HALO_OPENMP:
    field_halo_post(obj, &obj->h);
    field_halo_wait(obj, &obj->h);
    break;
  default:
    assert(0);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_leesedwards
 *
 *  Interpolate the phi field to take account of any local Lees Edwards
 *  boundaries.
 *
 *  The buffer region of obj->data[] is updated with the interpolated
 *  values.
 *
 *****************************************************************************/

__host__ int field_leesedwards(field_t * obj) {

  int nf;
  int nhalo;
  int nlocal[3]; /* Local system size */
  int nxbuffer;  /* Number of buffer planes */
  int ib;        /* Index in buffer region */
  int ib0;       /* buffer region offset */
  int ic;        /* Index corresponding x location in real system */
  int jc, kc, n;
  int index, index0, index1, index2, index3;
  int mpi_cartsz[3];

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  double ltot[3];
  const double r6 = (1.0/6.0);

  int jdy;               /* Integral part of displacement */
  int j0, j1, j2, j3;    /* j values in real system to interpolate between */

  assert(obj);
  assert(obj->data);

  if (obj->le == NULL) return 0;

  cs_ltot(obj->cs, ltot);
  cs_cartsz(obj->cs, mpi_cartsz);

  /* At the moment we require some copies for device version ... */
  /* ... here and at the end. */
  {
    int nplane = lees_edw_nplane_total(obj->le);
    if (nplane > 0) field_memcpy(obj, tdpMemcpyDeviceToHost);
  }

  if (mpi_cartsz[Y] > 1) {
    /* This has its own routine. */
    field_leesedwards_parallel(obj);
  }
  else {
    /* No messages are required... */

    nf = obj->nf;
    cs_nhalo(obj->cs, &nhalo);
    cs_nlocal(obj->cs, nlocal);
    lees_edw_nxbuffer(obj->le, &nxbuffer);
    ib0 = nlocal[X] + nhalo + 1;

    for (ib = 0; ib < nxbuffer; ib++) {

      ic = lees_edw_ibuff_to_real(obj->le, ib);

      lees_edw_buffer_dy(obj->le, ib, 0.0, &dy);
      dy = fmod(dy, ltot[Y]);
      jdy = floor(dy);
      fr  = 1.0 - (dy - jdy);

      for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

        /* Note that a linear interpolation here would involve
         * (1.0 - fr)*phi(ic,j1,kc) + fr*phi(ic,j2,kc)
         * This is just Lagrange four-point instead. */

        j0 = 1 + (jc - jdy - 3 + 2*nlocal[Y]) % nlocal[Y];
        j1 = 1 + j0 % nlocal[Y];
        j2 = 1 + j1 % nlocal[Y];
        j3 = 1 + j2 % nlocal[Y];

        for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
          index  = lees_edw_index(obj->le, ib0 + ib, jc, kc);
          index0 = lees_edw_index(obj->le, ic, j0, kc);
          index1 = lees_edw_index(obj->le, ic, j1, kc);
          index2 = lees_edw_index(obj->le, ic, j2, kc);
          index3 = lees_edw_index(obj->le, ic, j3, kc);
          for (n = 0; n < nf; n++) {
            obj->data[addr_rank1(obj->nsites, nf, index, n)] =
              -  r6*fr*(fr-1.0)*(fr-2.0)*obj->data[addr_rank1(obj->nsites, nf, index0, n)]
              + 0.5*(fr*fr-1.0)*(fr-2.0)*obj->data[addr_rank1(obj->nsites, nf, index1, n)]
              - 0.5*fr*(fr+1.0)*(fr-2.0)*obj->data[addr_rank1(obj->nsites, nf, index2, n)]
              +        r6*fr*(fr*fr-1.0)*obj->data[addr_rank1(obj->nsites, nf, index3, n)];
          }
        }
      }
    }
  }

  {
    int nplane = lees_edw_nplane_total(obj->le);
    if (nplane > 0) field_memcpy(obj, tdpMemcpyHostToDevice);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_leesedwards_parallel
 *
 *  The Lees Edwards transformation requires a certain amount of
 *  communication in parallel.
 *
 *  As we are using a 4-point interpolation, there is a requirement
 *  to communicate with as many as three different processors to
 *  send/receive appropriate interpolated values.
 *
 *  Note that the sends only involve the 'real' system, so there is
 *  no requirement that the halos be up-to-date (although it is
 *  expected that they will be for the gradient calculation).
 *
 *****************************************************************************/

static int field_leesedwards_parallel(field_t * obj) {

  int nf;
  int nlocal[3];           /* Local system size */
  int noffset[3];          /* Local starting offset */
  int nxbuffer;            /* Number of buffer planes */
  int ib;                  /* Index in buffer region */
  int ib0;                 /* buffer region offset */
  int ic;                  /* Index corresponding x location in real system */
  int jc, kc;
  int j0, j1, j2, j3;
  int n, n1, n2, n3;
  int nhalo;
  int jdy;                 /* Integral part of displacement */
  int index;
  int ntotal[3];

  double dy;               /* Displacement for current ic->ib pair */
  double fr;               /* Fractional displacement */
  double ltot[3];
  const double r6 = (1.0/6.0);

  int      nsend;          /* Send buffer size */
  int      nrecv;          /* Recv buffer size */
  int      nrank_s[3];     /* send ranks */
  int      nrank_r[3];     /* recv ranks */
  double * sendbuf = NULL; /* Send buffer */
  double * recvbuf = NULL; /* Interpolation buffer */

  const int tag0 = 1256;
  const int tag1 = 1257;
  const int tag2 = 1258;

  MPI_Comm le_comm;
  MPI_Request request[6];
  MPI_Status  status[3];

  assert(obj);
  assert(obj->le);

  field_nf(obj, &nf);
  cs_ltot(obj->cs, ltot);
  cs_nhalo(obj->cs, &nhalo);
  cs_nlocal(obj->cs, nlocal);
  cs_ntotal(obj->cs, ntotal);
  cs_nlocal_offset(obj->cs, noffset);
  ib0 = nlocal[X] + nhalo + 1;

  lees_edw_comm(obj->le, &le_comm);
  lees_edw_nxbuffer(obj->le, &nxbuffer);

  /* Allocate buffer space */

  nsend = nf*nlocal[Y]*(nlocal[Z] + 2*nhalo);
  nrecv = nf*(nlocal[Y] + 2*nhalo + 3)*(nlocal[Z] + 2*nhalo);

  sendbuf = (double *) malloc(nsend*sizeof(double));
  recvbuf = (double *) malloc(nrecv*sizeof(double));

  if (sendbuf == NULL) pe_fatal(obj->pe, "malloc(sendbuf) failed\n");
  if (recvbuf == NULL) pe_fatal(obj->pe, "malloc(recvbuf) failed\n");

  /* One round of communication for each buffer plane */

  for (ib = 0; ib < nxbuffer; ib++) {

    ic = lees_edw_ibuff_to_real(obj->le, ib);

    /* Work out the displacement-dependent quantities */

    lees_edw_buffer_dy(obj->le, ib, 0.0, &dy);
    dy = fmod(dy, ltot[Y]);
    jdy = floor(dy);
    fr  = 1.0 - (dy - jdy);
    /* In the real system the first point we require is
     * j1 = jc - jdy - 3
     * with jc = noffset[Y] + 1 - nhalo in the global coordinates.
     * Modular arithmetic ensures 1 <= j1 <= ntotal[Y] */

    jc = noffset[Y] + 1 - nhalo;
    j1 = 1 + (jc - jdy - 3 + 2*ntotal[Y]) % ntotal[Y];
    assert(j1 >= 1);
    assert(j1 <= ntotal[Y]);

    lees_edw_jstart_to_mpi_ranks(obj->le, j1, nrank_s, nrank_r);

    /* Local quantities: j2 is the position of j1 in local coordinates.
     * The three sections to send/receive are organised as follows:
     * jc is the number of j points in each case, while n is the
     * total number of data items. Note that n3 can be zero. */

    j2 = 1 + (j1 - 1) % nlocal[Y];
    assert(j2 >= 1);
    assert(j2 <= nlocal[Y]);

    jc = nlocal[Y] - j2 + 1;
    n1 = nf*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(recvbuf, n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);

    jc = imin(nlocal[Y], j2 + 2*nhalo + 2);
    n2 = nf*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(recvbuf + n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm,
              request + 1);

    jc = imax(0, j2 - nlocal[Y] + 2*nhalo + 2);
    n3 = nf*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(recvbuf + n1 + n2, n3, MPI_DOUBLE, nrank_r[2], tag2, le_comm,
              request + 2);

    /* Load contiguous send buffer */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	index = lees_edw_index(obj->le, ic, jc, kc);
	for (n = 0; n < nf; n++) {
	  j0 = nf*(jc - 1)*(nlocal[Z] + 2*nhalo);
	  j1 = j0 + nf*(kc + nhalo - 1);
	  assert((j1+n) >= 0 && (j1+n) < nsend);
	  sendbuf[j1+n] = obj->data[addr_rank1(obj->nsites, nf, index, n)];
	}
      }
    }

    /* Post sends and wait for receives. */

    index = nf*(j2 - 1)*(nlocal[Z] + 2*nhalo);
    MPI_Issend(sendbuf+index, n1, MPI_DOUBLE, nrank_s[0], tag0, le_comm,
               request + 3);
    MPI_Issend(sendbuf, n2, MPI_DOUBLE, nrank_s[1], tag1, le_comm,
               request + 4);
    MPI_Issend(sendbuf, n3, MPI_DOUBLE, nrank_s[2], tag2, le_comm,
               request + 5);

    MPI_Waitall(3, request, status);


    /* Perform the actual interpolation from temporary buffer. */

    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

      /* Note that the linear interpolation here would be
       * (1.0-fr)*buffer(j1, k, n) + fr*buffer(j2, k, n)
       * This is again Lagrange four point. */

      j0 = (jc + nhalo - 1    )*(nlocal[Z] + 2*nhalo);
      j1 = (jc + nhalo - 1 + 1)*(nlocal[Z] + 2*nhalo);
      j2 = (jc + nhalo - 1 + 2)*(nlocal[Z] + 2*nhalo);
      j3 = (jc + nhalo - 1 + 3)*(nlocal[Z] + 2*nhalo);

      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
        index = lees_edw_index(obj->le, ib0 + ib, jc, kc);
        for (n = 0; n < nf; n++) {
          obj->data[addr_rank1(obj->nsites, nf, index, n)] =
            -  r6*fr*(fr-1.0)*(fr-2.0)*recvbuf[nf*(j0 + kc+nhalo-1) + n]
            + 0.5*(fr*fr-1.0)*(fr-2.0)*recvbuf[nf*(j1 + kc+nhalo-1) + n]
            - 0.5*fr*(fr+1.0)*(fr-2.0)*recvbuf[nf*(j2 + kc+nhalo-1) + n]
            +        r6*fr*(fr*fr-1.0)*recvbuf[nf*(j3 + kc+nhalo-1) + n];
        }
      }
    }

    /* Clean up the sends, and move to next buffer location. */

    MPI_Waitall(3, request + 3, status);
  }

  free(recvbuf);
  free(sendbuf);

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar
 *
 *****************************************************************************/

__host__ __device__
int field_scalar(field_t * obj, int index, double * phi) {

  assert(obj);
  assert(obj->nf == 1);
  assert(obj->data);
  assert(phi);

  *phi = obj->data[addr_rank1(obj->nsites, 1, index, 0)];

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar_set
 *
 *****************************************************************************/

__host__ __device__
int field_scalar_set(field_t * obj, int index, double phi) {

  assert(obj);
  assert(obj->nf == 1);
  assert(obj->data);

  obj->data[addr_rank1(obj->nsites, 1, index, 0)] = phi;

  return 0;
}

/*****************************************************************************
 *
 *  field_vector
 *
 *****************************************************************************/

__host__ __device__
int field_vector(field_t * obj, int index, double p[3]) {

  int ia;

  assert(obj);
  assert(obj->nf == 3);
  assert(obj->data);

  for (ia = 0; ia < 3; ia++) {
    p[ia] = obj->data[addr_rank1(obj->nsites, 3, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_vector_set
 *
 *****************************************************************************/

__host__ __device__
int field_vector_set(field_t * obj, int index, const double p[3]) {

  int ia;

  assert(obj);
  assert(obj->nf == 3);
  assert(obj->data);
  assert(p);

  for (ia = 0; ia < 3; ia++) {
    obj->data[addr_rank1(obj->nsites, 3, index, ia)] = p[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_tensor
 *
 *  The tensor is expanded from the compressed form.
 *
 *****************************************************************************/

__host__ __device__
int field_tensor(field_t * obj, int index, double q[3][3]) {

  assert(obj);
  assert(obj->nf == NQAB);
  assert(obj->data);
  assert(q);

  q[X][X] = obj->data[addr_rank1(obj->nsites, NQAB, index, XX)];
  q[X][Y] = obj->data[addr_rank1(obj->nsites, NQAB, index, XY)];
  q[X][Z] = obj->data[addr_rank1(obj->nsites, NQAB, index, XZ)];
  q[Y][X] = q[X][Y];
  q[Y][Y] = obj->data[addr_rank1(obj->nsites, NQAB, index, YY)];
  q[Y][Z] = obj->data[addr_rank1(obj->nsites, NQAB, index, YZ)];
  q[Z][X] = q[X][Z];
  q[Z][Y] = q[Y][Z];
  q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];

  return 0;
}

/*****************************************************************************
 *
 *  field_tensor_set
 *
 *  The tensor supplied should be traceless and symmetric, as it will
 *  be stored in 'compressed' form.
 *
 *****************************************************************************/

__host__ __device__
int field_tensor_set(field_t * obj, int index, double q[3][3]) {

  assert(obj);
  assert(obj->nf == NQAB);
  assert(obj->data);
  assert(q);

  obj->data[addr_rank1(obj->nsites, NQAB, index, XX)] = q[X][X];
  obj->data[addr_rank1(obj->nsites, NQAB, index, XY)] = q[X][Y];
  obj->data[addr_rank1(obj->nsites, NQAB, index, XZ)] = q[X][Z];
  obj->data[addr_rank1(obj->nsites, NQAB, index, YY)] = q[Y][Y];
  obj->data[addr_rank1(obj->nsites, NQAB, index, YZ)] = q[Y][Z];

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar_array
 *
 *  Return whatever field data there are for this index in a flattened
 *  1d array of length obj->nf.
 *
 *  Array must be of at least obj->nf, but there is no check.
 *
 *****************************************************************************/

__host__ __device__
int field_scalar_array(field_t * obj, int index, double * array) {

  int n;

  assert(obj);
  assert(obj->data);
  assert(array);

  for (n = 0; n < obj->nf; n++) {
    array[n] = obj->data[addr_rank1(obj->nsites, obj->nf, index, n)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar_array_set
 *
 *****************************************************************************/

__host__ __device__
int field_scalar_array_set(field_t * obj, int index, const double * array) {

  int n;

  assert(obj);
  assert(obj->data);
  assert(array);

  for (n = 0; n < obj->nf; n++) {
    obj->data[addr_rank1(obj->nsites, obj->nf, index, n)] = array[n];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_read
 *
 *****************************************************************************/

static int field_read(FILE * fp, int index, void * self) {

  int n;
  double array[NQAB];              /* Largest field currently expected */
  field_t * obj = (field_t*) self;

  assert(fp);
  assert(obj);
  assert(obj->nf <= NQAB);

  n = fread(array, sizeof(double), obj->nf, fp);
  if (n != obj->nf) {
    pe_fatal(obj->pe, "fread(field) failed at index %d", index);
  }

  field_scalar_array_set(obj, index, array);

  return 0;
}

/*****************************************************************************
 *
 *  field_read_ascii
 *
 *****************************************************************************/

static int field_read_ascii(FILE * fp, int index, void * self) {

  int n, nread;
  double array[NQAB];                /* Largest currently expected */
  field_t * obj =  (field_t*) self;

  assert(fp);
  assert(obj);
  assert(obj->nf <= NQAB);

  for (n = 0; n < obj->nf; n++) {
    nread = fscanf(fp, "%le", array + n);
    if (nread != 1) {
      pe_fatal(obj->pe, "fscanf(field) failed at index %d\n", index);
    }
  }

  field_scalar_array_set(obj, index, array);

  return 0;
}

/*****************************************************************************
 *
 *  field_write
 *
 *****************************************************************************/

static int field_write(FILE * fp, int index, void * self) {

  int n;
  double array[NQAB];               /* Largest currently expected */
  field_t * obj =  (field_t*) self;

  assert(fp);
  assert(obj);
  assert(obj->nf <= NQAB);

  field_scalar_array(obj, index, array);

  n = fwrite(array, sizeof(double), obj->nf, fp);
  if (n != obj->nf) {
    pe_fatal(obj->pe, "fwrite(field) failed at index %d\n", index);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_write_ascii
 *
 *****************************************************************************/

static int field_write_ascii(FILE * fp, int index, void * self) {

  int n, nwrite;
  double array[NQAB];               /* Largest currently expected */
  field_t * obj =  (field_t*) self;

  assert(fp);
  assert(obj);
  assert(obj->nf <= NQAB);

  field_scalar_array(obj, index, array);

  for (n = 0; n < obj->nf; n++) {
    nwrite = fprintf(fp, "%22.15e ", array[n]);
    if (nwrite != 23) {
      pe_fatal(obj->pe, "fprintf(%s) failed at index %d\n", obj->name, index);
    }
  }

  nwrite = fprintf(fp, "\n");
  if (nwrite != 1) {
    pe_fatal(obj->pe, "fprintf(%s) failed at index %d\n", obj->name, index);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_size
 *
 *****************************************************************************/

int field_halo_size(cs_limits_t lim) {

  int szx = 1 + lim.imax - lim.imin;
  int szy = 1 + lim.jmax - lim.jmin;
  int szz = 1 + lim.kmax - lim.kmin;

  return szx*szy*szz;
}

/*****************************************************************************
 *
 *  field_halo_enqueue_send
 *
 *****************************************************************************/

int field_halo_enqueue_send(const field_t * field, field_halo_t * h, int ireq) {

  assert(field);
  assert(h);
  assert(1 <= ireq && ireq < h->nvel);

  int nx = 1 + h->slim[ireq].imax - h->slim[ireq].imin;
  int ny = 1 + h->slim[ireq].jmax - h->slim[ireq].jmin;
  int nz = 1 + h->slim[ireq].kmax - h->slim[ireq].kmin;

  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

#pragma omp for nowait
  for (int ih = 0; ih < nx*ny*nz; ih++) {
    int ic = h->slim[ireq].imin + ih/strx;
    int jc = h->slim[ireq].jmin + (ih % strx)/stry;
    int kc = h->slim[ireq].kmin + (ih % stry)/strz;
    int index = cs_index(field->cs, ic, jc, kc);

    for (int ibf = 0; ibf < field->nf; ibf++) {
      int faddr = addr_rank1(field->nsites, field->nf, index, ibf);
      h->send[ireq][ih*field->nf + ibf] = field->data[faddr];
    }
  }

  return 0;
}

__global__
void field_halo_enqueue_send_kernel(const field_t * field, field_halo_t * h, int ireq) {
  assert(field);
  assert(h);
  assert(1 <= ireq && ireq < h->nvel);

  int nx = 1 + h->slim[ireq].imax - h->slim[ireq].imin;
  int ny = 1 + h->slim[ireq].jmax - h->slim[ireq].jmin;
  int nz = 1 + h->slim[ireq].kmax - h->slim[ireq].kmin;

  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  int ih = 0;
  for_simt_parallel(ih, nx*ny*nz, 1) {
    int ic = h->slim[ireq].imin + ih/strx;
    int jc = h->slim[ireq].jmin + (ih % strx)/stry;
    int kc = h->slim[ireq].kmin + (ih % stry)/strz;
    int index = cs_index(field->cs, ic, jc, kc);

    for (int ibf = 0; ibf < field->nf; ibf++) {
      int faddr = addr_rank1(field->nsites, field->nf, index, ibf);
      h->send[ireq][ih*field->nf + ibf] = field->data[faddr];
    }
  }
}

/*****************************************************************************
 *
 *  field_halo_dequeue_recv
 *
 *****************************************************************************/

int field_halo_dequeue_recv(field_t * field, const field_halo_t * h, int ireq) {
  assert(field);
  assert(h);
  assert(1 <= ireq && ireq < h->nvel);

  int nx = 1 + h->rlim[ireq].imax - h->rlim[ireq].imin;
  int ny = 1 + h->rlim[ireq].jmax - h->rlim[ireq].jmin;
  int nz = 1 + h->rlim[ireq].kmax - h->rlim[ireq].kmin;

  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  double * recv = h->recv[ireq];

  /* Check if this a copy from our own send buffer */
  {
    int i = 1 + h->cv[h->nvel - ireq][X];
    int j = 1 + h->cv[h->nvel - ireq][Y];
    int k = 1 + h->cv[h->nvel - ireq][Z];

    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) recv = h->send[ireq];
  }

  #pragma omp for nowait
  for (int ih = 0; ih < nx*ny*nz; ih++) {
    int ic = h->rlim[ireq].imin + ih/strx;
    int jc = h->rlim[ireq].jmin + (ih % strx)/stry;
    int kc = h->rlim[ireq].kmin + (ih % stry)/strz;
    int index = cs_index(field->cs, ic, jc, kc);

    for (int ibf = 0; ibf < field->nf; ibf++) {
      int faddr = addr_rank1(field->nsites, field->nf, index, ibf);
      field->data[faddr] = recv[ih*field->nf + ibf];
    }
  }

  return 0;
}

__global__
void field_halo_dequeue_recv_kernel(field_t * field, const field_halo_t * h, int ireq) {
  assert(field);
  assert(h);
  assert(1 <= ireq && ireq < h->nvel);

  int nx = 1 + h->rlim[ireq].imax - h->rlim[ireq].imin;
  int ny = 1 + h->rlim[ireq].jmax - h->rlim[ireq].jmin;
  int nz = 1 + h->rlim[ireq].kmax - h->rlim[ireq].kmin;

  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  double * recv = h->recv[ireq];

  /* Check if this a copy from our own send buffer */
  {
    int i = 1 + h->cv[h->nvel - ireq][X];
    int j = 1 + h->cv[h->nvel - ireq][Y];
    int k = 1 + h->cv[h->nvel - ireq][Z];

    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) recv = h->send[ireq];
  }

  int ih = 0;
  for_simt_parallel(ih, nx*ny*nz, 1) {
    int ic = h->rlim[ireq].imin + ih/strx;
    int jc = h->rlim[ireq].jmin + (ih % strx)/stry;
    int kc = h->rlim[ireq].kmin + (ih % stry)/strz;
    int index = cs_index(field->cs, ic, jc, kc);

    for (int ibf = 0; ibf < field->nf; ibf++) {
      int faddr = addr_rank1(field->nsites, field->nf, index, ibf);
      field->data[faddr] = recv[ih*field->nf + ibf];
    }
  }
}

#if defined(__NVCC__)
void CUDART_CB field_halo_send_callback(void *data) {
  // printf("callback() data = %d\n", data);
  const static int tagbase = 2022;

  intptr_t *tmp = (intptr_t *)data;
  field_t *field = (field_t *)(tmp[0]);
  // printf("callback() field = %d\n", field);
  field_halo_t *h = (field_halo_t *)(tmp[1]);
  // printf("callback() h = %d\n", h);
  int ireq = tmp[2];
  // printf("callback() ireq = %d\n", ireq);
  // printf("callback() h->send = %d\n", h->send);
  // printf("callback() h->send[ireq] = %d\n", h->send[ireq]);

  int i = 1 + h->cv[ireq][X];
  int j = 1 + h->cv[ireq][Y];
  int k = 1 + h->cv[ireq][Z];
  int mcount = field->nf*field_halo_size(h->slim[ireq]);

  if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) {
    return;
  }
  
  MPI_Isend(h->send[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
      tagbase + ireq, h->comm, h->request + 27 + ireq);
  // MPI_Ssend(h->send[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
  //     tagbase + ireq, h->comm);
}

int create_send_graph(const field_t * field, field_halo_t * h) {
  h->send_graph = (cuda_graph_t *)malloc(sizeof(*h->send_graph));
  cudaGraphCreate(&h->send_graph->graph, 0);
  for (int ireq = 1; ireq < h->nvel; ireq++) {
    cudaGraphNode_t kernelNode;
    cudaKernelNodeParams kernelNodeParams = {0};
    void *kernelArgs[3] = {(void*)&field->target, 
                            (void*)&h->target, 
                            (void*)&ireq};
    kernelNodeParams.func = (void *)field_halo_enqueue_send_kernel;
    dim3 nblk, ntpb;
    int scount = field->nf*field_halo_size(h->slim[ireq]);
    kernel_launch_param(scount, &nblk, &ntpb);
    kernelNodeParams.gridDim = nblk;
    kernelNodeParams.blockDim = ntpb;
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = (void **)kernelArgs;
    kernelNodeParams.extra = NULL;
    cudaGraphAddKernelNode(&kernelNode, h->send_graph->graph, NULL,
                             0, &kernelNodeParams);
    int i = 1 + h->cv[h->nvel - ireq][X];
    int j = 1 + h->cv[h->nvel - ireq][Y];
    int k = 1 + h->cv[h->nvel - ireq][Z];

    if (h->nbrrank[i][j][k] != h->nbrrank[1][1][1]) {
      cudaGraphNode_t memcpyNode;
      cudaMemcpy3DParms memcpyParams = {0};
      memcpyParams.srcArray = NULL;
      memcpyParams.srcPos = make_cudaPos(0, 0, 0);
      memcpyParams.srcPtr =
          make_cudaPitchedPtr(h->send_d[ireq], sizeof(double) * scount, scount, 1);
      memcpyParams.dstArray = NULL;
      memcpyParams.dstPos = make_cudaPos(0, 0, 0);
      memcpyParams.dstPtr =
          make_cudaPitchedPtr(h->send[ireq], sizeof(double) * scount, scount, 1);
      memcpyParams.extent = make_cudaExtent(sizeof(double) * scount, 1, 1);
      memcpyParams.kind = cudaMemcpyDeviceToHost;
      cudaGraphAddMemcpyNode(&memcpyNode, 
                              h->send_graph->graph, 
                              &kernelNode, 
                              1, &memcpyParams);

      cudaGraphNode_t hostNode;
      h->send_graph->hostCallbackArgs[ireq][0] = (intptr_t)field;
      h->send_graph->hostCallbackArgs[ireq][1] = (intptr_t)h;
      h->send_graph->hostCallbackArgs[ireq][2] = (intptr_t)ireq;
      cudaHostNodeParams hostParams = {0};
      hostParams.fn = field_halo_send_callback;
      hostParams.userData = &h->send_graph->hostCallbackArgs[ireq];
      // printf("create_send() hostParams.userData = %d\n", hostParams.userData);
      // intptr_t* tmp = (intptr_t*)hostParams.userData;
      // printf("create_send() field = %d, h = %d, ireq = %d, h->send[ireq] = %d\n", field, h, ireq, h->send[ireq]);
      // printf("create_send() tmp[0] = %d, tmp[1] = %d, tmp[2] = %d, h->send[ireq] = %d\n", tmp[0], tmp[1], tmp[2], h->send[ireq]);

      // printf("create_send() %d -> %d\n", h->nbrrank[1][1][1], h->nbrrank[i][j][k]);
      // cudaGraphAddHostNode(&hostNode, h->send_graph->graph,
      //                       &memcpyNode,
      //                       1, &hostParams);
    }
  }
  // char errLog[1000];
  cudaGraphInstantiate(&h->send_graph->graphExec, h->send_graph->graph, NULL, NULL, 0);
  // errLog[999] = '\0';
  // printf("send graph log: %s\n", errLog);
  return 0;
}

int create_recv_graph(const field_t * field, field_halo_t * h) {
  h->recv_graph = (cuda_graph_t *)malloc(sizeof(*h->recv_graph));
  cudaGraphCreate(&h->recv_graph->graph, 0);
  for (int ireq = 1; ireq < h->nvel; ireq++) {
    int rcount = field->nf*field_halo_size(h->rlim[ireq]);
    cudaGraphNode_t memcpyNode;
    int i = 1 + h->cv[h->nvel - ireq][X];
    int j = 1 + h->cv[h->nvel - ireq][Y];
    int k = 1 + h->cv[h->nvel - ireq][Z];
    if (h->nbrrank[i][j][k] != h->nbrrank[1][1][1]) {
      cudaMemcpy3DParms memcpyParams = {0};
      memcpyParams.srcArray = NULL;
      memcpyParams.srcPos = make_cudaPos(0, 0, 0);
      memcpyParams.srcPtr =
          make_cudaPitchedPtr(h->recv[ireq], sizeof(double) * rcount, rcount, 1);
      memcpyParams.dstArray = NULL;
      memcpyParams.dstPos = make_cudaPos(0, 0, 0);
      memcpyParams.dstPtr =
          make_cudaPitchedPtr(h->recv_d[ireq], sizeof(double) * rcount, rcount, 1);
      memcpyParams.extent = make_cudaExtent(sizeof(double) * rcount, 1, 1);
      memcpyParams.kind = cudaMemcpyHostToDevice;
      cudaGraphAddMemcpyNode(&memcpyNode, h->recv_graph->graph, NULL, 0, &memcpyParams);
    }
    cudaGraphNode_t node;
    cudaKernelNodeParams kernelNodeParams = {0};
    // h->recv_graph->kernelArgs[ireq][0] = (intptr_t)field->target;
    // h->recv_graph->kernelArgs[ireq][1] = (intptr_t)h->target;
    // h->recv_graph->kernelArgs[ireq][2] = (intptr_t)ireq;
    void *kernelArgs[3] = {(void*)&field->target, 
                            (void*)&h->target, 
                            (void*)&ireq};
    kernelNodeParams.func = (void *)field_halo_dequeue_recv_kernel;
    dim3 nblk, ntpb;
    kernel_launch_param(rcount, &nblk, &ntpb);
    kernelNodeParams.gridDim = nblk;
    kernelNodeParams.blockDim = ntpb;
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = (void **)kernelArgs;
    kernelNodeParams.extra = NULL;
    if (h->nbrrank[i][j][k] != h->nbrrank[1][1][1]) {
      cudaGraphAddKernelNode(&node, h->recv_graph->graph, &memcpyNode,
                             1, &kernelNodeParams);
    } else {
      cudaGraphAddKernelNode(&node, h->recv_graph->graph, NULL,
                             0, &kernelNodeParams);
    }
  }
  cudaGraphInstantiate(&h->recv_graph->graphExec, h->recv_graph->graph, NULL, NULL, 0);
  return 0;
}

#endif
/*****************************************************************************
 *
 *  field_halo_create
 *
 *  It's convenient to borrow the velocity notation from the lb for
 *  the commnunication directions.
 *
 *****************************************************************************/

#include "lb_d3q27.h"

int field_halo_create(const field_t * field, field_halo_t * h) {

  int nlocal[3] = {0};
  int nhalo = 0;

  assert(field);
  assert(h);

  *h = (field_halo_t) {0};

  /* Communictation model */

  cs_cart_comm(field->cs, &h->comm);

  {
    LB_CV_D3Q27(cv27);

    h->nvel = 27;
    for (int p = 0; p < h->nvel; p++) {
      h->cv[p][X] = cv27[p][X];
      h->cv[p][Y] = cv27[p][Y];
      h->cv[p][Z] = cv27[p][Z];
    }
  }

  /* Ranks of Cartesian neighbours */

  {
    int dims[3] = {0};
    int periods[3] = {0};
    int coords[3] = {0};

    MPI_Cart_get(h->comm, 3, dims, periods, coords);

    for (int p = 0; p < h->nvel; p++) {
      int nbr[3] = {0};
      int out[3] = {0};  /* Out-of-range is erroneous for non-perioidic dims */
      int i = 1 + h->cv[p][X];
      int j = 1 + h->cv[p][Y];
      int k = 1 + h->cv[p][Z];

      nbr[X] = coords[X] + h->cv[p][X];
      nbr[Y] = coords[Y] + h->cv[p][Y];
      nbr[Z] = coords[Z] + h->cv[p][Z];
      out[X] = (!periods[X] && (nbr[X] < 0 || nbr[X] > dims[X] - 1));
      out[Y] = (!periods[Y] && (nbr[Y] < 0 || nbr[Y] > dims[Y] - 1));
      out[Z] = (!periods[Z] && (nbr[Z] < 0 || nbr[Z] > dims[Z] - 1));

      if (out[X] || out[Y] || out[Z]) {
	h->nbrrank[i][j][k] = MPI_PROC_NULL;
      }
      else {
	MPI_Cart_rank(h->comm, nbr, &h->nbrrank[i][j][k]);
      }
    }
    /* I must be in the middle */
    assert(h->nbrrank[1][1][1] == cs_cart_rank(field->cs));
  }

  /* Set out limits for send and recv regions. */

  cs_nlocal(field->cs, nlocal);
  cs_nhalo(field->cs, &nhalo);

  for (int p = 1; p < h->nvel; p++) {

    int8_t cx = h->cv[p][X];
    int8_t cy = h->cv[p][Y];
    int8_t cz = h->cv[p][Z];

    cs_limits_t send = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    cs_limits_t recv = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};

    if (cx == -1) send.imax = nhalo;
    if (cx == +1) send.imin = send.imax - (nhalo - 1);
    if (cy == -1) send.jmax = nhalo;
    if (cy == +1) send.jmin = send.jmax - (nhalo - 1);
    if (cz == -1) send.kmax = nhalo;
    if (cz == +1) send.kmin = send.kmax - (nhalo - 1);

    /* For recv, direction is reversed cf. send */
    if (cx == +1) { recv.imin = 1 - nhalo;     recv.imax = 0;}
    if (cx == -1) { recv.imin = recv.imax + 1; recv.imax = recv.imax + nhalo;}
    if (cy == +1) { recv.jmin = 1 - nhalo;     recv.jmax = 0;}
    if (cy == -1) { recv.jmin = recv.jmax + 1; recv.jmax = recv.jmax + nhalo;}
    if (cz == +1) { recv.kmin = 1 - nhalo;     recv.kmax = 0;}
    if (cz == -1) { recv.kmin = recv.kmax + 1; recv.kmax = recv.kmax + nhalo;}

    h->slim[p] = send;
    h->rlim[p] = recv;
  }

  /* Message count and buffers */
  h->max_buf_len = 0;
  for (int p = 1; p < h->nvel; p++) {

    int scount = field->nf*field_halo_size(h->slim[p]);
    int rcount = field->nf*field_halo_size(h->rlim[p]);
    if (scount > h->max_buf_len) {
      h->max_buf_len = scount;
    }
    h->send[p] = (double *) calloc(scount, sizeof(double));
    h->recv[p] = (double *) calloc(rcount, sizeof(double));
    assert(h->send[p]);
    assert(h->recv[p]);
  }
  tdpStreamCreate(&h->stream);

  // Device
  int ndevice = 0;
  tdpGetDeviceCount(&ndevice);
  if (ndevice == 0) {
    h->target = h;
  } else {
    tdpMalloc((void **) &h->target, sizeof(field_halo_t));
    tdpMemcpy(h->target, h, sizeof(field_halo_t), tdpMemcpyHostToDevice);
    for (int p = 1; p < h->nvel; p++) {
      int scount = field->nf*field_halo_size(h->slim[p]);
      int rcount = field->nf*field_halo_size(h->rlim[p]);
      tdpMalloc((void**) &h->send_d[p], scount * sizeof(double));
      tdpMalloc((void**) &h->recv_d[p], rcount * sizeof(double));
    }
    tdpMemcpy(h->target->send, h->send_d, 27 * sizeof(double *), tdpMemcpyHostToDevice);
    tdpMemcpy(h->target->recv, h->recv_d, 27 * sizeof(double *), tdpMemcpyHostToDevice);
#if defined(__NVCC__) 
    create_send_graph(field, h);
    create_recv_graph(field, h);
#endif
  }
  return 0;
}

/*****************************************************************************
 *
 *  field_halo_post
 *
 *****************************************************************************/

int field_halo_post(const field_t * field, field_halo_t * h) {

  const int tagbase = 2022;

  assert(field);
  assert(h);

  /* Post recvs */

  TIMER_start(TIMER_FIELD_HALO_IRECV);

  h->request[0] = MPI_REQUEST_NULL;

  for (int ireq = 1; ireq < h->nvel; ireq++) {

    int i = 1 + h->cv[h->nvel - ireq][X];
    int j = 1 + h->cv[h->nvel - ireq][Y];
    int k = 1 + h->cv[h->nvel - ireq][Z];
    int mcount = field->nf*field_halo_size(h->rlim[ireq]);

    h->request[ireq] = MPI_REQUEST_NULL;

    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) continue; //mcount = 0;

    MPI_Irecv(h->recv[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
	      tagbase + ireq, h->comm, h->request + ireq);
  }

  TIMER_stop(TIMER_FIELD_HALO_IRECV);

  /* Load send buffers; post sends */

  TIMER_start(TIMER_FIELD_HALO_PACK);

#if defined(__NVCC__)
  cudaGraphLaunch(h->send_graph->graphExec, h->stream);
  tdpStreamSynchronize(h->stream);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());
#else
  #pragma omp parallel
  {
    for (int ireq = 1; ireq < h->nvel; ireq++) {
      field_halo_enqueue_send(field, h, ireq);
    }
  }
#endif

  TIMER_stop(TIMER_FIELD_HALO_PACK);

  TIMER_start(TIMER_FIELD_HALO_ISEND);
// #ifndef __NVCC__
  h->request[27] = MPI_REQUEST_NULL;

  for (int ireq = 1; ireq < h->nvel; ireq++) {
    int i = 1 + h->cv[ireq][X];
    int j = 1 + h->cv[ireq][Y];
    int k = 1 + h->cv[ireq][Z];
    int mcount = field->nf*field_halo_size(h->slim[ireq]);

    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) continue;// mcount = 0;

    MPI_Isend(h->send[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
	      tagbase + ireq, h->comm, h->request + 27 + ireq);
  }
// #endif
  TIMER_stop(TIMER_FIELD_HALO_ISEND);

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_wait
 *
 *****************************************************************************/

int field_halo_wait(field_t * field, field_halo_t * h) {

  assert(field);
  assert(h);

  TIMER_start(TIMER_FIELD_HALO_WAITALL);

  MPI_Waitall(2*h->nvel, h->request, MPI_STATUSES_IGNORE);

  TIMER_stop(TIMER_FIELD_HALO_WAITALL);

  TIMER_start(TIMER_FIELD_HALO_UNPACK);

#if defined(__NVCC__)
  cudaGraphLaunch(h->recv_graph->graphExec, h->stream);
  tdpStreamSynchronize(h->stream);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());
#else
  #pragma omp parallel
  {
    for (int ireq = 1; ireq < h->nvel; ireq++) {
      field_halo_dequeue_recv(field, h, ireq);
    }
  }
#endif

  TIMER_stop(TIMER_FIELD_HALO_UNPACK);

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_info
 *
 *****************************************************************************/

int field_halo_info(const field_t * f) {

  assert(f);
  assert(f->pe);

  pe_t * pe = f->pe;
  const field_halo_t * h = &f->h;

  /* For each direction, send limits */

  pe_info(pe, "\n");
  pe_info(pe, "Field halo information at root: %s\n", f->name);
  pe_info(pe, "\n");
  pe_info(pe, "Send requests\n");
  pe_info(pe,
	  "Req (cx cy cz) imin imax jmin jmax kmin kmax     bytes\n");
  pe_info(pe,
	  "------------------------------------------------------\n");
  for (int ireq = 1; ireq < h->nvel; ireq++) {
    pe_info(pe, "%3d (%2d %2d %2d) %4d %4d %4d %4d %4d %4d %9ld\n", ireq,
	    h->cv[ireq][X], h->cv[ireq][Y], h->cv[ireq][Z],
	    h->slim[ireq].imin, h->slim[ireq].imax,
	    h->slim[ireq].jmin, h->slim[ireq].jmax,
	    h->slim[ireq].kmin, h->slim[ireq].kmax,
	    (size_t) f->nf*field_halo_size(h->slim[ireq])*sizeof(double));
  }

  /* Recv limits counts */
  pe_info(pe, "\n");
  pe_info(pe, "Receive requests\n");
  pe_info(pe,
	  "Req (cx cy cz) imin imax jmin jmax kmin kmax     bytes\n");
  pe_info(pe,
	  "------------------------------------------------------\n");
  for (int ireq = 1; ireq < h->nvel; ireq++) {
    pe_info(pe, "%3d (%2d %2d %2d) %4d %4d %4d %4d %4d %4d %9ld\n", ireq,
	    h->cv[ireq][X], h->cv[ireq][Y], h->cv[ireq][Z],
	    h->rlim[ireq].imin, h->rlim[ireq].imax,
	    h->rlim[ireq].jmin, h->rlim[ireq].jmax,
	    h->rlim[ireq].kmin, h->rlim[ireq].kmax,
	    (size_t) f->nf*field_halo_size(h->rlim[ireq])*sizeof(double));
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_free
 *
 *****************************************************************************/

int field_halo_free(field_halo_t * h) {

  assert(h);

  int ndevice = 0;
  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpMemcpy(h->send_d, h->target->send, 27 * sizeof(double *), tdpMemcpyDeviceToHost);
    tdpMemcpy(h->recv_d, h->target->recv, 27 * sizeof(double *), tdpMemcpyDeviceToHost);
    for (int p = 1; p < h->nvel; p++) {
      tdpFree(h->send_d[p]);
      tdpFree(h->recv_d[p]);
    }
    tdpFree(h->target);
  }

#if defined(__NVCC__)
  free(h->send_graph);
  free(h->recv_graph);
#endif

  for (int p = 1; p < h->nvel; p++) {
    free(h->send[p]);
    free(h->recv[p]);  
  }
  tdpStreamDestroy(h->stream);
  *h = (field_halo_t) {0};

  return 0;
}
