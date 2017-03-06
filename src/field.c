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
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2016 The University of Edinburgh
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
#include "util.h"
#include "field_s.h"

static int field_write(FILE * fp, int index, void * self);
static int field_write_ascii(FILE * fp, int index, void * self);
static int field_read(FILE * fp, int index, void * self);
static int field_read_ascii(FILE * fp, int index, void * self);

static int field_leesedwards_parallel(field_t * obj);

/*****************************************************************************
 *
 *  field_create
 *
 *  Allocation of data space is deferred until phi_init(), at which point
 *  a via coordinate system object should be available.
 *
 *  This just sets the type of field; often order parameter, e.g.,:
 *     nf = 1 for scalar "phi"
 *     nf = 3 for vector "p"
 *     nf = 5 for tensor "q" (compressed traceless, symmetric)
 *
 *****************************************************************************/

__host__ int field_create(pe_t * pe, cs_t * cs, int nf, const char * name,
			  field_t ** pobj) {

  field_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(nf > 0);
  assert(pobj);

  obj = (field_t *) calloc(1, sizeof(field_t));
  if (obj == NULL) fatal("calloc(obj) failed\n");

  obj->nf = nf;

  obj->name = (char *) calloc(strlen(name) + 1, sizeof(char));
  if (obj->name == NULL) fatal("calloc(name) failed\n");

  assert(strlen(name) < BUFSIZ);
  strncpy(obj->name, name, strlen(name));
  obj->name[strlen(name)] = '\0';

  obj->pe = pe;
  obj->cs = cs;
  pe_retain(pe);
  cs_retain(cs);

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

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    copyFromTarget(&tmp, &obj->target->data, sizeof(double *));
    targetFree(tmp);
    targetFree(obj->target);
  }

  if (obj->data) free(obj->data);
  if (obj->name) free(obj->name);
  if (obj->halo) halo_swap_free(obj->halo);
  if (obj->info) io_info_free(obj->info);
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
  double * tmp;

  assert(obj);
  assert(obj->data == NULL);

  cs_nsites(obj->cs, &nsites);
  if (le) lees_edw_nsites(le, &nsites);

  obj->le = le;
  obj->nhcomm = nhcomm;
  obj->nsites = nsites;
  obj->data = (double *) calloc(obj->nf*nsites, sizeof(double));
  if (obj->data == NULL) pe_fatal(obj->pe, "calloc(obj->data) failed\n");

  /* Allocate target copy of structure (or alias) */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    cs_t * cstarget = NULL;
    lees_edw_t * letarget = NULL;
    targetCalloc((void **) &obj->target, sizeof(field_t));
    targetCalloc((void **) &tmp, obj->nf*nsites*sizeof(double));
    copyToTarget(&obj->target->data, &tmp, sizeof(double *));

    cs_target(obj->cs, &cstarget);
    if (le) lees_edw_target(obj->le, &letarget);
    copyToTarget(&obj->target->cs, &cstarget, sizeof(cs_t *));
    copyToTarget(&obj->target->le, &letarget, sizeof(lees_edw_t *));
    field_memcpy(obj, cudaMemcpyHostToDevice);
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

__host__ int field_memcpy(field_t * obj, int flag) {

  int ndevice;
  double * tmp;

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {

    copyFromTarget(&tmp, &obj->target->data, sizeof(double *));

    switch (flag) {
    case cudaMemcpyHostToDevice:
      copyToTarget(&obj->target->nf, &obj->nf, sizeof(int));
      copyToTarget(&obj->target->nhcomm, &obj->nhcomm, sizeof(int));
      copyToTarget(&obj->target->nsites, &obj->nsites, sizeof(int));
      copyToTarget(tmp, obj->data, obj->nf*obj->nsites*sizeof(double));
      break;
    case cudaMemcpyDeviceToHost:
      copyFromTarget(obj->data, tmp, obj->nf*obj->nsites*sizeof(double));
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

  io_info_arg_t args;

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
  io_info_set_bytesize(obj->info, obj->nf*sizeof(double));

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
    field_memcpy(obj, cudaMemcpyDeviceToHost);
    field_halo_swap(obj, FIELD_HALO_HOST);
    field_memcpy(obj, cudaMemcpyHostToDevice);
  }
  else {
    /* Default to ... */
    field_halo_swap(obj, FIELD_HALO_TARGET);
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
    copyFromTarget(&data, &obj->target->data, sizeof(double *));
    halo_swap_packed(obj->halo, data);
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

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  const double r6 = (1.0/6.0);

  int jdy;               /* Integral part of displacement */
  int j0, j1, j2, j3;    /* j values in real system to interpolate between */

  assert(obj);
  assert(obj->data);

  if (obj->le == NULL) return 0;

  if (cart_size(Y) > 1) {
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
      dy = fmod(dy, L(Y));
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

  double dy;               /* Displacement for current ic->ib pair */
  double fr;               /* Fractional displacement */
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
  cs_nhalo(obj->cs, &nhalo);
  cs_nlocal(obj->cs, nlocal);
  cs_nlocal_offset(obj->cs, noffset);
  ib0 = nlocal[X] + nhalo + 1;

  lees_edw_comm(obj->le, &le_comm);
  lees_edw_nxbuffer(obj->le, &nxbuffer);

  /* Allocate buffer space */

  nsend = nf*nlocal[Y]*(nlocal[Z] + 2*nhalo);
  nrecv = nf*(nlocal[Y] + 2*nhalo + 3)*(nlocal[Z] + 2*nhalo);

  sendbuf = (double *) malloc(nsend*sizeof(double));
  recvbuf = (double *) malloc(nrecv*sizeof(double));

  if (sendbuf == NULL) fatal("malloc(sendbuf) failed\n");
  if (recvbuf == NULL) fatal("malloc(recvbuf) failed\n");

  /* One round of communication for each buffer plane */

  for (ib = 0; ib < nxbuffer; ib++) {

    ic = lees_edw_ibuff_to_real(obj->le, ib);
    kc = 1 - nhalo;

    /* Work out the displacement-dependent quantities */

    lees_edw_buffer_dy(obj->le, ib, 0.0, &dy);
    dy = fmod(dy, L(Y));
    jdy = floor(dy);
    fr  = 1.0 - (dy - jdy);
    /* In the real system the first point we require is
     * j1 = jc - jdy - 3
     * with jc = noffset[Y] + 1 - nhalo in the global coordinates.
     * Modular arithmetic ensures 1 <= j1 <= N_total(Y) */

    jc = noffset[Y] + 1 - nhalo;
    j1 = 1 + (jc - jdy - 3 + 2*N_total(Y)) % N_total(Y);
    assert(j1 >= 1);
    assert(j1 <= N_total(Y));

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
  if (n != obj->nf) fatal("fread(field) failed at index %d", index);

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
    if (nread != 1) fatal("fscanf(field) failed at index %d\n", index);
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
  if (n != obj->nf) fatal("fwrite(field) failed at index %d\n", index);

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
    nwrite = fprintf(fp, "%23.15e ", array[n]);
    if (nwrite != 24) 
	fatal("fprintf(%s) failed at index %d\n", obj->name, index);
  }

  nwrite = fprintf(fp, "\n");
  if (nwrite != 1) fatal("fprintf(%s) failed at index %d\n", obj->name, index);

  return 0;
}
