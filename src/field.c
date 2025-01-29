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
 *  (c) 2012-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Zhihong Zhai wrote the initial version of the graph API inplementaiton.
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"

#include "timer.h"
#include "util.h"
#include "field.h"

static int field_data_touch(field_t * field);
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
    pe_fatal(pe, "Internal error: invalid field options: %s\n", name);
  }

  obj = (field_t *) calloc(1, sizeof(field_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(obj) failed\n");

  obj->nf = opts->ndata;
  obj->name = name;

  obj->pe = pe;
  obj->cs = cs;
  pe_retain(pe);
  cs_retain(cs);

  obj->opts = *opts;

  field_init(obj, opts->nhcomm, le);
  field_halo_create(obj, &obj->h);

  /* I/O single record information */
  /* As the communicator creation is a relatively high overhead operation,
   * we only want to create the metadata objects once. They're here. */
  /* There should be a check on a valid i/o decomposition before this
   * point, but in preicpile we can fail here... */

  {
    io_element_t elasc = {.datatype = MPI_CHAR,
			  .datasize = sizeof(char),
			  .count    = 1 + 23*obj->opts.ndata,
			  .endian   = io_endianness()};
    io_element_t elbin = {.datatype = MPI_DOUBLE,
			  .datasize = sizeof(double),
			  .count    = obj->opts.ndata,
			  .endian   = io_endianness()};
    {
      /* Input metadata */
      int ifail = 0;
      io_element_t element = {0};
      if (opts->iodata.input.iorformat == IO_RECORD_ASCII)  element = elasc;
      if (opts->iodata.input.iorformat == IO_RECORD_BINARY) element = elbin;
      ifail = io_metadata_initialise(cs, &opts->iodata.input, &element,
				     &obj->iometadata_in);

      assert(ifail == 0);
      if (ifail != 0) pe_fatal(pe, "Field: Bad input i/o decomposition\n");

      /* Output metadata */
      if (opts->iodata.output.iorformat == IO_RECORD_ASCII)  element = elasc;
      if (opts->iodata.output.iorformat == IO_RECORD_BINARY) element = elbin;
      ifail = io_metadata_initialise(cs, &opts->iodata.output, &element,
				     &obj->iometadata_out);

      assert(ifail == 0);
      if (ifail != 0) pe_fatal(pe, "Field: Bad output i/o decomposition\n");
    }
  }

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

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice > 0) {
    tdpAssert( tdpMemcpy(&tmp, &obj->target->data, sizeof(double *),
			 tdpMemcpyDeviceToHost) );
    tdpAssert( tdpFree(tmp) );
    tdpAssert( tdpFree(obj->target) );
  }

  free(obj->data);

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

  if (nfsz < 1 || INT_MAX/nfsz < 1) {
    pe_info(obj->pe, "field_init: failure in int32_t indexing\n");
    return -1;
  }

  if (obj->opts.usefirsttouch) {

    obj->data = (double *) mem_aligned_malloc(MEM_PAGESIZE, nfsz*sizeof(double));
    if (obj->data == NULL) pe_fatal(obj->pe, "calloc(obj->data) failed\n");
    field_data_touch(obj);
  }
  else {
    obj->data = (double *) calloc(nfsz, sizeof(double));
    if (obj->data == NULL) pe_fatal(obj->pe, "calloc(obj->data) failed\n");
  }

  /* Allocate target copy of structure (or alias) */

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    cs_t * cstarget = NULL;
    lees_edw_t * letarget = NULL;
    tdpAssert( tdpMalloc((void **) &obj->target, sizeof(field_t)) );
    tdpAssert( tdpMalloc((void **) &tmp, nfsz*sizeof(double)) );
    tdpAssert( tdpMemcpy(&obj->target->data, &tmp, sizeof(double *),
			 tdpMemcpyHostToDevice) );

    cs_target(obj->cs, &cstarget);
    if (le) lees_edw_target(obj->le, &letarget);
    tdpAssert( tdpMemcpy(&obj->target->cs, &cstarget, sizeof(cs_t *),
			 tdpMemcpyHostToDevice) );
    tdpAssert( tdpMemcpy(&obj->target->le, &letarget, sizeof(lees_edw_t *),
			 tdpMemcpyHostToDevice) );
    field_memcpy(obj, tdpMemcpyHostToDevice);
  }

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

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {

    nfsz = (size_t) obj->nf*obj->nsites;
    tdpAssert( tdpMemcpy(&tmp, &obj->target->data, sizeof(double *),
			 tdpMemcpyDeviceToHost) );

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpAssert( tdpMemcpy(&obj->target->nf, &obj->nf, sizeof(int), flag) );
      tdpAssert( tdpMemcpy(&obj->target->nhcomm, &obj->nhcomm, sizeof(int), flag) );
      tdpAssert( tdpMemcpy(&obj->target->nsites, &obj->nsites, sizeof(int), flag) );
      tdpAssert( tdpMemcpy(tmp, obj->data, nfsz*sizeof(double), flag) );
      break;
    case tdpMemcpyDeviceToHost:
      tdpAssert( tdpMemcpy(obj->data, tmp, nfsz*sizeof(double), flag) );
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
 *  field_data_touch
 *
 *****************************************************************************/

__host__ void field_data_touch_kernel(cs_limits_t lim, field_t * f) {

  int nx = 1 + lim.imax - lim.imin;
  int ny = 1 + lim.jmax - lim.jmin;
  int nz = 1 + lim.kmax - lim.kmin;

  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  #pragma omp for nowait
  for (int ik = 0; ik < nx*ny*nz; ik++) {
    int ic = lim.imin + (ik       )/strx;
    int jc = lim.jmin + (ik % strx)/stry;
    int kc = lim.kmin + (ik % stry)/strz;
    int index = cs_index(f->cs, ic, jc, kc);
    for (int n = 0; n < f->nf; n++) {
      int laddr = addr_rank1(f->nsites, f->nf, index, n);
      f->data[laddr] = 0.0;
    }
  }

  return;
}

/*****************************************************************************
 *
 *  field_data_touch
 *
 *  Run only for OpenMP on the host.
 *
 *****************************************************************************/

__host__ int field_data_touch(field_t * field) {

  int nlocal[3] = {0};

  assert(field);

  cs_nlocal(field->cs, nlocal);

  {
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};

    #pragma omp parallel
    {
      field_data_touch_kernel(lim, field);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_halo
 *
 *****************************************************************************/

__host__ int field_halo(field_t * obj) {

  field_halo_swap(obj, obj->opts.haloscheme);

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_swap
 *
 *****************************************************************************/

__host__ int field_halo_swap(field_t * obj, field_halo_enum_t flag) {

  assert(obj);

  /* Everything is "OPENMP", a name which is historical ... */
  /* There may be a need for host (only) halo swaps
   * if the target (GPU) is active. But not at the moment.  */

  switch (flag) {
  case FIELD_HALO_HOST:
  case FIELD_HALO_TARGET:
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
 *  field_write_buf
 *
 *  Per-lattice site binary write.
 *
 *****************************************************************************/

int field_write_buf(field_t * field, int index, char * buf) {

  double array[NQAB] = {0};

  assert(field);

  field_scalar_array(field, index, array);
  memcpy(buf, array, field->nf*sizeof(double));

  return 0;
}

/*****************************************************************************
 *
 *  field_read_buf
 *
 *  Per lattice site read (binary)
 *
 *****************************************************************************/

int field_read_buf(field_t * field, int index, const char * buf) {

  double array[NQAB] = {0};

  assert(field);
  assert(buf);

  memcpy(array, buf, field->nf*sizeof(double));
  field_scalar_array_set(field, index, array);

  return 0;
}

/*****************************************************************************
 *
 *  field_write_buf_ascii
 *
 *  Per lattice site write (binary).
 *
 *****************************************************************************/

int field_write_buf_ascii(field_t * field, int index, char * buf) {

  const int nbyte = 23;

  double array[NQAB] = {0};
  int ifail = 0;

  assert(field);
  assert(buf);

  field_scalar_array(field, index, array);

  /* We will overwrite any `\0` coming from sprintf() */
  /* Use tmp with +1 to allow for the \0 */

  for (int n = 0; n < field->nf; n++) {
    char tmp[BUFSIZ] = {0};
    int np = snprintf(tmp, nbyte + 1, " %22.15e", array[n]);
    if (np != nbyte) ifail = 1;
    memcpy(buf + n*nbyte, tmp, nbyte*sizeof(char));
    if (n == field->nf - 1) {
      np = snprintf(tmp, 2, "\n");
      if (np != 1) ifail = 2;
      memcpy(buf + field->nf*nbyte, tmp, sizeof(char));
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  field_read_buf_ascii
 *
 *  Per lattice site read (ascii).
 *
 *****************************************************************************/

int field_read_buf_ascii(field_t * field, int index, const char * buf) {

  const int nbyte = 23;

  double array[NQAB] = {0};
  int ifail = 0;

  assert(field);
  assert(buf);

  for (int n = 0; n < field->nf; n++) {
    /* First, make sure we have a \0, before sscanf() */
    char tmp[BUFSIZ] = {0};
    memcpy(tmp, buf + n*nbyte, nbyte*sizeof(char));
    int nr = sscanf(tmp, "%le", array + n);
    if (nr != 1) ifail = 1;
  }

  field_scalar_array_set(field, index, array);

  return ifail;
}

/*****************************************************************************
 *
 *  field_io_aggr_pack
 *
 *  Aggregator for packing the field to io_aggr_buf_t.
 *
 *****************************************************************************/

int field_io_aggr_pack(field_t * field, io_aggregator_t * aggr) {

  assert(field);
  assert(aggr);
  assert(aggr->buf);

  #pragma omp parallel
  {
    int iasc = field->opts.iodata.output.iorformat == IO_RECORD_ASCII;
    int ibin = field->opts.iodata.output.iorformat == IO_RECORD_BINARY;
    assert(iasc ^ ibin); /* one or other */

    #pragma omp for
    for (int ib = 0; ib < cs_limits_size(aggr->lim); ib++) {
      int ic = cs_limits_ic(aggr->lim, ib);
      int jc = cs_limits_jc(aggr->lim, ib);
      int kc = cs_limits_kc(aggr->lim, ib);

      /* Read/write data for (ic,jc,kc) */
      int index = cs_index(field->cs, ic, jc, kc);
      int offset = ib*aggr->szelement;
      if (iasc) field_write_buf_ascii(field, index, aggr->buf + offset);
      if (ibin) field_write_buf(field, index, aggr->buf + offset);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_io_aggr_unpack
 *
 *  Aggregator for the upack (read) stage.
 *
 *****************************************************************************/

int field_io_aggr_unpack(field_t * field, const io_aggregator_t * aggr) {

  assert(field);
  assert(aggr);
  assert(aggr->buf);

  #pragma omp parallel
  {
    int iasc = field->opts.iodata.input.iorformat == IO_RECORD_ASCII;
    int ibin = field->opts.iodata.input.iorformat == IO_RECORD_BINARY;
    assert(iasc ^ ibin); /* one or other */

    #pragma omp for
    for (int ib = 0; ib < cs_limits_size(aggr->lim); ib++) {
      int ic = cs_limits_ic(aggr->lim, ib);
      int jc = cs_limits_jc(aggr->lim, ib);
      int kc = cs_limits_kc(aggr->lim, ib);

      /* Read data for (ic,jc,kc) */
      int index = cs_index(field->cs, ic, jc, kc);
      size_t offset = ib*aggr->szelement;
      if (iasc) field_read_buf_ascii(field, index, aggr->buf + offset);
      if (ibin) field_read_buf(field, index, aggr->buf + offset);
    }
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

/*****************************************************************************
 *
 *  field_halo_enqueue_send_kernel
 *
 *  As above, but this is a device target implementation.
 *
 *****************************************************************************/

__global__ void field_halo_enqueue_send_kernel(const field_t * field,
					       field_halo_t * h, int ireq) {
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

/*****************************************************************************
 *
 *  field_halo_dequeue_recv_kernel
 *
 *  As above, but a target implementation.
 *
 *****************************************************************************/

__global__ void field_halo_dequeue_recv_kernel(field_t * field,
					       const field_halo_t * h,
					       int ireq) {
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

/*****************************************************************************
 *
 *  field_halo_create
 *
 *  It's convenient to borrow the velocity notation from the lb for
 *  the communication directions.
 *
 *****************************************************************************/

#include "lb_d3q27.h"

int field_halo_create(const field_t * field, field_halo_t * h) {

  int nlocal[3] = {0};
  int nhalo = 0;
  int ndevice = 0;

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

  for (int p = 1; p < h->nvel; p++) {

    int scount = field->nf*field_halo_size(h->slim[p]);
    int rcount = field->nf*field_halo_size(h->rlim[p]);

    h->send[p] = (double *) calloc(scount, sizeof(double));
    h->recv[p] = (double *) calloc(rcount, sizeof(double));
    assert(h->send[p]);
    assert(h->recv[p]);
  }

  for (int i = 0; i < 2 * h->nvel; i++) {
    h->request[i] = MPI_REQUEST_NULL;
  }

  /* Device */

  tdpAssert( tdpGetDeviceCount(&ndevice) );
  tdpAssert( tdpStreamCreate(&h->stream) );

  if (ndevice == 0) {
    h->target = h;
  }
  else {
    tdpAssert( tdpMalloc((void **) &h->target, sizeof(field_halo_t)) );
    tdpAssert( tdpMemcpy(h->target, h, sizeof(field_halo_t),
			 tdpMemcpyHostToDevice) );

    for (int p = 1; p < h->nvel; p++) {
      int scount = field->nf*field_halo_size(h->slim[p]);
      int rcount = field->nf*field_halo_size(h->rlim[p]);
      tdpAssert( tdpMalloc((void**) &h->send_d[p], scount * sizeof(double)) );
      tdpAssert( tdpMalloc((void**) &h->recv_d[p], rcount * sizeof(double)) );
    }
    /* Slightly tricksy. Could use send_d and recv_d on target copy ...*/
    tdpAssert( tdpMemcpy(h->target->send, h->send_d, 27*sizeof(double *),
			 tdpMemcpyHostToDevice) );
    tdpAssert( tdpMemcpy(h->target->recv, h->recv_d, 27*sizeof(double *),
			 tdpMemcpyHostToDevice) );

    field_graph_halo_send_create(field, h);
    field_graph_halo_recv_create(field, h);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_post
 *
 *****************************************************************************/

int field_halo_post(const field_t * field, field_halo_t * h) {

  int ndevice = 0;
  const int tagbase = 2022;

  assert(field);
  assert(h);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  /* Post recvs */

  TIMER_start(TIMER_FIELD_HALO_IRECV);

  h->request[0] = MPI_REQUEST_NULL;

  for (int ireq = 1; ireq < h->nvel; ireq++) {

    int i = 1 + h->cv[h->nvel - ireq][X];
    int j = 1 + h->cv[h->nvel - ireq][Y];
    int k = 1 + h->cv[h->nvel - ireq][Z];
    int mcount = field->nf*field_halo_size(h->rlim[ireq]);
    double * buf = h->recv[ireq];
    if (have_gpu_aware_mpi_()) buf = h->recv_d[ireq];

    h->request[ireq] = MPI_REQUEST_NULL;

    /* Skip messages to self */
    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) continue;

    MPI_Irecv(buf, mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
	      tagbase + ireq, h->comm, h->request + ireq);
  }

  TIMER_stop(TIMER_FIELD_HALO_IRECV);

  /* Load send buffers; post sends */

  TIMER_start(TIMER_FIELD_HALO_PACK);

  if (ndevice) {
    tdpAssert( tdpGraphLaunch(h->gsend.exec, h->stream) );
    tdpAssert( tdpStreamSynchronize(h->stream) );
  }
  else {
    #pragma omp parallel
    {
      for (int ireq = 1; ireq < h->nvel; ireq++) {
        field_halo_enqueue_send(field, h, ireq);
      }
    }
  }

  TIMER_stop(TIMER_FIELD_HALO_PACK);

  TIMER_start(TIMER_FIELD_HALO_ISEND);

  h->request[27] = MPI_REQUEST_NULL;

  for (int ireq = 1; ireq < h->nvel; ireq++) {
    int i = 1 + h->cv[ireq][X];
    int j = 1 + h->cv[ireq][Y];
    int k = 1 + h->cv[ireq][Z];
    int mcount = field->nf*field_halo_size(h->slim[ireq]);
    double * buf = h->send[ireq];
    if (have_gpu_aware_mpi_()) buf = h->send_d[ireq];

    /* Skip messages to self ... */
    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) continue;

    MPI_Isend(buf, mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
	      tagbase + ireq, h->comm, h->request + 27 + ireq);
  }

  TIMER_stop(TIMER_FIELD_HALO_ISEND);

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_wait
 *
 *****************************************************************************/

int field_halo_wait(field_t * field, field_halo_t * h) {

  int ndevice = 0;

  assert(field);
  assert(h);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  TIMER_start(TIMER_FIELD_HALO_WAITALL);

  MPI_Waitall(2*h->nvel, h->request, MPI_STATUSES_IGNORE);

  TIMER_stop(TIMER_FIELD_HALO_WAITALL);

  TIMER_start(TIMER_FIELD_HALO_UNPACK);

  if (ndevice) {
    tdpAssert( tdpGraphLaunch(h->grecv.exec, h->stream) );
    tdpAssert( tdpStreamSynchronize(h->stream) );
  }
  else {
    /* Use explicit copies */
    #pragma omp parallel
    {
      for (int ireq = 1; ireq < h->nvel; ireq++) {
        field_halo_dequeue_recv(field, h, ireq);
      }
    }
  }

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

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice > 0) {
    tdpAssert( tdpMemcpy(h->send_d, h->target->send, 27*sizeof(double *),
			 tdpMemcpyDeviceToHost) );
    tdpAssert( tdpMemcpy(h->recv_d, h->target->recv, 27*sizeof(double *),
			 tdpMemcpyDeviceToHost) );
    for (int p = 1; p < h->nvel; p++) {
      tdpAssert( tdpFree(h->send_d[p]) );
      tdpAssert( tdpFree(h->recv_d[p]) );
    }
    tdpAssert( tdpFree(h->target) );
  }

  for (int p = 1; p < h->nvel; p++) {
    free(h->send[p]);
    free(h->recv[p]);
  }

  if (ndevice > 0) {
    tdpAssert( tdpGraphDestroy(h->gsend.graph) );
    tdpAssert( tdpGraphDestroy(h->grecv.graph) );
  }

  tdpAssert( tdpStreamDestroy(h->stream) );
  *h = (field_halo_t) {0};

  return 0;
}

/*****************************************************************************
 *
 *  field_io_write
 *
 *****************************************************************************/

int field_io_write(field_t * field, int timestep, io_event_t * event) {

  int ifail = 0;
  io_impl_t * io = NULL;
  char filename[BUFSIZ] = {0};
  const io_metadata_t * meta = &field->iometadata_out;

  /* Metadata */
  if (meta->iswriten == 0) {
    ifail = io_metadata_write(meta, field->name, event->extra_name,
			      event->extra_json);
    if (ifail == 0) field->iometadata_out.iswriten = 1;
  }

  io_subfile_name(&meta->subfile, field->name, timestep, filename, BUFSIZ);
  ifail = io_impl_create(meta, &io);
  assert(ifail == 0);

  if (ifail == 0) {
    io_event_record(event, IO_EVENT_AGGR);
    field_memcpy(field, tdpMemcpyDeviceToHost);
    field_io_aggr_pack(field, io->aggr);

    io_event_record(event, IO_EVENT_WRITE);
    io->impl->write(io, filename);

    if (meta->options.report) {
      pe_info(field->pe, "MPIIO wrote to %s\n", filename);
    }

    io->impl->free(&io);
    io_event_report_write(event, meta, field->name);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  field_io_read
 *
 *****************************************************************************/

int field_io_read(field_t * field, int timestep, io_event_t * event) {

  int ifail = 0;
  io_impl_t * io = NULL;
  char filename[BUFSIZ] = {0};
  const io_metadata_t * meta = &field->iometadata_in;

  io_subfile_name(&meta->subfile, field->name, timestep, filename, BUFSIZ);

  ifail = io_impl_create(meta, &io);
  assert(ifail == 0);

  if (ifail == 0) {
    io_event_record(event, IO_EVENT_READ);
    io->impl->read(io, filename);
    io_event_record(event, IO_EVENT_DISAGGR);
    field_io_aggr_unpack(field, io->aggr);
    io->impl->free(&io);

    if (meta->options.report) {
      pe_info(field->pe, "MPIIO read from %s\n", filename);
      io_event_report_read(event, meta, field->name);
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  field_graph_halo_send_create
 *
 *****************************************************************************/

int field_graph_halo_send_create(const field_t * field, field_halo_t * h) {

  assert(field);
  assert(h);

  tdpAssert( tdpGraphCreate(&h->gsend.graph, 0) );

  for (int ireq = 1; ireq < h->nvel; ireq++) {
    tdpGraphNode_t kernelNode;
    tdpKernelNodeParams kernelNodeParams = {0};
    void * kernelArgs[3] = {(void *) &field->target,
                            (void *) &h->target,
                            (void *) &ireq};
    kernelNodeParams.func = (void *) field_halo_enqueue_send_kernel;
    dim3 nblk;
    dim3 ntpb;
    int scount = field->nf*field_halo_size(h->slim[ireq]);

    kernel_launch_param(scount, &nblk, &ntpb);

    kernelNodeParams.gridDim        = nblk;
    kernelNodeParams.blockDim       = ntpb;
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams   = (void **) kernelArgs;
    kernelNodeParams.extra          = NULL;

    tdpAssert( tdpGraphAddKernelNode(&kernelNode, h->gsend.graph, NULL, 0,
				     &kernelNodeParams) );

    if (have_gpu_aware_mpi_()) {
      /* Don't need explicit device -> host copy */
    }
    else {
      /* We do need to add the memcpys to the graph definition
       * (except messages to self... ) */

      int i = 1 + h->cv[h->nvel - ireq][X];
      int j = 1 + h->cv[h->nvel - ireq][Y];
      int k = 1 + h->cv[h->nvel - ireq][Z];

      if (h->nbrrank[i][j][k] != h->nbrrank[1][1][1]) {
	tdpGraphNode_t memcpyNode;
        tdpMemcpy3DParms memcpyParams = {0};

	memcpyParams.srcArray = NULL;
	memcpyParams.srcPos   = make_tdpPos(0, 0, 0);
	memcpyParams.srcPtr   = make_tdpPitchedPtr(h->send_d[ireq],
						   sizeof(double)*scount,
						   scount, 1);
	memcpyParams.dstArray = NULL;
	memcpyParams.dstPos   = make_tdpPos(0, 0, 0);
	memcpyParams.dstPtr   = make_tdpPitchedPtr(h->send[ireq],
						   sizeof(double)*scount,
						   scount, 1);
	memcpyParams.extent   = make_tdpExtent(sizeof(double)*scount, 1, 1);
	memcpyParams.kind     = tdpMemcpyDeviceToHost;

	tdpAssert( tdpGraphAddMemcpyNode(&memcpyNode, h->gsend.graph,
					 &kernelNode, 1, &memcpyParams) );
      }
    }
  }

  tdpAssert( tdpGraphInstantiate(&h->gsend.exec, h->gsend.graph, 0) );

  return 0;
}

/*****************************************************************************
 *
 *  field_graph_halo_recv_create
 *
 *****************************************************************************/

int field_graph_halo_recv_create(const field_t * field, field_halo_t * h) {

  assert(field);
  assert(h);

  tdpAssert( tdpGraphCreate(&h->grecv.graph, 0) );

  for (int ireq = 1; ireq < h->nvel; ireq++) {
    int rcount = field->nf*field_halo_size(h->rlim[ireq]);
    tdpGraphNode_t memcpyNode = {0};

    if (have_gpu_aware_mpi_()) {
      /* Don't need explicit copies */
    }
    else {
      int i = 1 + h->cv[h->nvel - ireq][X];
      int j = 1 + h->cv[h->nvel - ireq][Y];
      int k = 1 + h->cv[h->nvel - ireq][Z];

      if (h->nbrrank[i][j][k] != h->nbrrank[1][1][1]) {
	tdpMemcpy3DParms memcpyParams = {0};

	memcpyParams.srcArray = NULL;
	memcpyParams.srcPos   = make_tdpPos(0, 0, 0);
	memcpyParams.srcPtr   = make_tdpPitchedPtr(h->recv[ireq],
						   sizeof(double)*rcount,
						   rcount, 1);
	memcpyParams.dstArray = NULL;
	memcpyParams.dstPos   = make_tdpPos(0, 0, 0);
	memcpyParams.dstPtr   = make_tdpPitchedPtr(h->recv_d[ireq],
						   sizeof(double)*rcount,
						   rcount, 1);
	memcpyParams.extent   = make_tdpExtent(sizeof(double)*rcount, 1, 1);
	memcpyParams.kind     = tdpMemcpyHostToDevice;

	tdpAssert( tdpGraphAddMemcpyNode(&memcpyNode, h->grecv.graph, NULL,
					 0, &memcpyParams) );
      }
    }

    /* Always need the dis-aggregateion kernel */

    dim3 nblk;
    dim3 ntpb;
    tdpGraphNode_t node;
    tdpKernelNodeParams kernelNodeParams = {0};
    void * kernelArgs[3] = {(void *) &field->target,
                            (void *) &h->target,
                            (void *) &ireq};
    kernelNodeParams.func = (void *) field_halo_dequeue_recv_kernel;

    kernel_launch_param(rcount, &nblk, &ntpb);

    kernelNodeParams.gridDim        = nblk;
    kernelNodeParams.blockDim       = ntpb;
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams   = (void **) kernelArgs;
    kernelNodeParams.extra          = NULL;

    if (have_gpu_aware_mpi_()) {
      tdpAssert( tdpGraphAddKernelNode(&node, h->grecv.graph, NULL,
				       0, &kernelNodeParams) );
    }
    else {
      int i = 1 + h->cv[h->nvel - ireq][X];
      int j = 1 + h->cv[h->nvel - ireq][Y];
      int k = 1 + h->cv[h->nvel - ireq][Z];
      if (h->nbrrank[i][j][k] != h->nbrrank[1][1][1]) {
	tdpAssert( tdpGraphAddKernelNode(&node, h->grecv.graph, &memcpyNode,
					 1, &kernelNodeParams) );
      }
      else {
	tdpAssert( tdpGraphAddKernelNode(&node, h->grecv.graph, NULL, 0,
					 &kernelNodeParams) );
      }
    }
  }

  tdpAssert( tdpGraphInstantiate(&h->grecv.exec, h->grecv.graph, 0) );

  return 0;
}
