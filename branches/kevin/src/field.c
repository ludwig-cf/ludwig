/*****************************************************************************
 *
 *  field.c
 *
 *  Data layout.
 *
 *  We store (conceptually) data[nsites][nf], so that components of
 *  a field are stored contiguously. A flattened 1d addressing is
 *  used so that data[index][n] becomes data[nf*index + 1*n].
 *
 *  To reverse this, use data[nf][nsites], so that addressing
 *  becomes data[n][index] -> data[1*index + nsites*n].
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coords_field.h"
#include "util.h"
#include "control.h" /* Can we move get_step() to LE please? */
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
 *  Allocation of data space is deferred until field_init(), at which point
 *  a coordinate system object should be available.
 *
 *  This just sets the type of field; often order parameter, e.g.,:
 *     nf = 1 for scalar "phi"
 *     nf = 3 for vector "p"
 *     nf = 5 for tensor "q" (compressed traceless, symmetric)
 *
 *****************************************************************************/

int field_create(coords_t * cs, int nf, const char * name, field_t ** pobj) {

  field_t * obj = NULL;

  assert(cs);
  assert(nf > 0);
  assert(pobj);

  obj = (field_t*) calloc(1, sizeof(field_t));
  if (obj == NULL) fatal("calloc(obj) failed\n");

  obj->nf = nf;
  obj->halo[0] = MPI_DATATYPE_NULL;
  obj->halo[1] = MPI_DATATYPE_NULL;
  obj->halo[2] = MPI_DATATYPE_NULL;

  obj->name = (char*) calloc(strlen(name) + 1, sizeof(char));
  if (obj->name == NULL) fatal("calloc(name) failed\n");

  assert(strlen(name) < BUFSIZ);
  strncpy(obj->name, name, strlen(name));
  obj->name[strlen(name)] = '\0';

  obj->cs = cs;
  coords_retain(cs);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  field_free
 *
 *****************************************************************************/

int field_free(field_t * obj) {

  assert(obj);

  if (obj->data) free(obj->data);
  if (obj->t_data) targetFree(obj->t_data);

  if (obj->siteMask) free(obj->siteMask);
  if (obj->t_siteMask) targetFree(obj->t_siteMask);

  if (obj->halo[0] != MPI_DATATYPE_NULL) MPI_Type_free(&obj->halo[0]);
  if (obj->halo[1] != MPI_DATATYPE_NULL) MPI_Type_free(&obj->halo[1]);
  if (obj->halo[2] != MPI_DATATYPE_NULL) MPI_Type_free(&obj->halo[2]);

  if (obj->name) free(obj->name);
  if (obj->info) io_info_free(obj->info);
  if (obj->le) le_free(obj->le);

  coords_free(obj->cs);
  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  field_init
 *
 *  Initialise the lattice data, MPI halo information.
 *
 *****************************************************************************/

int field_init(field_t * obj, int nhcomm, le_t * le) {

  assert(obj);
  assert(obj->data == NULL);

  /* assert(nhcomm <= coords_nhalo());*/

  coords_nsites(obj->cs, &obj->nsites);
  if (le) le_nsites(le, &obj->nsites);

  obj->data = (double*) calloc(obj->nf*obj->nsites, sizeof(double));

  if (obj->data == NULL) fatal("calloc(obj->data) failed\n");

  /* allocate target copy */
  targetCalloc((void **) &obj->t_data, obj->nf*obj->nsites*sizeof(double));

  /* allocate boolean lattice-shaped struture for masking*/
  
  obj->siteMask = (char  *) calloc(obj->nsites,sizeof(char));
  targetCalloc((void **) &obj->t_siteMask, obj->nsites*sizeof(char));


  /* MPI datatypes for halo */

  obj->nhcomm = nhcomm;
  coords_field_init_mpi_indexed(obj->cs, obj->nhcomm, obj->nf, MPI_DOUBLE,
				obj->halo);

  obj->le = le;
  if (le) le_retain(le);

  return 0;
}

/*****************************************************************************
 *
 *  field_nf
 *
 *****************************************************************************/

int field_nf(field_t * obj, int * nf) {

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

int field_init_io_info(field_t * obj, int grid[3], int form_in,
			     int form_out) {
  assert(obj);
  assert(grid);
  assert(obj->info == NULL);

  io_info_create_with_grid(obj->cs, grid, &obj->info);
  if (obj->info == NULL) fatal("io_info_create(field) failed\n");

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

int field_io_info(field_t * obj, io_info_t ** info) {

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

int field_halo(field_t * obj) {

  assert(obj);
  assert(obj->data);
  coords_field_halo(obj->cs, obj->nhcomm, obj->nf, obj->data, MPI_DOUBLE,
		    obj->halo);

  return 0;
}

/*****************************************************************************
 *
 *  field_leesedwards
 *
 *  Interpolate the phi field to take account of any local Lees Edwards
 *  boundaries.
 *
 *  If a non-NULL Lees Edwards coordinate system has not been provided,
 *  this will just return as if there are no planes.
 *
 *  The buffer region of obj->data[] is updated with the interpolated
 *  values.
 *
 *****************************************************************************/

int field_leesedwards(field_t * obj) {

  int nf;
  int nxb;
  int nhalo;
  int nlocal[3]; /* Local system size */
  int ib;        /* Index in buffer region */
  int ib0;       /* buffer region offset */
  int ic;        /* Index corresponding x location in real system */
  int jc, kc, n;
  int index, index0, index1, index2, index3;
  int cartsz[3];

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  double t;      /* Time */
  double ltot[3];

  const double r6 = (1.0/6.0);

  int jdy;               /* Integral part of displacement */
  int j0, j1, j2, j3;    /* j values in real system to interpolate between */

  assert(obj);
  assert(obj->data);
 
  if (obj->le == NULL) return 0;

  le_ltot(obj->le, ltot);
  le_nxbuffer(obj->le, &nxb);
  le_cartsz(obj->le, cartsz);

  if (cartsz[Y] > 1) {
    /* This has its own routine. */
    field_leesedwards_parallel(obj);
  }
  else {
    /* No messages are required... */

    field_nf(obj, &nf);
    le_nhalo(obj->le, &nhalo);
    le_nlocal(obj->le, nlocal);
    ib0 = nlocal[X] + nhalo + 1;

    /* -1.0 as zero required for first step; a 'feature' to
     * maintain the regression tests */
    t = 1.0*get_step() - 1.0;

    for (ib = 0; ib < nxb; ib++) {

      ic = le_index_buffer_to_real(obj->le, ib);
      le_buffer_displacement(obj->le, ib, t, &dy);
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
          index  = nf*le_site_index(obj->le, ib0 + ib, jc, kc);
          index0 = nf*le_site_index(obj->le, ic, j0, kc);
          index1 = nf*le_site_index(obj->le, ic, j1, kc);
          index2 = nf*le_site_index(obj->le, ic, j2, kc);
          index3 = nf*le_site_index(obj->le, ic, j3, kc);
          for (n = 0; n < nf; n++) {
            obj->data[index + n] =
              -  r6*fr*(fr-1.0)*(fr-2.0)*obj->data[index0 + n]
              + 0.5*(fr*fr-1.0)*(fr-2.0)*obj->data[index1 + n]
              - 0.5*fr*(fr+1.0)*(fr-2.0)*obj->data[index2 + n]
              +        r6*fr*(fr*fr-1.0)*obj->data[index3 + n];
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
  int ntotal[3];
  int nxb;
  int      nlocal[3];      /* Local system size */
  int      noffset[3];     /* Local starting offset */
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
  double t;                /* Time */
  double * buffer;         /* Interpolation buffer */
  double ltot[3];
  const double r6 = (1.0/6.0);

  int      nrank_s[3];     /* send ranks */
  int      nrank_r[3];     /* recv ranks */

  const int tag0 = 1256;
  const int tag1 = 1257;
  const int tag2 = 1258;

  MPI_Comm comm;
  MPI_Request request[6];
  MPI_Status  status[3];

  assert(obj);
  assert(obj->le);

  field_nf(obj, &nf);

  le_ltot(obj->le, ltot);
  le_nhalo(obj->le, &nhalo);
  le_ntotal(obj->le, ntotal);
  le_nlocal(obj->le, nlocal);
  le_nlocal_offset(obj->le, noffset);
  ib0 = nlocal[X] + nhalo + 1;

  le_nxbuffer(obj->le, &nxb);
  le_comm(obj->le, &comm);

  /* Allocate the temporary buffer */

  n = nf*(nlocal[Y] + 2*nhalo + 3)*(nlocal[Z] + 2*nhalo);

  buffer = (double *) malloc(n*sizeof(double));
  if (buffer == NULL) fatal("malloc(buffer) failed\n");
  /* -1.0 as zero required for fisrt step; this is a 'feature'
   * to ensure the regression tests stay te same */

  t = 1.0*get_step() - 1.0;

  /* One round of communication for each buffer plane */

  for (ib = 0; ib < nxb; ib++) {

    ic = le_index_buffer_to_real(obj->le, ib);
    kc = 1 - nhalo;

    /* Work out the displacement-dependent quantities */

    le_buffer_displacement(obj->le, ib, t, &dy);
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

    le_jstart_to_mpi_ranks(obj->le, j1, nrank_s, nrank_r);

    /* Local quantities: j2 is the position of j1 in local coordinates.
     * The three sections to send/receive are organised as follows:
     * jc is the number of j points in each case, while n is the
     * total number of data items. Note that n3 can be zero. */

    j2 = 1 + (j1 - 1) % nlocal[Y];
    assert(j2 >= 1);
    assert(j2 <= nlocal[Y]);

    jc = nlocal[Y] - j2 + 1;
    n1 = nf*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(buffer, n1, MPI_DOUBLE, nrank_r[0], tag0, comm, request);

    jc = imin(nlocal[Y], j2 + 2*nhalo + 2);
    n2 = nf*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(buffer + n1, n2, MPI_DOUBLE, nrank_r[1], tag1, comm,
              request + 1);

    jc = imax(0, j2 - nlocal[Y] + 2*nhalo + 2);
    n3 = nf*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(buffer + n1 + n2, n3, MPI_DOUBLE, nrank_r[2], tag2, comm,
              request + 2);

    /* Post sends and wait for receives. */

    index = nf*le_site_index(obj->le, ic, j2, kc);
    MPI_Issend(&obj->data[index], n1, MPI_DOUBLE, nrank_s[0], tag0, comm,
               request + 3);
    index = nf*le_site_index(obj->le, ic, 1, kc);
    MPI_Issend(&obj->data[index], n2, MPI_DOUBLE, nrank_s[1], tag1, comm,
               request + 4);
    MPI_Issend(&obj->data[index], n3, MPI_DOUBLE, nrank_s[2], tag2, comm,
               request + 5);

    MPI_Waitall(3, request, status);


    /* Perform the actual interpolation from temporary buffer to
     * phi_site[] buffer region. */

    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

      /* Note that the linear interpolation here would be
       * (1.0-fr)*buffer(j1, k, n) + fr*buffer(j2, k, n)
       * This is again Lagrange four point. */

      j0 = (jc + nhalo - 1    )*(nlocal[Z] + 2*nhalo);
      j1 = (jc + nhalo - 1 + 1)*(nlocal[Z] + 2*nhalo);
      j2 = (jc + nhalo - 1 + 2)*(nlocal[Z] + 2*nhalo);
      j3 = (jc + nhalo - 1 + 3)*(nlocal[Z] + 2*nhalo);

      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
        index = nf*le_site_index(obj->le, ib0 + ib, jc, kc);
        for (n = 0; n < nf; n++) {
          obj->data[index + n] =
            -  r6*fr*(fr-1.0)*(fr-2.0)*buffer[nf*(j0 + kc+nhalo-1) + n]
            + 0.5*(fr*fr-1.0)*(fr-2.0)*buffer[nf*(j1 + kc+nhalo-1) + n]
            - 0.5*fr*(fr+1.0)*(fr-2.0)*buffer[nf*(j2 + kc+nhalo-1) + n]
            +        r6*fr*(fr*fr-1.0)*buffer[nf*(j3 + kc+nhalo-1) + n];
        }
      }
    }

    /* Clean up the sends, and move to next buffer location. */

    MPI_Waitall(3, request + 3, status);
  }

  free(buffer);

  return 0;
}


/*****************************************************************************
 *
 *  field_scalar
 *
 *****************************************************************************/

int field_scalar(field_t * obj, int index, double * phi) {

  assert(obj);
  assert(obj->nf == 1);
  assert(obj->data);
  assert(phi);

  *phi = obj->data[index];

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar_set
 *
 *****************************************************************************/

int field_scalar_set(field_t * obj, int index, double phi) {

  assert(obj);
  assert(obj->nf == 1);
  assert(obj->data);

  obj->data[index] = phi;

  return 0;
}

/*****************************************************************************
 *
 *  field_vector
 *
 *****************************************************************************/

int field_vector(field_t * obj, int index, double p[3]) {

  int ia;

  assert(obj);
  assert(obj->nf == 3);
  assert(obj->data);
  assert(p);

  for (ia = 0; ia < 3; ia++) {
    p[ia] = obj->data[3*index + ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_vector_set
 *
 *****************************************************************************/

int field_vector_set(field_t * obj, int index, const double p[3]) {

  int ia;

  assert(obj);
  assert(obj->nf == 3);
  assert(obj->data);
  assert(p);

  for (ia = 0; ia < 3; ia++) {
    obj->data[3*index + ia] = p[ia];
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

int field_tensor(field_t * obj, int index, double q[3][3]) {

  assert(obj);
  assert(obj->nf == NQAB);
  assert(obj->data);
  assert(q);

  q[X][X] = obj->data[NQAB*index + XX];
  q[X][Y] = obj->data[NQAB*index + XY];
  q[X][Z] = obj->data[NQAB*index + XZ];
  q[Y][X] = q[X][Y];
  q[Y][Y] = obj->data[NQAB*index + YY];
  q[Y][Z] = obj->data[NQAB*index + YZ];
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

int field_tensor_set(field_t * obj, int index, double q[3][3]) {

  assert(obj);
  assert(obj->nf == NQAB);
  assert(obj->data);
  assert(q);

  obj->data[NQAB*index + XX] = q[X][X];
  obj->data[NQAB*index + XY] = q[X][Y];
  obj->data[NQAB*index + XZ] = q[X][Z];
  obj->data[NQAB*index + YY] = q[Y][Y];
  obj->data[NQAB*index + YZ] = q[Y][Z];

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

int field_scalar_array(field_t * obj, int index, double * array) {

  int n;

  assert(obj);
  assert(obj->data);
  assert(array);

  for (n = 0; n < obj->nf; n++) {
    array[n] = obj->data[obj->nf*index + n];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar_array_set
 *
 *****************************************************************************/

int field_scalar_array_set(field_t * obj, int index, const double * array) {

  int n;

  assert(obj);
  assert(obj->data);
  assert(array);

  for (n = 0; n < obj->nf; n++) {
    obj->data[obj->nf*index + n] = array[n];
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
  field_t * obj = (field_t*) self;

  assert(fp);
  assert(obj);

  n = fread(&obj->data[obj->nf*index], sizeof(double), obj->nf, fp);
  if (n != obj->nf) fatal("fread(field) failed at index %d", index);

  return 0;
}

/*****************************************************************************
 *
 *  field_read_ascii
 *
 *****************************************************************************/

static int field_read_ascii(FILE * fp, int index, void * self) {

  int n, nread;
  field_t * obj =  (field_t*) self;

  assert(fp);
  assert(obj);

  for (n = 0; n < obj->nf; n++) {
    nread = fscanf(fp, "%le", obj->data + obj->nf*index + n);
    if (nread != 1) fatal("fscanf(field) failed at index %d\n", index);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_write
 *
 *****************************************************************************/

static int field_write(FILE * fp, int index, void * self) {

  int n;
  field_t * obj =  (field_t*) self;

  assert(fp);
  assert(obj);

  n = fwrite(&obj->data[obj->nf*index], sizeof(double), obj->nf, fp);
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
  field_t * obj =  (field_t*) self;

  assert(fp);
  assert(obj);

  for (n = 0; n < obj->nf; n++) {
    nwrite = fprintf(fp, "%22.15e ", obj->data[obj->nf*index + n]);
    if (nwrite != 23) fatal("fprintf(%s) failed at index %d\n", obj->name,
			    index);
  }

  nwrite = fprintf(fp, "\n");
  if (nwrite != 1) fatal("fprintf(%s) failed at index %d\n", obj->name, index);

  return 0;
}
