/*****************************************************************************
 *
 *  phi.c
 *
 *  Scalar, vector, tensor, order parameter.
 *
 *  $Id: phi.c,v 1.13 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "control.h"
#include "io_harness.h"
#include "leesedwards.h"
#include "util.h"
#include "phi.h"

struct io_info_t * io_info_phi;

double * phi_site;

static int nop_ = 0;                    /* Number of order parameter fields */
static int initialised_ = 0;
static int phi_finite_difference_ = 0;  /* Default is LB for order parameter */
static MPI_Datatype phi_xy_t_;
static MPI_Datatype phi_xz_t_;
static MPI_Datatype phi_yz_t_;

static void phi_init_mpi(void);
static void phi_init_io(void);
static int  phi_read(FILE *, const int, const int, const int);
static int  phi_write(FILE *, const int, const int, const int);
static int  phi_read_ascii(FILE *, const int, const int, const int);
static int  phi_write_ascii(FILE *, const int, const int, const int);
static void phi_leesedwards_parallel(void);


/****************************************************************************
 *
 *  phi_init
 *
 *  Allocate memory for the order parameter.
 *
 *  Space for buffers to hold Lees Edwards interpolated quantities
 *  is added to the main array.
 *
 ****************************************************************************/

void phi_init() {

  int nhalo;
  int nsites;
  int nbuffer;
  int nlocal[3];

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  nbuffer = le_get_nxbuffer();

  nsites = (nlocal[X]+2*nhalo + nbuffer)
    *(nlocal[Y]+2*nhalo)*(nlocal[Z]+2*nhalo);

  phi_site = (double *) calloc(nop_*nsites, sizeof(double));
  if (phi_site == NULL) fatal("calloc(phi) failed\n");

  phi_init_mpi();
  phi_init_io();
  initialised_ = 1;

  return;
}

/*****************************************************************************
 *
 *  phi_init_mpi
 *
 *****************************************************************************/

static void phi_init_mpi() {

  int nhalo;
  int nlocal[3], nh[3];

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nh[X] = nlocal[X] + 2*nhalo;
  nh[Y] = nlocal[Y] + 2*nhalo;
  nh[Z] = nlocal[Z] + 2*nhalo;

  /* YZ planes in the X direction */
  MPI_Type_vector(1, nh[Y]*nh[Z]*nhalo*nop_, 1, MPI_DOUBLE, &phi_yz_t_);
  MPI_Type_commit(&phi_yz_t_);

  /* XZ planes in the Y direction */
  MPI_Type_vector(nh[X], nh[Z]*nhalo*nop_, nh[Y]*nh[Z]*nop_, MPI_DOUBLE,
		  &phi_xz_t_);
  MPI_Type_commit(&phi_xz_t_);

  /* XY planes in Z direction */
  MPI_Type_vector(nh[X]*nh[Y], nhalo*nop_, nh[Z]*nop_, MPI_DOUBLE,
		  &phi_xy_t_);
  MPI_Type_commit(&phi_xy_t_);

  return;
}

/*****************************************************************************
 *
 *  phi_init_io
 *
 *****************************************************************************/

static void phi_init_io() {

  /* Take a default I/O struct */
  io_info_phi = io_info_create();

  io_info_set_name(io_info_phi, "Compositional order parameter");
  io_info_set_read_binary(io_info_phi, phi_read);
  io_info_set_write_binary(io_info_phi, phi_write);
  io_info_set_read_ascii(io_info_phi, phi_read_ascii);
  io_info_set_write_ascii(io_info_phi, phi_write_ascii);
  io_info_set_bytesize(io_info_phi, nop_*sizeof(double));

  io_info_set_format_binary(io_info_phi);
  io_write_metadata("phi", io_info_phi);

  return;
}

/*****************************************************************************
 *
 *  phi_finish
 *
 *****************************************************************************/

void phi_finish() {

  io_info_destroy(io_info_phi);
  MPI_Type_free(&phi_xy_t_);
  MPI_Type_free(&phi_xz_t_);
  MPI_Type_free(&phi_yz_t_);

  free(phi_site);

  initialised_ = 0;

  return;
}

/****************************************************************************
 *
 *  phi_nop
 *
 *  Return the number of order parameters.
 *
 ****************************************************************************/

int phi_nop(void) {

  return nop_;
}

/*****************************************************************************
 *
 *  phi_nop_set
 *
 *****************************************************************************/

void phi_nop_set(const int n) {

  assert(initialised_ == 0);
  assert(n >= 1);
  nop_ = n;

  return;
}

/*****************************************************************************
 *
 *  phi_halo
 *
 *****************************************************************************/

void phi_halo() {

  int nhalo;
  int nlocal[3];
  int ic, jc, kc, ihalo, ireal, nh, n;
  int back, forw;
  const int btag = 2061;
  const int ftag = 2062;
  MPI_Comm comm = cart_comm();
  MPI_Request request[4];
  MPI_Status status[4];

  assert(initialised_);
  
  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  /* YZ planes in the X direction */

  if (cart_size(X) == 1) {
    for (nh = 0; nh < nhalo; nh++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1 ; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nop_; n++) {
	    phi_site[nop_*le_site_index(0-nh, jc,kc) + n]
	      = phi_site[nop_*le_site_index(nlocal[X]-nh, jc, kc) + n];
	    phi_site[nop_*le_site_index(nlocal[X]+1+nh, jc,kc) + n]
	      = phi_site[nop_*le_site_index(1+nh, jc, kc) + n];
	  }
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, X);
    forw = cart_neighb(FORWARD, X);

    ihalo = nop_*le_site_index(nlocal[X] + 1, 1-nhalo, 1-nhalo);
    MPI_Irecv(phi_site + ihalo,  1, phi_yz_t_, forw, btag, comm, request);
    ihalo = nop_*le_site_index(1-nhalo, 1-nhalo, 1-nhalo);
    MPI_Irecv(phi_site + ihalo,  1, phi_yz_t_, back, ftag, comm, request+1);
    ireal = nop_*le_site_index(1, 1-nhalo, 1-nhalo);
    MPI_Issend(phi_site + ireal, 1, phi_yz_t_, back, btag, comm, request+2);
    ireal = nop_*le_site_index(nlocal[X] - nhalo + 1, 1-nhalo, 1-nhalo);
    MPI_Issend(phi_site + ireal, 1, phi_yz_t_, forw, ftag, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  /* XZ planes in the Y direction */

  if (cart_size(Y) == 1) {
    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1-nhalo; ic <= nlocal[X] + nhalo; ic++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nop_; n++) {
	    phi_site[nop_*le_site_index(ic,0-nh, kc) + n]
	      = phi_site[nop_*le_site_index(ic, nlocal[Y]-nh, kc) + n];
	    phi_site[nop_*le_site_index(ic,nlocal[Y]+1+nh, kc) + n]
	      = phi_site[nop_*le_site_index(ic, 1+nh, kc) + n];
	  }
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, Y);
    forw = cart_neighb(FORWARD, Y);

    ihalo = nop_*le_site_index(1-nhalo, nlocal[Y] + 1, 1-nhalo);
    MPI_Irecv(phi_site + ihalo,  1, phi_xz_t_, forw, btag, comm, request);
    ihalo = nop_*le_site_index(1-nhalo, 1-nhalo, 1-nhalo);
    MPI_Irecv(phi_site + ihalo,  1, phi_xz_t_, back, ftag, comm, request+1);
    ireal = nop_*le_site_index(1-nhalo, 1, 1-nhalo);
    MPI_Issend(phi_site + ireal, 1, phi_xz_t_, back, btag, comm, request+2);
    ireal = nop_*le_site_index(1-nhalo, nlocal[Y] - nhalo + 1, 1-nhalo);
    MPI_Issend(phi_site + ireal, 1, phi_xz_t_, forw, ftag, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  /* XY planes in the Z direction */

  if (cart_size(Z) == 1) {
    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
        for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
	  for (n = 0; n < nop_; n++) {
	    phi_site[nop_*le_site_index(ic,jc, 0-nh) + n]
	      = phi_site[nop_*le_site_index(ic, jc, nlocal[Z]-nh) + n];
	    phi_site[nop_*le_site_index(ic,jc, nlocal[Z]+1+nh) + n]
	      = phi_site[nop_*le_site_index(ic, jc, 1+nh) + n];
	  }
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, Z);
    forw = cart_neighb(FORWARD, Z);

    ihalo = nop_*le_site_index(1-nhalo, 1-nhalo, nlocal[Z] + 1);
    MPI_Irecv(phi_site + ihalo,  1, phi_xy_t_, forw, btag, comm, request);
    ihalo = nop_*le_site_index(1-nhalo, 1-nhalo, 1-nhalo);
    MPI_Irecv(phi_site + ihalo,  1, phi_xy_t_, back, ftag, comm, request+1);
    ireal = nop_*le_site_index(1-nhalo, 1-nhalo, 1);
    MPI_Issend(phi_site + ireal, 1, phi_xy_t_, back, btag, comm, request+2);
    ireal = nop_*le_site_index(1-nhalo, 1-nhalo, nlocal[Z] - nhalo + 1);
    MPI_Issend(phi_site + ireal, 1, phi_xy_t_, forw, ftag, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_get_phi_site
 *
 *****************************************************************************/

double phi_get_phi_site(const int index) {

  assert(initialised_);
  return phi_site[nop_*index];
}

/*****************************************************************************
 *
 *  phi_set_phi_site
 *
 *****************************************************************************/

void phi_set_phi_site(const int index, const double phi) {

  assert(initialised_);
  phi_site[nop_*index] = phi;
  return;
}

/*****************************************************************************
 *
 *  phi_read
 *
 *****************************************************************************/

static int phi_read(FILE * fp, const int ic, const int jc, const int kc) {

  int index, n;

  index = le_site_index(ic, jc, kc);
  n = fread(phi_site + nop_*index, sizeof(double), nop_, fp);

  if (n != nop_) fatal("fread(phi) failed at index %d", index);

  return n;
}

/*****************************************************************************
 *
 *  phi_write
 *
 *****************************************************************************/

static int phi_write(FILE * fp, const int ic, const int jc, const int kc) {

  int index, n;

  index = le_site_index(ic, jc, kc);
  n = fwrite(phi_site + nop_*index, sizeof(double), nop_, fp);

  if (n != nop_) fatal("fwrite(phi) failed at index %d\n", index);
 
  return n;
}

/*****************************************************************************
 *
 *  phi_read_ascii
 *
 *****************************************************************************/

static int phi_read_ascii(FILE * fp, const int ic, const int jc,
			  const int kc) {
  int index, n, nread;

  index = le_site_index(ic, jc, kc);

  for (n = 0; n < nop_; n++) {
    nread = fscanf(fp, "%le", phi_site + nop_*index + n);
    if (nread != 1) fatal("fscanf(phi) failed at index %d\n", index);
  }

  return n;
}

/*****************************************************************************
 *
 *  phi_write_ascii
 *
 ****************************************************************************/

static int phi_write_ascii(FILE * fp, const int ic, const int jc,
			   const int kc) {
  int index, n, nwrite;

  index = le_site_index(ic, jc, kc);

  for (n = 0; n < nop_; n++) {
    nwrite = fprintf(fp, "%22.15e ", phi_site[nop_*index + n]);
    if (nwrite != 23) fatal("fprintf(phi) failed at index %d\n", index);
  }

  nwrite = fprintf(fp, "\n");
  if (nwrite != 1) fatal("fprintf(phi) failed at index %d\n", index);

  return n;
}

/*****************************************************************************
 *
 *  phi_leesedwards_transformation
 *
 *  Interpolate the phi field to take account of any local
 *  Lees Edwards boundaries.
 *
 *  Effect:
 *    The buffer region of phi_site[] is updated with the interpolated
 *    values.
 *
 *****************************************************************************/

void phi_leesedwards_transformation() {

  int nhalo;
  int nlocal[3]; /* Local system size */
  int ib;        /* Index in buffer region */
  int ib0;       /* buffer region offset */
  int ic;        /* Index corresponding x location in real system */
  int jc, kc, n;
  int index, index0, index1, index2, index3;

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  double t;      /* Time */

  const double r6 = (1.0/6.0);

  int jdy;               /* Integral part of displacement */
  int j0, j1, j2, j3;    /* j values in real system to interpolate between */

  if (cart_size(Y) > 1) {
    /* This has its own routine. */
    phi_leesedwards_parallel();
  }
  else {
    /* No messages are required... */

    nhalo = coords_nhalo();
    coords_nlocal(nlocal);
    ib0 = nlocal[X] + nhalo + 1;

    /* -1.0 as zero required for first step; a 'feature' to
     * maintain the regression tests */
    t = 1.0*get_step() - 1.0;

    for (ib = 0; ib < le_get_nxbuffer(); ib++) {

      ic = le_index_buffer_to_real(ib);
      dy = le_buffer_displacement(ib, t);
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
	  index  = nop_*le_site_index(ib0 + ib, jc, kc);
	  index0 = nop_*le_site_index(ic, j0, kc);
	  index1 = nop_*le_site_index(ic, j1, kc);
	  index2 = nop_*le_site_index(ic, j2, kc);
	  index3 = nop_*le_site_index(ic, j3, kc);
	  for (n = 0; n < nop_; n++) {
	    phi_site[index + n] =
	      -  r6*fr*(fr-1.0)*(fr-2.0)*phi_site[index0 + n]
	      + 0.5*(fr*fr-1.0)*(fr-2.0)*phi_site[index1 + n]
	      - 0.5*fr*(fr+1.0)*(fr-2.0)*phi_site[index2 + n]
	      +        r6*fr*(fr*fr-1.0)*phi_site[index3 + n];
	  }
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_leesedwards_parallel
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
 *  Effect:
 *    Buffer region of phi_site[] is updated with the interpolated
 *    values.
 *
 *****************************************************************************/

static void phi_leesedwards_parallel(void) {

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
  const double r6 = (1.0/6.0);

  int      nrank_s[3];     /* send ranks */
  int      nrank_r[3];     /* recv ranks */

  const int tag0 = 1256;
  const int tag1 = 1257;
  const int tag2 = 1258;

  MPI_Comm le_comm;
  MPI_Request request[6];
  MPI_Status  status[3];


  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  ib0 = nlocal[X] + nhalo + 1;

  le_comm = le_communicator();

  /* Allocate the temporary buffer */

  n = nop_*(nlocal[Y] + 2*nhalo + 3)*(nlocal[Z] + 2*nhalo);

  buffer = (double *) malloc(n*sizeof(double));
  if (buffer == NULL) fatal("malloc(buffer) failed\n");

  /* -1.0 as zero required for fisrt step; this is a 'feature'
   * to ensure the regression tests stay te same */

  t = 1.0*get_step() - 1.0;

  /* One round of communication for each buffer plane */

  for (ib = 0; ib < le_get_nxbuffer(); ib++) {

    ic = le_index_buffer_to_real(ib);
    kc = 1 - nhalo;

    /* Work out the displacement-dependent quantities */

    dy = le_buffer_displacement(ib, t);
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

    le_jstart_to_ranks(j1, nrank_s, nrank_r);

    /* Local quantities: j2 is the position of j1 in local coordinates.
     * The three sections to send/receive are organised as follows:
     * jc is the number of j points in each case, while n is the
     * total number of data items. Note that n3 can be zero. */

    j2 = 1 + (j1 - 1) % nlocal[Y];
    assert(j2 >= 1);
    assert(j2 <= nlocal[Y]);

    jc = nlocal[Y] - j2 + 1;
    n1 = nop_*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(buffer, n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);

    jc = imin(nlocal[Y], j2 + 2*nhalo + 2);
    n2 = nop_*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(buffer + n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm,
	      request + 1);

    jc = imax(0, j2 - nlocal[Y] + 2*nhalo + 2);
    n3 = nop_*jc*(nlocal[Z] + 2*nhalo);
    MPI_Irecv(buffer + n1 + n2, n3, MPI_DOUBLE, nrank_r[2], tag2, le_comm,
	      request + 2);

    /* Post sends and wait for receives. */

    index = nop_*le_site_index(ic, j2, kc);
    MPI_Issend(phi_site + index, n1, MPI_DOUBLE, nrank_s[0], tag0, le_comm,
	       request + 3);
    index = nop_*le_site_index(ic, 1, kc);
    MPI_Issend(phi_site + index, n2, MPI_DOUBLE, nrank_s[1], tag1, le_comm,
	       request + 4);
    MPI_Issend(phi_site + index, n3, MPI_DOUBLE, nrank_s[2], tag2, le_comm,
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
	index = nop_*le_site_index(ib0 + ib, jc, kc);
	for (n = 0; n < nop_; n++) {
	  phi_site[index + n] =
	    -  r6*fr*(fr-1.0)*(fr-2.0)*buffer[nop_*(j0 + kc+nhalo-1) + n]
	    + 0.5*(fr*fr-1.0)*(fr-2.0)*buffer[nop_*(j1 + kc+nhalo-1) + n]
	    - 0.5*fr*(fr+1.0)*(fr-2.0)*buffer[nop_*(j2 + kc+nhalo-1) + n]
	    +        r6*fr*(fr*fr-1.0)*buffer[nop_*(j3 + kc+nhalo-1) + n];
	}
      }
    }

    /* Clean up the sends, and move to next buffer location. */

    MPI_Waitall(3, request + 3, status);
  }

  free(buffer);

  return;
}

/*****************************************************************************
 *
 *  phi_is_finite_difference
 *
 *****************************************************************************/

int phi_is_finite_difference() {

  return phi_finite_difference_;
}

/*****************************************************************************
 *
 *  phi_set_finite_difference
 *
 *****************************************************************************/

void phi_set_finite_difference() {

  phi_finite_difference_ = 1;
  return;
}

/*****************************************************************************
 *
 *  phi_op_get_phi_site
 *
 *****************************************************************************/

double phi_op_get_phi_site(const int index, const int nop) {

  assert(nop < nop_);
  return phi_site[nop_*index + nop];
}

/*****************************************************************************
 *
 *  phi_op_set_phi_site
 *
 *****************************************************************************/

void phi_op_set_phi_site(const int index, const int nop, const double value) {

  assert(nop < nop_);
  phi_site[nop_*index + nop] = value;

  return;
}

/*****************************************************************************
 *
 *  phi_set_q_tensor
 *
 *  Set the independent elements of the q tensor at lattice site index
 *  in phi_site. We assume q is constructed appropriately.
 *
 *****************************************************************************/

void phi_set_q_tensor(const int index, double q[3][3]) {

  assert(initialised_);
  assert(nop_ == 5);

  phi_site[nop_*index + XX] = q[X][X];
  phi_site[nop_*index + XY] = q[X][Y];
  phi_site[nop_*index + XZ] = q[X][Z];
  phi_site[nop_*index + YY] = q[Y][Y];
  phi_site[nop_*index + YZ] = q[Y][Z];

  return;
}

/*****************************************************************************
 *
 *  phi_get_q_tensor
 *
 *  Construct and return the symmetric q tensor at lattice index.
 *
 *****************************************************************************/

void phi_get_q_tensor(int index, double q[3][3]) {

  assert(initialised_);
  assert(nop_ == 5);

  q[X][X] = phi_site[nop_*index + XX];
  q[X][Y] = phi_site[nop_*index + XY];
  q[X][Z] = phi_site[nop_*index + XZ];
  q[Y][X] = q[X][Y];
  q[Y][Y] = phi_site[nop_*index + YY];
  q[Y][Z] = phi_site[nop_*index + YZ];
  q[Z][X] = q[X][Z];
  q[Z][Y] = q[Y][Z];
  q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];

  return;
}

/*****************************************************************************
 *
 *  phi_vector_set
 *
 *  Set q_a at site index.
 *
 *****************************************************************************/

void phi_vector_set(const int index, const double q[3]) {

  int ia;

  assert(initialised_);
  assert(nop_ == 3);

  for (ia = 0; ia < 3; ia++) {
    phi_site[nop_*index + ia] = q[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_vector
 *
 *  Retrieve vector order parameter at site index.
 *
 *****************************************************************************/

void phi_vector(const int index, double q[3]) {

  int ia;

  assert(initialised_);
  assert(nop_ == 3);

  for (ia = 0; ia < 3; ia++) {
    q[ia] = phi_site[nop_*index + ia];
  }

  return;
}
