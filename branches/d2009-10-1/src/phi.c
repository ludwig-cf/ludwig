/*****************************************************************************
 *
 *  phi.c
 *
 *  Scalar order parameter.
 *
 *  $Id: phi.c,v 1.11.4.2 2009-11-13 14:34:50 kevin Exp $
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
#include "timer.h"
#include "phi.h"

struct io_info_t * io_info_phi;

/* Currently global scope */
int nop_ = 1;

/* Shift the gradients to phi-gradients */
double * phi_site;
double * delsq_phi_site;
double * grad_phi_site;
double * delsq_delsq_phi_site;
double * grad_delsq_phi_site;

static int initialised_ = 0;
static int phi_finite_difference_ = 0;  /* Default is LB for order parameter */
static int ngradient_ = 0;
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
 *  Allocate memory for the order parameter arra. If MPI2 is used
 *  this must use MPI_Alloc_mem() to allow use of Windows in the
 *  LE code.
 *
 *  Space for buffers to hold Lees Edwards interpolated quantities
 *  is added to the main array for convenince. It has no effect on
 *  the halo regions.
 *
 *  Note that only the first derivatives of the order parameters
 *  grad_phi_site, and delsq_phi_site are required for nop_ > 1.
 *  Storage is arranged as phi[nx][ny][nz][nop]
 *                         grad_phi[nx][ny[nz][nop][3] 
 *
 ****************************************************************************/

void phi_init() {

  int nsites;
  int nbuffer;
  int nlocal[3];

  get_N_local(nlocal);
  nbuffer = le_get_nxbuffer();

  nsites = (nlocal[X]+2*nhalo_ + nbuffer)
    *(nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_);

  phi_site = (double *) calloc(nop_*nsites, sizeof(double));
  if (phi_site == NULL) fatal("calloc(phi) failed\n");

  /* Gradients */

  delsq_phi_site = (double *) calloc(nop_*nsites, sizeof(double));
  grad_phi_site = (double *) calloc(3*nop_*nsites, sizeof(double));
  grad_delsq_phi_site = (double *) calloc(3*nsites, sizeof(double));
  delsq_delsq_phi_site = (double *) calloc(nsites, sizeof(double));

  if (delsq_phi_site == NULL) fatal("calloc(delsq_phi_site) failed\n");
  if (grad_phi_site == NULL) fatal("calloc(grad_phi_site) failed\n");
  if (grad_delsq_phi_site == NULL) fatal("calloc(grad_delsq_phi) failed\n");
  if (delsq_delsq_phi_site == NULL) fatal("calloc(delsq_delsq_phi) failed\n");

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

  int nlocal[3], nh[3];

  get_N_local(nlocal);

  nh[X] = nlocal[X] + 2*nhalo_;
  nh[Y] = nlocal[Y] + 2*nhalo_;
  nh[Z] = nlocal[Z] + 2*nhalo_;

  /* YZ planes in the X direction */
  MPI_Type_vector(1, nh[Y]*nh[Z]*nhalo_*nop_, 1, MPI_DOUBLE, &phi_yz_t_);
  MPI_Type_commit(&phi_yz_t_);

  /* XZ planes in the Y direction */
  MPI_Type_vector(nh[X], nh[Z]*nhalo_*nop_, nh[Y]*nh[Z]*nop_, MPI_DOUBLE,
		  &phi_xz_t_);
  MPI_Type_commit(&phi_xz_t_);

  /* XY planes in Z direction */
  MPI_Type_vector(nh[X]*nh[Y], nhalo_*nop_, nh[Z]*nop_, MPI_DOUBLE,
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
  free(delsq_phi_site);
  free(grad_phi_site);
  free(grad_delsq_phi_site);
  free(delsq_delsq_phi_site);

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

/****************************************************************************
 *
 *  phi_gradient_level
 *
 *  Return the order of spatial derivatives currently required.
 *
 ****************************************************************************/

int phi_gradient_level(void) {

  return ngradient_;
}

/****************************************************************************
 *
 *  phi_gradient_level_set
 *
 *  This can be either 2 or 4.
 *
 ****************************************************************************/

void phi_gradient_level_set(const int n) {

  assert(n == 2 || n == 4);
  ngradient_ = n;
  return;
}

/*****************************************************************************
 *
 *  phi_halo
 *
 *****************************************************************************/

void phi_halo() {

  int nlocal[3];
  int ic, jc, kc, ihalo, ireal, nh, n;
  int back, forw;
  const int btag = 2061;
  const int ftag = 2062;
  MPI_Comm comm = cart_comm();
  MPI_Request request[4];
  MPI_Status status[4];

  assert(initialised_);
  
  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(nlocal);

  /* YZ planes in the X direction */

  if (cart_size(X) == 1) {
    for (nh = 0; nh < nhalo_; nh++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1 ; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nop_; n++) {
	    phi_site[nop_*ADDR(0-nh, jc,kc) + n]
	      = phi_site[nop_*ADDR(nlocal[X]-nh, jc, kc) + n];
	    phi_site[nop_*ADDR(nlocal[X]+1+nh, jc,kc) + n]
	      = phi_site[nop_*ADDR(1+nh, jc, kc) + n];
	  }
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, X);
    forw = cart_neighb(FORWARD, X);

    ihalo = ADDR(nlocal[X] + 1, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_yz_t_, forw, btag, comm, request);
    ihalo = ADDR(1-nhalo_, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_yz_t_, back, ftag, comm, request+1);
    ireal = ADDR(1, 1-nhalo_, 1-nhalo_);
    MPI_Issend(phi_site + ireal, 1, phi_yz_t_, back, btag, comm, request+2);
    ireal = ADDR(nlocal[X] - nhalo_ + 1, 1-nhalo_, 1-nhalo_);
    MPI_Issend(phi_site + ireal, 1, phi_yz_t_, forw, ftag, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  /* XZ planes in the Y direction */

  if (cart_size(Y) == 1) {
    for (nh = 0; nh < nhalo_; nh++) {
      for (ic = 1-nhalo_; ic <= nlocal[X] + nhalo_; ic++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nop_; n++) {
	    phi_site[nop_*ADDR(ic,0-nh, kc) + n]
	      = phi_site[nop_*ADDR(ic, nlocal[Y]-nh, kc) + n];
	    phi_site[nop_*ADDR(ic,nlocal[Y]+1+nh, kc) + n]
	      = phi_site[nop_*ADDR(ic, 1+nh, kc) + n];
	  }
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, Y);
    forw = cart_neighb(FORWARD, Y);

    ihalo = ADDR(1-nhalo_, nlocal[Y] + 1, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_xz_t_, forw, btag, comm, request);
    ihalo = ADDR(1-nhalo_, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_xz_t_, back, ftag, comm, request+1);
    ireal = ADDR(1-nhalo_, 1, 1-nhalo_);
    MPI_Issend(phi_site + ireal, 1, phi_xz_t_, back, btag, comm, request+2);
    ireal = ADDR(1-nhalo_, nlocal[Y] - nhalo_ + 1, 1-nhalo_);
    MPI_Issend(phi_site + ireal, 1, phi_xz_t_, forw, ftag, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  /* XY planes in the Z direction */

  if (cart_size(Z) == 1) {
    for (nh = 0; nh < nhalo_; nh++) {
      for (ic = 1 - nhalo_; ic <= nlocal[X] + nhalo_; ic++) {
        for (jc = 1 - nhalo_; jc <= nlocal[Y] + nhalo_; jc++) {
	  for (n = 0; n < nop_; n++) {
	    phi_site[nop_*ADDR(ic,jc, 0-nh) + n]
	      = phi_site[nop_*ADDR(ic, jc, nlocal[Z]-nh) + n];
	    phi_site[nop_*ADDR(ic,jc, nlocal[Z]+1+nh) + n]
	      = phi_site[nop_*ADDR(ic, jc, 1+nh) + n];
	  }
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, Z);
    forw = cart_neighb(FORWARD, Z);

    ihalo = ADDR(1-nhalo_, 1-nhalo_, nlocal[Z] + 1);
    MPI_Irecv(phi_site + ihalo,  1, phi_xy_t_, forw, btag, comm, request);
    ihalo = ADDR(1-nhalo_, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_xy_t_, back, ftag, comm, request+1);
    ireal = ADDR(1-nhalo_, 1-nhalo_, 1);
    MPI_Issend(phi_site + ireal, 1, phi_xy_t_, back, btag, comm, request+2);
    ireal = ADDR(1-nhalo_, 1-nhalo_, nlocal[Z] - nhalo_ + 1);
    MPI_Issend(phi_site + ireal, 1, phi_xy_t_, forw, ftag, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  TIMER_stop(TIMER_HALO_LATTICE);

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
 *  phi_get_grad_phi_site
 *
 *****************************************************************************/

void phi_get_grad_phi_site(const int index, double grad[3]) {

  int ia;

  assert(initialised_);
  for (ia = 0; ia < 3; ia++) {
    grad[ia] = grad_phi_site[3*nop_*index + ia];
  }
 
  return;
}

/*****************************************************************************
 *
 *  phi_set_grad_phi_site
 *
 *****************************************************************************/

void phi_set_grad_phi_site(const int index, const double grad[3]) {

  int ia;

  assert(initialised_);
  for (ia = 0; ia < 3; ia++) {
    grad_phi_site[3*nop_*index + ia] = grad[ia];
  }
 
  return;
}

/*****************************************************************************
 *
 *  phi_get_delsq_phi_site
 *
 *****************************************************************************/

double phi_get_delsq_phi_site(const int index) {

  assert(initialised_);
  return delsq_phi_site[nop_*index];
}

/*****************************************************************************
 *
 *  phi_set_delsq_phi_site
 *
 *****************************************************************************/

void phi_set_delsq_phi_site(const int index, const double delsq) {

  assert(initialised_);
  delsq_phi_site[nop_*index] = delsq;

  return;
}

/*****************************************************************************
 *
 *  phi_get_grad_delsq_phi_site
 *
 *****************************************************************************/

void phi_get_grad_delsq_phi_site(const int index, double grad[3]) {

  int ia;

  assert(initialised_);
  for (ia = 0; ia < 3; ia++) {
    grad[ia] = grad_delsq_phi_site[3*index + ia];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_get_delsq_delsq_phi_site
 *
 *****************************************************************************/

double phi_get_delsq_delsq_phi_site(const int index) {

  assert(initialised_);
  return delsq_delsq_phi_site[index];
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
  int index, n;

  assert(nop_ == 1);
  index = le_site_index(ic, jc, kc);
  n = fscanf(fp, "%22le", phi_site + index);

  if (n != 1) fatal("fscanf(phi) failed at index %d", index);

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

  int nlocal[3]; /* Local system size */
  int ib;        /* Index in buffer region */
  int ib0;       /* buffer region offset */
  int ic;        /* Index corresponding x location in real system */
  int jc, kc, n;

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  double t;      /* Time */
  int jdy;       /* Integral part of displacement */
  int j1, j2;    /* j values in real system to interpolate between */

  if (cart_size(Y) > 1) {
    /* This has its own routine. */
    phi_leesedwards_parallel();
  }
  else {
    /* No messages are required... */

    get_N_local(nlocal);
    ib0 = nlocal[X] + nhalo_ + 1;

    /* -1.0 as zero required for first step; a 'feature' to
     * maintain the regression tests */
    t = 1.0*get_step() - 1.0;

    for (ib = 0; ib < le_get_nxbuffer(); ib++) {

      ic = le_index_buffer_to_real(ib);
      dy = le_buffer_displacement(ib, t);
      dy = fmod(dy, L(Y));
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1 - nhalo_; jc <= nlocal[Y] + nhalo_; jc++) {
	/* Actually required here is j1 = jc - jdy - 1, but there's
	 * horrible modular arithmetic for the periodic boundaries
	 * to ensure 1 <= j1,j2 <= nlocal[Y] */
	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];
	for (kc = 1 - nhalo_; kc <= nlocal[Z] + nhalo_; kc++) {
	  for (n = 0; n < nop_; n++) {
	    phi_site[nop_*ADDR(ib0+ib,jc,kc) + n] =
	      fr*phi_site[nop_*ADDR(ic,j1,kc) + n]
	      + (1.0-fr)*phi_site[nop_*ADDR(ic,j2,kc) + n];
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
 *  communication in parallel. The phi halo regions must be
 *  up-to-date here.
 *
 *  Each process may require communication with at most 4 others
 *  (and at least 2 others) to specify completely the interpolated
 *  buffer region (extending into the halo). The direction of the
 *  communcation in the cartesian communicator depends on the
 *  velocity jump at a given boundary. 
 *
 *  Effect:
 *    Buffer region of phi_site[] is updated with the interpolated
 *    values.
 *
 *****************************************************************************/

static void phi_leesedwards_parallel() {

  int      nlocal[3];      /* Local system size */
  int      noffset[3];     /* Local starting offset */
  double * buffer;         /* Interpolation buffer */
  int ib;                  /* Index in buffer region */
  int ib0;                 /* buffer region offset */
  int ic;                  /* Index corresponding x location in real system */
  int jc, kc, j1, j2;
  int n, n1, n2;
  double dy;               /* Displacement for current ic->ib pair */
  double fr;               /* Fractional displacement */
  double t;                /* Time */
  int jdy;                 /* Integral part of displacement */

  MPI_Comm le_comm = le_communicator();
  int      nrank_s[2];     /* send ranks */
  int      nrank_r[2];     /* recv ranks */
  const int tag0 = 1254;
  const int tag1 = 1255;

  MPI_Request request[4];
  MPI_Status  status[4];

  get_N_local(nlocal);
  get_N_offset(noffset);
  ib0 = nlocal[X] + nhalo_ + 1;

  /* Allocate the temporary buffer */

  n = nop_*(nlocal[Y] + 2*nhalo_ + 1)*(nlocal[Z] + 2*nhalo_);
  buffer = (double *) malloc(n*sizeof(double));
  if (buffer == NULL) fatal("malloc(buffer) failed\n");

  /* -1.0 as zero required for fisrt step; this is a 'feature'
   * to ensure the regression tests stay te same */

  t = 1.0*get_step() - 1.0;

  /* One round of communication for each buffer plane */

  for (ib = 0; ib < le_get_nxbuffer(); ib++) {

    ic = le_index_buffer_to_real(ib);
    kc = 1 - nhalo_;

    /* Work out the displacement-dependent quantities */

    dy = le_buffer_displacement(ib, t);
    dy = fmod(dy, L(Y));
    jdy = floor(dy);
    fr  = dy - jdy;

    /* In the real system (no halos) the first point we require is
     * j1 = jc - jdy - 1 with 
     * jc = noffset[Y] + 1 in the global coordinates. This determines
     * which processors we are going to comunicate with. Modular
     * arithmetic ensures 1 <= j1 <= N_total(Y) */

    jc = noffset[Y] + 1;
    j1 = 1 + (jc - jdy - 2 + 2*N_total(Y)) % N_total(Y);
    assert(j1 >= 1);
    assert(j1 <= N_total(Y));

    le_jstart_to_ranks(j1, nrank_s, nrank_r);

    /* Local quantities: j2 is the position of j1 in local coordinates
     * and marks the dividing line of the two send sections. However,
     * we can grab at least nhalo_ points to the left of j2, and another
     * nhalo_ points at the right to the other send section, making up
     * the (nlocal[Y] + 2*nhalo_ + 1) points we need to send. */

    j2 = 1 + (j1 - 1) % nlocal[Y];
    assert(j2 >= 1);
    assert(j2 <= nlocal[Y]);

    n1 = nop_*(nlocal[Y] - j2 + 1 + nhalo_)*(nlocal[Z] + 2*nhalo_);
    n2 = nop_*(j2 + nhalo_)*(nlocal[Z] + 2*nhalo_);

    /* Post receives, sends and wait. */

    MPI_Irecv(buffer,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);
    MPI_Irecv(buffer+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm, request+1);
    MPI_Issend(phi_site + nop_*ADDR(ic,j2-nhalo_,kc), n1, MPI_DOUBLE,
	       nrank_s[0], tag0, le_comm, request+2);
    MPI_Issend(phi_site + nop_*ADDR(ic,1,kc), n2, MPI_DOUBLE,
	       nrank_s[1], tag1, le_comm, request+3);

    MPI_Waitall(4, request, status);

    /* Perform the actual interpolation from temporary buffer to
     * phi_site[] buffer region. */

    for (jc = 1 - nhalo_; jc <= nlocal[Y] + nhalo_; jc++) {
      j1 = (jc + nhalo_ - 1    )*(nlocal[Z] + 2*nhalo_);
      j2 = (jc + nhalo_ - 1 + 1)*(nlocal[Z] + 2*nhalo_);
      for (kc = 1 - nhalo_; kc <= nlocal[Z] + nhalo_; kc++) {
	for (n = 0; n < nop_; n++) {
	  phi_site[nop_*ADDR(ib0+ib,jc,kc) + n]
	    = fr*buffer[nop_*(j1 + kc+nhalo_-1) + n]
	    + (1.0-fr)*buffer[nop_*(j2 + kc+nhalo_-1) + n];
	}
      }
    }
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
 *  phi_op_get_delsq_phi_site
 *
 *****************************************************************************/

double phi_op_get_delsq_phi_site(const int index, const int nop) {

  assert(nop < nop_);
  return delsq_phi_site[nop_*index + nop];
}

/*****************************************************************************
 *
 *  phi_op_get_grad_phi_site
 *
 *****************************************************************************/

void phi_op_get_grad_phi_site(const int index, const int nop, double * grad) {

  int ia;

  assert(nop < nop_);

  for (ia = 0; ia < 3; ia++) {
    grad[ia] = grad_phi_site[3*(nop_*index + nop) + ia];
  }

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

  phi_site[nop_*index + QXX] = q[X][X];
  phi_site[nop_*index + QXY] = q[X][Y];
  phi_site[nop_*index + QXZ] = q[X][Z];
  phi_site[nop_*index + QYY] = q[Y][Y];
  phi_site[nop_*index + QYZ] = q[Y][Z];

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

  q[X][X] = phi_site[nop_*index + QXX];
  q[X][Y] = phi_site[nop_*index + QXY];
  q[X][Z] = phi_site[nop_*index + QXZ];
  q[Y][X] = q[X][Y];
  q[Y][Y] = phi_site[nop_*index + QYY];
  q[Y][Z] = phi_site[nop_*index + QYZ];
  q[Z][X] = q[X][Z];
  q[Z][Y] = q[Y][Z];
  q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];

  return;
}

/*****************************************************************************
 *
 *  phi_get_q_gradient_tensor
 *
 *  Return the rank 3 gradient tensor of q.
 *
 *****************************************************************************/

void phi_get_q_gradient_tensor(const int index, double dq[3][3][3]) {

  int ia;

  assert(initialised_);
  assert(nop_ == 5);

  for (ia = 0; ia < 3; ia++) {
    dq[ia][X][X] = grad_phi_site[3*(nop_*index + QXX) + ia];
    dq[ia][X][Y] = grad_phi_site[3*(nop_*index + QXY) + ia];
    dq[ia][X][Z] = grad_phi_site[3*(nop_*index + QXZ) + ia];
    dq[ia][Y][X] = dq[ia][X][Y];
    dq[ia][Y][Y] = grad_phi_site[3*(nop_*index + QYY) + ia];
    dq[ia][Y][Z] = grad_phi_site[3*(nop_*index + QYZ) + ia];
    dq[ia][Z][X] = dq[ia][X][Z];
    dq[ia][Z][Y] = dq[ia][Y][Z];
    dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_get_q_delsq_tensor
 *
 *  Return the delsq Q_ab tensor at site index.
 *
 *****************************************************************************/

void phi_get_q_delsq_tensor(const int index, double dsq[3][3]) {

  dsq[X][X] = delsq_phi_site[nop_*index + QXX];
  dsq[X][Y] = delsq_phi_site[nop_*index + QXY];
  dsq[X][Z] = delsq_phi_site[nop_*index + QXZ];
  dsq[Y][X] = dsq[X][Y];
  dsq[Y][Y] = delsq_phi_site[nop_*index + QYY];
  dsq[Y][Z] = delsq_phi_site[nop_*index + QYZ];
  dsq[Z][X] = dsq[X][Z];
  dsq[Z][Y] = dsq[Y][Z];
  dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];

  return;
}
