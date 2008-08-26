/***************************************************************************
 *
 *  hydrodynamics.c
 *
 *  Deals with the hydrodynamic sector quantities one would expect
 *  in Navier Stokes, rho, u, ...
 *
 *  $Id: lattice.c,v 1.8 2008-08-24 17:57:55 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 ***************************************************************************/

#include <assert.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "io_harness.h"
#include "lattice.h"


struct io_info_t * io_info_velocity_;

struct vector {double c[3];};

static struct vector * f;               /* Force on fluid nodes. */
static struct vector * u;               /* The fluid velocity field. */
static const int       nhalolocal = 1;  /* Actual halo swap extent */
static MPI_Datatype    mpi_vector_t;    /* For vector halo type */
static MPI_Datatype    halo_xy_t;
static MPI_Datatype    halo_xz_t;
static MPI_Datatype    halo_yz_t;

static int             initialised_ = 0;
static void hydrodynamics_init_mpi(void);
static void hydrodynamics_leesedwards_parallel(void);
static void hydrodynamics_init_io(void);
static int  hydrodynamics_u_read(FILE *, const int, const int, const int);
static int  hydrodynamics_u_write(FILE *, const int, const int, const int);

/****************************************************************************
 *
 *  hydrodynamics_init
 *
 *  Initialise.
 *
 ****************************************************************************/

void hydrodynamics_init() {

  int nlocal[3];
  int nsites, nbuffer;

  get_N_local(nlocal);
  nbuffer = le_get_nxbuffer();

  nsites = (nlocal[X]+2*nhalo_ + nbuffer)
    *(nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_);

  u = (struct vector *) calloc(nsites, sizeof(struct vector));
  f = (struct vector *) calloc(nsites, sizeof(struct vector));

  if (u == NULL) fatal("calloc(u) failed\n");
  if (f == NULL) fatal("calloc(f) failed\n");

  hydrodynamics_init_mpi();
  hydrodynamics_init_io();
  initialised_ = 1;

  return;
}

/****************************************************************************
 *
 *  hydrodynmaics_init_mpi
 *
 ****************************************************************************/

static void hydrodynamics_init_mpi() {

  int nlocal[3], nx, ny, nz;

  get_N_local(nlocal);
  nx = nlocal[X] + 2*nhalo_;
  ny = nlocal[Y] + 2*nhalo_;
  nz = nlocal[Z] + 2*nhalo_;

  MPI_Type_contiguous(sizeof(struct vector), MPI_BYTE, &mpi_vector_t);
  MPI_Type_commit(&mpi_vector_t);

  MPI_Type_vector(nx*ny, nhalolocal, nz, mpi_vector_t, &halo_xy_t);
  MPI_Type_commit(&halo_xy_t);

  MPI_Type_vector(nx, nz*nhalolocal, ny*nz, mpi_vector_t, &halo_xz_t);
  MPI_Type_commit(&halo_xz_t);

  MPI_Type_vector(1, ny*nz*nhalolocal, 1, mpi_vector_t, &halo_yz_t);
  MPI_Type_commit(&halo_yz_t);

  return;
}

/****************************************************************************
 *
 *  hydrodynamics_init_io
 *
 ****************************************************************************/

static void hydrodynamics_init_io() {

  io_info_velocity_ = io_info_create();

  io_info_set_name(io_info_velocity_, "Velocity field");
  io_info_set_read(io_info_velocity_, hydrodynamics_u_read);
  io_info_set_write(io_info_velocity_, hydrodynamics_u_write);
  io_info_set_bytesize(io_info_velocity_, 3*sizeof(double));

  io_write_metadata("phi", io_info_velocity_);

  return;
}

/****************************************************************************
 *
 *  hydrodynamics_finish
 *
 ****************************************************************************/

void hydrodynamics_finish(void) {

  free(u);
  free(f);

  MPI_Type_free(&mpi_vector_t);
  MPI_Type_free(&halo_xy_t);
  MPI_Type_free(&halo_yz_t);
  MPI_Type_free(&halo_xz_t);

  return;
}

/****************************************************************************
 *
 *  hydrodynamics_halo_u
 *
 *  Swap the velocity field.
 *
 ****************************************************************************/

void hydrodynamics_halo_u() {

  int nlocal[3];
  int ic, jc, kc, ihalo, ireal;
  int forw, back;

  const int   tagb = 8703;
  const int   tagf = 8704;
  MPI_Comm    comm = cart_comm();
  MPI_Request request[4];
  MPI_Status  status[4];

  assert(initialised_);
  assert(nhalolocal == 1); /* Code below is not general */

  get_N_local(nlocal);

  /* The x-direction (YZ plane) */

  if (cart_size(X) == 1) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        ihalo = get_site_index(0, jc, kc);
        ireal = get_site_index(nlocal[X], jc, kc);
	u[ihalo] = u[ireal];

        ihalo = get_site_index(nlocal[X]+1, jc, kc);
        ireal = get_site_index(1, jc, kc);
        u[ihalo] = u[ireal];
      }
    }
  }
  else {

    forw = cart_neighb(FORWARD, X);
    back = cart_neighb(BACKWARD, X);

    ihalo = get_site_index(nlocal[X] + 1, 1 - nhalo_, 1 - nhalo_);
    MPI_Irecv(u[ihalo].c, 1, halo_yz_t, forw, tagb, comm, request);
    ihalo = get_site_index(0, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(u[ihalo].c, 1, halo_yz_t, back, tagf, comm, request+1);
    ireal = get_site_index(1, 1-nhalo_, 1-nhalo_);
    MPI_Issend(u[ireal].c, 1, halo_yz_t, back, tagb, comm, request+2);
    ireal = get_site_index(nlocal[X], 1-nhalo_, 1-nhalo_);
    MPI_Issend(u[ireal].c, 1, halo_yz_t, forw, tagf, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  /* The y-direction (XZ plane) */

  if (cart_size(Y) == 1) {
    for (ic = 0; ic <= nlocal[X]+1; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        ihalo = get_site_index(ic, 0, kc);
        ireal = get_site_index(ic, nlocal[Y], kc);
        u[ihalo] = u[ireal];

        ihalo = get_site_index(ic, nlocal[Y]+1, kc);
        ireal = get_site_index(ic, 1, kc);
        u[ihalo] = u[ireal];
      }
    }
  }
  else {

    forw = cart_neighb(FORWARD, Y);
    back = cart_neighb(BACKWARD, Y);

    ihalo = get_site_index(1-nhalo_, nlocal[Y] + 1, 1-nhalo_);
    MPI_Irecv(u[ihalo].c, 1, halo_xz_t, forw, tagb, comm, request);
    ihalo = get_site_index(1-nhalo_, 0, 1-nhalo_);
    MPI_Irecv(u[ihalo].c, 1, halo_xz_t, back, tagf, comm, request+1);
    ireal = get_site_index(1-nhalo_, 1, 1-nhalo_);
    MPI_Issend(u[ireal].c, 1, halo_xz_t, back, tagb, comm, request+2);
    ireal = get_site_index(1-nhalo_, nlocal[Y], 1-nhalo_);
    MPI_Issend(u[ireal].c, 1, halo_xz_t, forw, tagf, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  /* Finally, z-direction (XY plane) */

  if (cart_size(Z) == 1) {
    for (ic = 0; ic <= nlocal[X]+1; ic++) {
      for (jc = 0; jc <= nlocal[Y]+1; jc++) {

        ihalo = get_site_index(ic, jc, 0);
        ireal = get_site_index(ic, jc, nlocal[Z]);
        u[ihalo] = u[ireal];

        ihalo = get_site_index(ic, jc, nlocal[Z]+1);
        ireal = get_site_index(ic, jc, 1);
        u[ihalo] = u[ireal];
      }
    }
  }
  else {

    forw = cart_neighb(FORWARD, Z);
    back = cart_neighb(BACKWARD, Z);

    ihalo = get_site_index(1-nhalo_, 1-nhalo_, nlocal[Z] + 1);
    MPI_Irecv(u[ihalo].c, 1, halo_xy_t, forw, tagb, comm, request);
    ihalo = get_site_index(1-nhalo_, 1-nhalo_, 0);
    MPI_Irecv(u[ihalo].c, 1, halo_xy_t, back, tagf, comm, request+1);
    ireal = get_site_index(1-nhalo_, 1-nhalo_, 1);
    MPI_Issend(u[ireal].c, 1, halo_xy_t, back, tagb, comm, request+2);
    ireal = get_site_index(1-nhalo_, 1-nhalo_, nlocal[Z]);
    MPI_Issend(u[ireal].c, 1, halo_xy_t, forw, tagf, comm, request+3);  
    MPI_Waitall(4, request, status);
  }

  return;
}

/****************************************************************************
 *
 *  hydrodynamics_set_force_local
 *
 *  Set the fluid force at site index
 *
 ****************************************************************************/

void hydrodynamics_set_force_local(const int index, const double force[3]) {

  int ia;

  assert(initialised_);
  for (ia = 0; ia < 3; ia++) {
    f[index].c[ia] = force[ia];
  }

  return;
}

/****************************************************************************
 *
 *  hydrodynamics_get_force_local
 *
 *  Return the velocity at site index.
 *
 ****************************************************************************/

void hydrodynamics_get_force_local(const int index, double force[3]) {

  int ia;

  assert(initialised_);
  for (ia = 0; ia < 3; ia++) {
    force[ia] = f[index].c[ia];
  }

  return;
}

/****************************************************************************
 *
 *  hydrodynamics_add_force_local
 *
 *  Accumulate (repeat, accumulate) the fluid force at site index
 *
 ****************************************************************************/

void hydrodynamics_add_force_local(const int index, const double force[3]) {

  int ia;

  assert(initialised_);
  for (ia = 0; ia < 3; ia++) {
    f[index].c[ia] += force[ia];
  }

  return;
}


/****************************************************************************
 *
 *  hydrodynamics_set_velocity
 *
 *  Return the velocity at site index.
 *
 ****************************************************************************/

void hydrodynamics_set_velocity(const int index, const double ulocal[3]) {

  int ia;

  assert(initialised_);
  for (ia = 0; ia < 3; ia++) {
    u[index].c[ia] = ulocal[ia];
  }

  return;
}

/****************************************************************************
 *
 *  hydrodynamics_get_velocity
 *
 *  Return the velocity at site index.
 *
 ****************************************************************************/

void hydrodynamics_get_velocity(const int index, double ulocal[3]) {

  int ia;

  assert(initialised_);
  for (ia = 0; ia < 3; ia++) {
    ulocal[ia] = u[index].c[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  hydrodynamics_zero_force
 *
 *  Set the force on the lattice sites to zero everywhere
 *  in the local domain.
 *
 *****************************************************************************/

void hydrodynamics_zero_force() {

  int ic, jc, kc, ia, index;
  int nlocal[3];

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  f[index].c[ia] = 0.0;
	}
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  hydrodynamics_stats
 *
 *  Report some velocity statistics, principally to check for Courant
 *  number violations.
 *
 ****************************************************************************/

void hydrodynamics_stats() {

  int ic, jc, kc, ia, index;
  int nlocal[3];
  double umin[3];
  double umax[3];
  double utmp[3];
  MPI_Comm comm = cart_comm();

  get_N_local(nlocal);

  for (ia = 0; ia < 3; ia++) {
    umin[ia] = FLT_MAX;
    umax[ia] = FLT_MIN;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  umin[ia] = dmin(umin[ia], u[index].c[ia]);
	  umax[ia] = dmax(umax[ia], u[index].c[ia]);
	}
      }
    }
  }

  MPI_Reduce(umin, utmp, 3, MPI_DOUBLE, MPI_MIN, 0, comm);

  for (ia = 0; ia < 3; ia++) {
    umin[ia] = utmp[ia];
  }

  MPI_Reduce(umax, utmp, 3, MPI_DOUBLE, MPI_MAX, 0, comm);

  for (ia = 0; ia < 3; ia++) {
    umax[ia] = utmp[ia];
  }

  info("Velocity stats:\n");
  info("[ umin ][ %g %g %g ]\n", umin[X], umin[Y], umin[Z]);
  info("[ umax ][ %g %g %g ]\n", umax[X], umax[Y], umax[Z]);

  return;
}

/*****************************************************************************
 *
 *  hydrodynamics_lees_edwards_transformation
 *
 *  Compute the 'look-across-the-boundary' values of the velocity field.
 *
 *****************************************************************************/

void hydrodynamics_leesedwards_transformation() {

  int nlocal[3]; /* Local system size */
  int ib;        /* Index in buffer region */
  int ib0;       /* buffer region offset */
  int ic;        /* Index corresponding x location in real system */
  int jc, kc, ia;

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  int jdy;       /* Integral part of displacement */
  int j1, j2;    /* j values in real system to interpolate between */

  assert(initialised_);

  if (cart_size(Y) > 1) {
    hydrodynamics_leesedwards_parallel();
  }
  else {

    get_N_local(nlocal);
    ib0 = nlocal[X] + nhalo_ + 1;

    for (ib = 0; ib < le_get_nxbuffer(); ib++) {

      ic = le_index_buffer_to_real(ib);
      dy = le_buffer_displacement(ib);
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
	  for (ia = 0; ia < 3; ia++) {
	  u[ADDR(ib0+ib,jc,kc)].c[ia] =
	    fr*u[ADDR(ic,j1,kc)].c[ia] + (1.0-fr)*u[ADDR(ic,j2,kc)].c[ia];
	  }
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  hydrodynamics_leesedwards_parallel
 *
 *  The Lees Edwards transformation for the velocity field in parallel.
 *
 *****************************************************************************/

static void hydrodynamics_leesedwards_parallel() {

  int      nlocal[3];      /* Local system size */
  int      noffset[3];     /* Local starting offset */
  struct vector * buffer;  /* Interpolation buffer */
  int ib;                  /* Index in buffer region */
  int ib0;                 /* buffer region offset */
  int ic;                  /* Index corresponding x location in real system */
  int jc, kc, j1, j2;
  int n, n1, n2;
  double dy;               /* Displacement for current ic->ib pair */
  double fr;               /* Fractional displacement */
  int jdy;                 /* Integral part of displacement */
  int ia;

  MPI_Comm le_comm = le_communicator();
  int      nrank_s[2];     /* send ranks */
  int      nrank_r[2];     /* recv ranks */
  const int tag0 = 1256;
  const int tag1 = 1257;

  MPI_Request request[4];
  MPI_Status  status[4];

  get_N_local(nlocal);
  get_N_offset(noffset);
  ib0 = nlocal[X] + nhalo_ + 1;

  /* Allocate the temporary buffer */

  n = (nlocal[Y] + 2*nhalo_ + 1)*(nlocal[Z] + 2*nhalo_);
  buffer = (struct vector *) malloc(n*sizeof(struct vector));
  if (buffer == NULL) fatal("hydrodynamics: malloc(le buffer) failed\n");

  /* One round of communication for each buffer plane */

  for (ib = 0; ib < le_get_nxbuffer(); ib++) {

    ic = le_index_buffer_to_real(ib);
    kc = 1 - nhalo_;

    /* Work out the displacement-dependent quantities */

    dy = le_buffer_displacement(ib);
    dy = fmod(dy, L(Y));
    jdy = floor(dy);
    fr  = dy - jdy;

    /* First j1 required is j1 = jc - jdy - 1 with jc = 1 - nhalo_.
     * Modular arithmetic ensures 1 <= j1 <= N_total(Y). */

    jc = noffset[Y] + 1 - nhalo_;
    j1 = 1 + (jc - jdy - 2 + 2*N_total(Y)) % N_total(Y);

    le_displacement_ranks(dy, nrank_r, nrank_s);

    /* Local quantities: given a local starting index j2, we receive
     * n1 + n2 sites into the buffer, and send n1 sites starting with
     * j2, and the remaining n2 sites from starting position nhalo_. */

    j2 = j1 % nlocal[Y];

    n1 = (nlocal[Y] - j2 + nhalo_)*(nlocal[Z] + 2*nhalo_);
    n2 = (j2 + nhalo_ + 1)*(nlocal[Z] + 2*nhalo_);

    /* Post receives, sends and wait. */

    MPI_Irecv(buffer[0].c, n1, mpi_vector_t, nrank_r[0], tag0,
	      le_comm, request);
    MPI_Irecv(buffer[n1].c, n2, mpi_vector_t, nrank_r[1], tag1,
	      le_comm, request+1);
    MPI_Issend(u[ADDR(ic,j2,kc)].c, n1, mpi_vector_t, nrank_s[0], tag0,
	       le_comm, request+2);
    MPI_Issend(u[ADDR(ic,nhalo_,kc)].c, n2, mpi_vector_t, nrank_s[1], tag1,
	       le_comm, request+3);

    MPI_Waitall(4, request, status);

    /* Perform the actual interpolation from temporary buffer to
     * phi_site[] buffer region. */

    for (jc = 1 - nhalo_; jc <= nlocal[Y] + nhalo_; jc++) {
      j1 = (jc + nhalo_ - 1    )*(nlocal[Z] + 2*nhalo_);
      j2 = (jc + nhalo_ - 1 + 1)*(nlocal[Z] + 2*nhalo_);
      for (kc = 1 - nhalo_; kc <= nlocal[Z] + nhalo_; kc++) {
	for (ia = 0; ia < 3; ia++) {
	  u[ADDR(ib0+ib,jc,kc)].c[ia] = fr*buffer[j1+kc+nhalo_-1].c[ia]
	    + (1.0-fr)*buffer[j2+kc+nhalo_-1].c[ia];
	}
      }
    }
  }

  free(buffer);

  return;
}

/*****************************************************************************
 *
 *  hydrodynamics_u_write
 *
 *****************************************************************************/

static int hydrodynamics_u_write(FILE * fp, const int ic, const int jc,
				 const int kc) {

  int index, n;

  index = le_site_index(ic, jc, kc);
  n = fwrite(u[index].c, sizeof(double), 3, fp);

  if (n != 3) fatal("fwrite(velocity) failed at %d %d %d\n", ic, jc, kc);

  return n;
}

/*****************************************************************************
 *
 *  hydrodynamics_u_read
 *
 *  Should not need to read u, as it is not strictly state.
 *  So it's a no op at the moment.
 *
 *****************************************************************************/

static int hydrodynamics_u_read(FILE * fp, const int ic, const int jc,
				const int kc) {

  return 0;
}

