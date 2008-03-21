/***************************************************************************
 *
 *  hydrodynamics.c
 *
 *  Deals with the hydrodynamic sector quantities one would expect
 *  in Navier Stokes, rho, u, ...
 *
 *  $Id: lattice.c,v 1.7.2.3 2008-03-21 09:22:34 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 ***************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "lattice.h"

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

/****************************************************************************
 *
 *  hydrodynamics_init
 *
 *  Initialise.
 *
 ****************************************************************************/

void hydrodynamics_init() {

  int nlocal[3];
  int nsites;

  get_N_local(nlocal);
  nsites = (nlocal[X]+2*nhalo_)*(nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_);

  u = (struct vector *) calloc(nsites, sizeof(struct vector));
  f = (struct vector *) calloc(nsites, sizeof(struct vector));

  if (u == NULL) fatal("calloc(u) failed\n");
  if (f == NULL) fatal("calloc(f) failed\n");

  hydrodynamics_init_mpi();
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

  MPI_Type_vector(nx*ny, 3*nhalolocal, 3*nz, mpi_vector_t, &halo_xy_t);
  MPI_Type_commit(&halo_xy_t);

  MPI_Type_vector(nx, 3*nz*nhalolocal, 3*ny*nz, mpi_vector_t, &halo_xz_t);
  MPI_Type_commit(&halo_xz_t);

  MPI_Type_vector(1, 3*nx*ny*nhalolocal, 1, mpi_vector_t, &halo_yz_t);
  MPI_Type_commit(&halo_yz_t);

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
