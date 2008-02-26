/*****************************************************************************
 *
 *  phi.c
 *
 *  Scalar order parameter(s).
 *
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

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "site_map.h"
#include "timer.h"
#include "phi.h"

/* These are to be considered static */
double * phi_site;
double * delsq_phi_site;
double * grad_phi_site;
double * delsq_delsq_phi_site;
double * grad_delsq_phi_site;

static int initialised_ = 0;
static MPI_Datatype phi_xy_t_;
static MPI_Datatype phi_xz_t_;
static MPI_Datatype phi_yz_t_;

static void phi_init_mpi(void);

/****************************************************************************
 *
 *  phi_init
 *
 *  Allocate memory for the order parameter arra. If MPI2 is used
 *  this must use MPI_Alloc_mem() to allow use of Windows in the
 *  LE code.
 *
 ****************************************************************************/

void phi_init() {

  int nsites;
  int nlocal[3];

  get_N_local(nlocal);
  nsites = (nlocal[X]+2*nhalo_)*(nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_);

#ifdef _MPI_2_
 {
   int ifail;
   ifail = MPI_Alloc_mem(nsites*sizeof(double), MPI_INFO_NULL, &phi_site);
   if (ifail == MPI_ERR_NO_MEM) fatal("MPI_Alloc_mem(phi) failed\n");
 }
#else

  phi_site = (double *) calloc(nsites, sizeof(double));
  if (phi_site == NULL) fatal("calloc(phi) failed\n");

#endif

  /* Gradients */

  delsq_phi_site = (double *) calloc(nsites, sizeof(double));
  grad_phi_site = (double *) calloc(3*nsites, sizeof(double));
  grad_delsq_phi_site = (double *) calloc(3*nsites, sizeof(double));
  delsq_delsq_phi_site = (double *) calloc(nsites, sizeof(double));

  if (delsq_phi_site == NULL) fatal("calloc(delsq_phi_site) failed\n");
  if (grad_phi_site == NULL) fatal("calloc(grad_phi_site) failed\n");
  if (grad_delsq_phi_site == NULL) fatal("calloc(grad_delsq_phi) failed\n");
  if (delsq_delsq_phi_site == NULL) fatal("calloc(delsq_delsq_phi) failed\n");

  phi_init_mpi();
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
  MPI_Type_vector(1, nh[Y]*nh[Z]*nhalo_, 1, MPI_DOUBLE, &phi_yz_t_);
  MPI_Type_commit(&phi_yz_t_);

  /* XZ planes in the Y direction */
  MPI_Type_vector(nh[X], nh[Z]*nhalo_, nh[Y]*nh[Z], MPI_DOUBLE, &phi_xz_t_);
  MPI_Type_commit(&phi_xz_t_);

  /* XY planes in Z direction */
  MPI_Type_vector(nh[X]*nh[Y], nhalo_, nh[Z], MPI_DOUBLE, &phi_xy_t_);
  MPI_Type_commit(&phi_xy_t_);

  return;
}

/*****************************************************************************
 *
 *  phi_finish
 *
 *****************************************************************************/

void phi_finish() {

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

/*****************************************************************************
 *
 *  phi_compute_phi_site
 *
 *  Recompute the value of the order parameter at all the current
 *  fluid sites (domain proper).
 *
 *  The halo regions are immediately updated to reflect the new
 *  values.
 *
 *****************************************************************************/

void phi_compute_phi_site() {

  int     ic, jc, kc, index;
  int     nlocal[3];

  assert(initialised_);

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = get_site_index(ic, jc, kc);
	phi_site[index] = get_phi_at_site(index);
      }
    }
  }

  phi_halo();

  return;
}

/*****************************************************************************
 *
 *  phi_set_mean_phi
 *
 *  Compute the current mean phi in the system and remove the excess
 *  so that the mean phi is phi_global (allowing for presence of any
 *  particles or, for that matter, other solids).
 *
 *  The value of phi_global is generally (but not necessilarily) zero.
 *
 *****************************************************************************/

void phi_set_mean_phi(double phi_global) {

  int     index, ic, jc, kc;
  int     nlocal[3];
  double  phi_local = 0.0, phi_total, phi_correction;
  double  vlocal = 0.0, vtotal;

  get_N_local(nlocal);
  assert(initialised_);

  /* Compute the mean phi in the domain proper */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = get_site_index(ic, jc, kc);
	phi_local += get_phi_at_site(index);
	vlocal += 1.0;
      }
    }
  }

  /* All processes need the total phi, and number of fluid sites
   * to compute the mean */

  MPI_Allreduce(&phi_local, &phi_total, 1, MPI_DOUBLE, MPI_SUM, cart_comm());
  MPI_Allreduce(&vlocal, &vtotal,   1, MPI_DOUBLE,    MPI_SUM, cart_comm());

  /* The correction requied at each fluid site is then ... */
  phi_correction = phi_global - phi_total / vtotal;

  /* The correction is added to the rest distribution g[0],
   * which should be good approximation to where it should
   * all end up if there were a full reprojection. */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) == FLUID) {
	  index = get_site_index(ic, jc, kc);
	  phi_local = get_g_at_site(index, 0) + phi_correction;
	  set_g_at_site(index, 0,  phi_local);
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_halo
 *
 *****************************************************************************/

void phi_halo() {

  int nlocal[3];
  int ic, jc, kc, ihalo, ireal, nh;
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
          ihalo = get_site_index(0-nh, jc, kc);
          ireal = get_site_index(nlocal[X]-nh, jc, kc);
          phi_site[ihalo] = phi_site[ireal];

          ihalo = get_site_index(nlocal[X]+1+nh, jc, kc);
          ireal = get_site_index(1+nh, jc, kc);
          phi_site[ihalo] = phi_site[ireal];
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, X);
    forw = cart_neighb(FORWARD, X);

    ihalo = get_site_index(nlocal[X] + 1, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_yz_t_, forw, btag, comm, request);
    ihalo = get_site_index(1-nhalo_, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_yz_t_, back, ftag, comm, request+1);
    ireal = get_site_index(1, 1-nhalo_, 1-nhalo_);
    MPI_Issend(phi_site + ireal, 1, phi_yz_t_, back, btag, comm, request+2);
    ireal = get_site_index(nlocal[X] - nhalo_ + 1, 1-nhalo_, 1-nhalo_);
    MPI_Issend(phi_site + ireal, 1, phi_yz_t_, forw, ftag, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  /* XZ planes in the Y direction */

  if (cart_size(Y) == 1) {
    for (nh = 0; nh < nhalo_; nh++) {
      for (ic = 1-nhalo_; ic <= nlocal[X] + nhalo_; ic++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {
          ihalo = get_site_index(ic, 0-nh, kc);
          ireal = get_site_index(ic, nlocal[Y]-nh, kc);
          phi_site[ihalo] = phi_site[ireal];

          ihalo = get_site_index(ic, nlocal[Y]+1+nh, kc);
          ireal = get_site_index(ic, 1+nh, kc);
          phi_site[ihalo] = phi_site[ireal];
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, Y);
    forw = cart_neighb(FORWARD, Y);

    ihalo = get_site_index(1-nhalo_, nlocal[Y] + 1, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_xz_t_, forw, btag, comm, request);
    ihalo = get_site_index(1-nhalo_, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_xz_t_, back, ftag, comm, request+1);
    ireal = get_site_index(1-nhalo_, 1, 1-nhalo_);
    MPI_Issend(phi_site + ireal, 1, phi_xz_t_, back, btag, comm, request+2);
    ireal = get_site_index(1-nhalo_, nlocal[Y] - nhalo_ + 1, 1-nhalo_);
    MPI_Issend(phi_site + ireal, 1, phi_xz_t_, forw, ftag, comm, request+3);
    MPI_Waitall(4, request, status);
  }

  /* XY planes in the Z direction */

  if (cart_size(Z) == 1) {
    for (nh = 0; nh < nhalo_; nh++) {
      for (ic = 1 - nhalo_; ic <= nlocal[X] + nhalo_; ic++) {
        for (jc = 1 - nhalo_; jc <= nlocal[Y] + nhalo_; jc++) {
          ihalo = get_site_index(ic, jc, 0-nh);
          ireal = get_site_index(ic, jc, nlocal[Z]-nh);
          phi_site[ihalo] = phi_site[ireal];

          ihalo = get_site_index(ic, jc, nlocal[Z]+1+nh);
          ireal = get_site_index(ic, jc,            1+nh);
          phi_site[ihalo] = phi_site[ireal];
        }
      }
    }
  }
  else {

    back = cart_neighb(BACKWARD, Z);
    forw = cart_neighb(FORWARD, Z);

    ihalo = get_site_index(1-nhalo_, 1-nhalo_, nlocal[Z] + 1);
    MPI_Irecv(phi_site + ihalo,  1, phi_xy_t_, forw, btag, comm, request);
    ihalo = get_site_index(1-nhalo_, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(phi_site + ihalo,  1, phi_xy_t_, back, ftag, comm, request+1);
    ireal = get_site_index(1-nhalo_, 1-nhalo_, 1);
    MPI_Issend(phi_site + ireal, 1, phi_xy_t_, back, btag, comm, request+2);
    ireal = get_site_index(1-nhalo_, 1-nhalo_, nlocal[Z] - nhalo_ + 1);
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
  return phi_site[index];
}

/*****************************************************************************
 *
 *  phi_set_phi_site
 *
 *****************************************************************************/

void phi_set_phi_site(const int index, const double phi) {

  assert(initialised_);
  phi_site[index] = phi;
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
    grad[ia] = grad_phi_site[3*index + ia];
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
    grad_phi_site[3*index + ia] = grad[ia];
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
  return delsq_phi_site[index];
}

/*****************************************************************************
 *
 *  phi_set_delsq_phi_site
 *
 *****************************************************************************/

void phi_set_delsq_phi_site(const int index, const double delsq) {

  assert(initialised_);
  delsq_phi_site[index] = delsq;

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
 *  phi_get_delsq_sq_phi_site
 *
 *****************************************************************************/

double phi_get_delsq_sq_phi_site(const int index) {

  assert(initialised_);
  return delsq_delsq_phi_site[index];
}
