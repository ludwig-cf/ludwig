/*****************************************************************************
 *
 *  model.c
 *
 *  This encapsulates data/operations related to distributions.
 *  However, the implementation of "Site" is exposed for performance
 *  reasons. For non-performance critical operations, prefer the
 *  access functions.
 *
 *  The LB model is either _D3Q15_ or _D3Q19_, as included in model.h.
 *
 *  $Id: model.c,v 1.9.4.5 2008-06-13 19:15:31 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *  (c) 2008 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "model.h"
#include "timer.h"

const double cs2  = (1.0/3.0);
const double rcs2 = 3.0;
const double d_[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

Site  * site;

static int nsites_ = 0;
static int initialised_ = 0;

static MPI_Datatype DT_plane_XY;
static MPI_Datatype DT_plane_XZ;
static MPI_Datatype DT_plane_YZ;
MPI_Datatype DT_Site; /* currently referenced in leesedwards */
enum mpi_tags {TAG_FWD = 900, TAG_BWD}; 


/***************************************************************************
 *
 *  init_site
 *
 *  Allocate memory for the distributions. If MPI2 is used, then
 *  this must use the appropriate utility to accomodate LE planes.
 *
 *  Irrespective of the value of nhalo_, we only ever at the moment
 *  pass one plane worth of distribution values (ie., nhalo = 1).
 *
 ***************************************************************************/
 
void init_site() {

  int N[3];
  int nx, ny, nz;
  int nhalolocal = 1;

  get_N_local(N);

  nx = N[X] + 2*nhalo_;
  ny = N[Y] + 2*nhalo_;
  nz = N[Z] + 2*nhalo_;
  nsites_ = nx*ny*nz;

  info("Requesting %d bytes for site data\n", nsites_*sizeof(Site));

  site = (Site  *) calloc(nsites_, sizeof(Site));
  if (site == NULL) fatal("calloc(site) failed\n");

  /* Set up the MPI Datatypes used for site, and its corresponding
   * halo messages:
   *
   * in XY plane nx*ny blocks of 1 site with stride nz;
   * in XZ plane nx blocks of nz sites with stride ny*nz;
   * in YZ plane one contiguous block of ny*nz sites.
   *
   * This is only confirmed for nhalo_ = 1. */

  MPI_Type_contiguous(sizeof(Site), MPI_BYTE, &DT_Site);
  MPI_Type_commit(&DT_Site);

  MPI_Type_vector(nx*ny, nhalolocal, nz, DT_Site, &DT_plane_XY);
  MPI_Type_commit(&DT_plane_XY);

  MPI_Type_vector(nx, nz*nhalolocal, ny*nz, DT_Site, &DT_plane_XZ);
  MPI_Type_commit(&DT_plane_XZ);

  MPI_Type_vector(1, ny*nz*nhalolocal, 1, DT_Site, &DT_plane_YZ);
  MPI_Type_commit(&DT_plane_YZ);

  initialised_ = 1;

  return;
}

/*****************************************************************************
 *
 *  finish_site
 *
 *  Clean up.
 *
 *****************************************************************************/

void finish_site() {

#ifdef _MPI_2_
  MPI_Free_mem(site);
#else
  free(site);
#endif

  MPI_Type_free(&DT_Site);
  MPI_Type_free(&DT_plane_XY);
  MPI_Type_free(&DT_plane_XZ);
  MPI_Type_free(&DT_plane_YZ);

  return;
}

/*****************************************************************************
 *
 *  set_rho
 *
 *  Project rho onto the distribution at position index, assuming zero
 *  velocity and zero stress.
 *
 *****************************************************************************/

void set_rho(const double rho, const int index) {

  int   p;

  assert(index >= 0 || index < nsites_);

  for (p = 0; p < NVEL; p++) {
    site[index].f[p] = wv[p]*rho;
  }

  return;
}

/*****************************************************************************
 *
 *  set_rho_u_at_site
 *
 *  Project rho, u onto distribution at position index, assuming
 *  zero stress.
 *
 *****************************************************************************/

void set_rho_u_at_site(const double rho, const double u[], const int index) {

  int p;
  double udotc;

  assert(index >= 0 || index < nsites_);

  for (p = 0; p < NVEL; p++) {
    udotc = u[X]*cv[p][X] + u[Y]*cv[p][Y] + u[Z]*cv[p][Z];
    site[index].f[p] = wv[p]*(rho + rcs2*udotc);
  }

  return;
}

/*****************************************************************************
 *
 *  set_phi
 *
 *  Sets the order parameter distribution at index address, assuming
 *  zero order parameter flux and zero stress.
 *
 *  Note that this is currently incompatible with the reprojection
 *  at the collision stage where all the phi would go into the rest
 *  distribution.
 *
 ****************************************************************************/

void set_phi(const double phi, const int index) {

  int   p;

  assert(index >= 0 || index < nsites_);

  for (p = 0; p < NVEL; p++) {
    site[index].g[p] = wv[p]*phi;
  }

  return;
}

/*****************************************************************************
 *
 *  set_f_at_site
 *
 *****************************************************************************/

void set_f_at_site(const int index, const int p, const double fp) {

  assert(index >= 0 || index < nsites_);
  site[index].f[p] = fp;

  return;
}

/*****************************************************************************
 *
 *  get_f_at_site
 *
 *****************************************************************************/

double get_f_at_site(const int index, const int p) {

  assert(index >= 0 || index < nsites_);
  return site[index].f[p];
}

/*****************************************************************************
 *
 *  set_g_at_site
 *
 *****************************************************************************/

void set_g_at_site(const int index, const int p, const double gp) {

  assert(index >= 0 || index < nsites_);
  site[index].g[p] = gp;

  return;
}

/*****************************************************************************
 *
 *  get_g_at_site
 *
 *****************************************************************************/

double get_g_at_site(const int index, const int p) {

  assert(index >= 0 || index < nsites_);
  return site[index].g[p];
}

/*****************************************************************************
 *
 *  get_rho_at_site
 *
 *  Return the density at lattice node index.
 *
 *****************************************************************************/

double get_rho_at_site(const int index) {

  double rho;
  double * f;
  int   p;

  assert(index >= 0 || index < nsites_);

  rho = 0.0;
  f = site[index].f;

  for (p = 0; p < NVEL; p++)
    rho += f[p];

  return rho;
}

/****************************************************************************
 *
 *  get_phi_at_site
 *
 *  Return the order parameter at lattice node index.
 *
 ****************************************************************************/

double get_phi_at_site(const int index) {

  double   phi;
  double * g;
  int     p;

  assert(index >= 0 || index < nsites_);

  phi = 0.0;
  g = site[index].g;

  for (p = 0; p < NVEL; p++) {
    phi += g[p];
  }

  return phi;
}

/*****************************************************************************
 *
 *  get_momentum_at_site
 *
 *  Return momentum density at lattice node index.
 *
 *****************************************************************************/

void get_momentum_at_site(const int index, double rhou[ND]) {

  double  * f;
  int       i, p;

  assert(index >= 0 || index < nsites_);

  for (i = 0; i < ND; i++) {
    rhou[i] = 0.0;
  }

  f  = site[index].f;

  for (p = 0; p < NVEL; p++) {
    for (i = 0; i < ND; i++) {
      rhou[i] += f[p]*cv[p][i];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  halo_site
 *
 *  Swap the distributions at the periodic/processor boundaries
 *  in each direction.
 *
 *****************************************************************************/

void halo_site() {

  int i, j, k;
  int ihalo, ireal;
  int N[3];

  MPI_Request request[4];
  MPI_Status status[4];

  assert(initialised_);
  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);

  /* The x-direction (YZ plane) */

  if (cart_size(X) == 1) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	ihalo = get_site_index(0, j, k);
	ireal = get_site_index(N[X], j, k);
	site[ihalo] = site[ireal];

	ihalo = get_site_index(N[X]+1, j, k);
	ireal = get_site_index(1, j, k);
	site[ihalo] = site[ireal];
      }
    }
  }
  else {

    ihalo = get_site_index(N[X] + 1, 1 - nhalo_, 1 - nhalo_);
    MPI_Irecv(&site[ihalo].f[0], 1, DT_plane_YZ,
	      cart_neighb(FORWARD,X), TAG_BWD, cart_comm(), &request[0]);
    ihalo = get_site_index(0, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(&site[ihalo].f[0], 1, DT_plane_YZ, cart_neighb(BACKWARD,X),
	      TAG_FWD, cart_comm(), &request[1]);
    ireal = get_site_index(1, 1-nhalo_, 1-nhalo_);
    MPI_Issend(&site[ireal].f[0], 1, DT_plane_YZ, cart_neighb(BACKWARD,X),
	       TAG_BWD, cart_comm(), &request[2]);
    ireal = get_site_index(N[X], 1-nhalo_, 1-nhalo_);
    MPI_Issend(&site[ireal].f[0], 1, DT_plane_YZ, cart_neighb(FORWARD,X),
	       TAG_FWD, cart_comm(), &request[3]);
    MPI_Waitall(4, request, status);
  }
  
  /* The y-direction (XZ plane) */

  if (cart_size(Y) == 1) {
    for (i = 0; i <= N[X]+1; i++) {
      for (k = 1; k <= N[Z]; k++) {

	ihalo = get_site_index(i, 0, k);
	ireal = get_site_index(i, N[Y], k);
	site[ihalo] = site[ireal];

	ihalo = get_site_index(i, N[Y]+1, k);
	ireal = get_site_index(i, 1, k);
	site[ihalo] = site[ireal];
      }
    }
  }
  else {
    ihalo = get_site_index(1-nhalo_, N[Y] + 1, 1-nhalo_);
    MPI_Irecv(site[ihalo].f, 1, DT_plane_XZ,
	      cart_neighb(FORWARD,Y), TAG_BWD, cart_comm(), &request[0]);
    ihalo = get_site_index(1-nhalo_, 0, 1-nhalo_);
    MPI_Irecv(site[ihalo].f, 1, DT_plane_XZ, cart_neighb(BACKWARD,Y),
	      TAG_FWD, cart_comm(), &request[1]);
    ireal = get_site_index(1-nhalo_, 1, 1-nhalo_);
    MPI_Issend(site[ireal].f, 1, DT_plane_XZ, cart_neighb(BACKWARD,Y),
	       TAG_BWD, cart_comm(), &request[2]);
    ireal = get_site_index(1-nhalo_, N[Y], 1-nhalo_);
    MPI_Issend(site[ireal].f, 1, DT_plane_XZ, cart_neighb(FORWARD,Y),
	       TAG_FWD, cart_comm(), &request[3]);
    MPI_Waitall(4, request, status);
  }
  
  /* Finally, z-direction (XY plane) */

  if (cart_size(Z) == 1) {
    for (i = 0; i<= N[X]+1; i++) {
      for (j = 0; j <= N[Y]+1; j++) {

	ihalo = get_site_index(i, j, 0);
	ireal = get_site_index(i, j, N[Z]);
	site[ihalo] = site[ireal];

	ihalo = get_site_index(i, j, N[Z]+1);
	ireal = get_site_index(i, j, 1);
	site[ihalo] = site[ireal];
      }
    }
  }
  else {

    ihalo = get_site_index(1-nhalo_, 1-nhalo_, N[Z] + 1);
    MPI_Irecv(site[ihalo].f, 1, DT_plane_XY, cart_neighb(FORWARD,Z),
	      TAG_BWD, cart_comm(), &request[0]);
    ihalo = get_site_index(1-nhalo_, 1-nhalo_, 0);
    MPI_Irecv(site[ihalo].f, 1, DT_plane_XY, cart_neighb(BACKWARD,Z),
	      TAG_FWD, cart_comm(), &request[1]);
    ireal = get_site_index(1-nhalo_, 1-nhalo_, 1);
    MPI_Issend(site[ireal].f, 1, DT_plane_XY, cart_neighb(BACKWARD,Z),
	       TAG_BWD, cart_comm(), &request[2]);
    ireal = get_site_index(1-nhalo_, 1-nhalo_, N[Z]);
    MPI_Issend(site[ireal].f, 1, DT_plane_XY, cart_neighb(FORWARD,Z),
	       TAG_FWD, cart_comm(), &request[3]);  
    MPI_Waitall(4, request, status);
  }
 
  TIMER_stop(TIMER_HALO_LATTICE);

  return;
}
