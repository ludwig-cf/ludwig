/*****************************************************************************
 *
 *  coords.c
 *
 *  The physical coordinate system and the MPI Cartesian Communicator.
 *
 *  $Id: coords.c,v 1.4 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>

#include "pe.h"
#include "coords.h"

/* The effective state here is, with default values: */

static int nhalo_ = 1;
static int ntotal_[3]                  = {64, 64, 64};
static int periodic[3]                 = {1, 1, 1};
static int pe_cartesian_size[3]        = {1, 1, 1};
static MPI_Comm cartesian_communicator = MPI_COMM_NULL;
static MPI_Comm cart_periodic          = MPI_COMM_NULL;
static int reorder_                    = 1;
static int initialised_                = 0;

/* The following lookups will be set by a call to coords_init(), for
 * convenience, based on the current state: */

static int xfac_;
static int yfac_;
static int n_local[3];

static int pe_cartesian_rank             = 0;
static int pe_cartesian_coordinates[3]   = {0, 0, 0};
static int pe_cartesian_neighbours[2][3] = {{0, 0, 0}, {0, 0, 0}};

/* Lmin is fixed for all current use. */

static double lmin[3] = {0.5, 0.5, 0.5};

static double radius_ = FLT_MAX;

static void default_decomposition(void);
static int  is_ok_decomposition(void);

/*****************************************************************************
 *
 *  coords_init
 *
 *  Set up the Cartesian communicator for the current state.
 *  We must check here that the decomposition is allowable.
 *
 *****************************************************************************/

void coords_init() {

  int n;
  int iperiodic[3] = {1, 1, 1};

  assert(initialised_ == 0);

  if (is_ok_decomposition()) {
    /* The user decomposition is selected */
  }
  else {
    /* Look for a default */
    default_decomposition();
  }

  /* A communicator which is always periodic: */

  MPI_Cart_create(pe_comm(), 3, pe_cartesian_size, iperiodic, reorder_,
		  &cart_periodic);

  /* Set up the communicator and the Cartesian neighbour lists for
   * the requested communicator. */

  for (n = 0; n < 3; n++) {
    iperiodic[n] = is_periodic(n);
  }

  MPI_Cart_create(pe_comm(), 3, pe_cartesian_size, iperiodic, reorder_,
		  &cartesian_communicator);
  MPI_Comm_rank(cartesian_communicator, &pe_cartesian_rank);
  MPI_Cart_coords(cartesian_communicator, pe_cartesian_rank, 3,
		  pe_cartesian_coordinates);

  for (n = 0; n < 3; n++) {
    MPI_Cart_shift(cartesian_communicator, n, 1,
		   pe_cartesian_neighbours[BACKWARD] + n,
		   pe_cartesian_neighbours[FORWARD] + n);
  }

  /* Set local number of lattice sites and offsets. */

  for (n = 0; n < 3; n++) {
    n_local[n] = N_total(n) / pe_cartesian_size[n];
  }

  xfac_ = (n_local[Y] + 2*nhalo_)*(n_local[Z] + 2*nhalo_);
  yfac_ = (n_local[Z] + 2*nhalo_);

  initialised_ = 1;

  return;
}

/****************************************************************************
 *
 *  coords_finish
 *
 *  Free Cartesian communicator; reset defaults.
 *
 ****************************************************************************/

void coords_finish(void) {

  int ia;
  assert(initialised_);

  MPI_Comm_free(&cartesian_communicator);
  MPI_Comm_free(&cart_periodic);

  cartesian_communicator = MPI_COMM_NULL;
  cart_periodic = MPI_COMM_NULL;

  for (ia = 0; ia < 3; ia++) {
    ntotal_[ia] = 64;
    pe_cartesian_size[ia] = 1;
    periodic[ia] = 1;
  }
  initialised_ = 0;

  return;
}

/*****************************************************************************
 *
 *  coords_info()
 *
 *****************************************************************************/

void coords_info(void) {

  info("System size:    %d %d %d\n", ntotal_[X], ntotal_[Y], ntotal_[Z]);
  info("Decomposition:  %d %d %d\n", cart_size(X), cart_size(Y), cart_size(Z));
  info("Local domain:   %d %d %d\n", n_local[X], n_local[Y], n_local[Z]);
  info("Periodic:       %d %d %d\n", periodic[X], periodic[Y], periodic[Z]);
  info("Halo nhalo:     %d\n", nhalo_);
  info("Reorder:        %s\n", reorder_ ? "true" : "false");
  info("Initialised:    %d\n", initialised_);

  return;
}

/*****************************************************************************
 *
 *  cart_rank access function
 *
 *****************************************************************************/

int cart_rank() {
  return pe_cartesian_rank;
}

/*****************************************************************************
 *
 *  cart_size access function
 *
 *****************************************************************************/

int cart_size(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return pe_cartesian_size[dim];
}

/*****************************************************************************
 *
 *  cart_coords access function
 *
 *****************************************************************************/

int cart_coords(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return pe_cartesian_coordinates[dim];
}

/*****************************************************************************
 *
 *  cart_neighb access function
 *
 *****************************************************************************/

int cart_neighb(const int dir, const int dim) {
  assert(dir == FORWARD || dir == BACKWARD);
  assert(dim == X || dim == Y || dim == Z);
  return pe_cartesian_neighbours[dir][dim];
}

/*****************************************************************************
 *
 *  Cartesian communicator
 *
 *****************************************************************************/

MPI_Comm cart_comm() {
  return cartesian_communicator;
}

/*****************************************************************************
 *
 *  N_total access function
 *
 *****************************************************************************/

int N_total(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return ntotal_[dim];
}

/*****************************************************************************
 *
 *  is_periodic
 *
 *****************************************************************************/

int is_periodic(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return periodic[dim];
}

/*****************************************************************************
 *
 *  L access function
 *
 *****************************************************************************/

double L(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return ((double) ntotal_[dim]);
}

/*****************************************************************************
 *
 *  Lmin access function
 *
 *****************************************************************************/

double Lmin(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return lmin[dim];
}

/*****************************************************************************
 *
 *  coords_nlocal
 *
 *  These quantities are used in performance-critical regions, so
 *  the strategy at the moment is to unload the 3-vector into a
 *  local array via these functions when required.
 *
 *****************************************************************************/

void coords_nlocal(int n[3]) {

  int ia;
  assert(initialised_);

  for (ia = 0; ia < 3; ia++) {
    n[ia] = n_local[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  coords_nsites
 *
 *  Return the total number of lattice sites, including the
 *  halo regions.
 *
 *****************************************************************************/

int coords_nsites(void) {

  int nsites;

  assert(initialised_);

  nsites = (n_local[X] + 2*nhalo_)*(n_local[Y] + 2*nhalo_)
    *(n_local[Z] + 2*nhalo_);

  return nsites;
}

/*****************************************************************************
 *
 *  coords_nlocal_offset
 *
 *  For the local domain, return the location of the first latttice
 *  site in the global domain.
 *
 *****************************************************************************/

void coords_nlocal_offset(int n[3]) {

  int i;
  assert(initialised_);

  for (i = 0; i < 3; i++) {
    n[i] = pe_cartesian_coordinates[i]*n_local[i];
  }

  return;
}

/*****************************************************************************
 *
 *  default_decomposition
 *
 *  This does not, at the moment, take account of the system size,
 *  so user-defined decomposition prefered for high aspect ratio systems.
 *
 *****************************************************************************/

static void default_decomposition() {

  int pe0[3] = {0, 0, 0};

  /* Trap 2-d systems */
  if (ntotal_[X] == 1) pe0[X] = 1;
  if (ntotal_[Y] == 1) pe0[Y] = 1;
  if (ntotal_[Z] == 1) pe0[Z] = 1;

  MPI_Dims_create(pe_size(), 3, pe0);

  pe_cartesian_size[X] = pe0[X];
  pe_cartesian_size[Y] = pe0[Y];
  pe_cartesian_size[Z] = pe0[Z];
  
  if (is_ok_decomposition() == 0) {
    fatal("No default decomposition available!\n");
  }

  return;
}

/*****************************************************************************
 *
 *  is_ok_decomposition
 *
 *  Sanity check for the current processor / lattice decomposition.
 *
 *****************************************************************************/

static int is_ok_decomposition() {

  int ok = 1;
  int nnodes;

  if (N_total(X) % pe_cartesian_size[X]) ok = 0;
  if (N_total(Y) % pe_cartesian_size[Y]) ok = 0;
  if (N_total(Z) % pe_cartesian_size[Z]) ok = 0;

  /*  The Cartesian decomposition must use all processors in COMM_WORLD. */
  nnodes = pe_cartesian_size[X]*pe_cartesian_size[Y]*pe_cartesian_size[Z];
  if (nnodes != pe_size()) ok = 0;

  return ok;
}

/*****************************************************************************
 *
 *  coords_index
 *
 *  Compute the one-dimensional index from coordinates ic, jc, kc.
 *
 *****************************************************************************/

int coords_index(const int ic, const int jc, const int kc) {


  assert(initialised_);
  assert(ic >= 1-nhalo_);
  assert(jc >= 1-nhalo_);
  assert(kc >= 1-nhalo_);
  assert(ic <= n_local[X] + nhalo_);
  assert(jc <= n_local[Y] + nhalo_);
  assert(kc <= n_local[Z] + nhalo_);

  return (xfac_*(nhalo_ + ic - 1) + yfac_*(nhalo_ + jc -1) + nhalo_ + kc - 1);
}

/*****************************************************************************
 *
 *  coords_nhalo_set
 *
 *****************************************************************************/

void coords_nhalo_set(const int n) {

  assert(n > 0);
  assert(initialised_ == 0);

  nhalo_ = n;
  return;
}

/*****************************************************************************
 *
 *  coords_nhalo
 *
 *****************************************************************************/

int coords_nhalo(void) {

  return nhalo_;
}

/*****************************************************************************
 *
 *  coords_ntotal
 *
 *****************************************************************************/

int coords_ntotal(int ntotal[3]) {

  assert(ntotal);

  ntotal[X] = ntotal_[X];
  ntotal[Y] = ntotal_[Y];
  ntotal[Z] = ntotal_[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_ntotal_set
 *
 *****************************************************************************/

void coords_ntotal_set(const int ntotal[3]) {

  int ia;
  assert(initialised_ == 0);

  for (ia = 0; ia < 3; ia++) {
    ntotal_[ia] = ntotal[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  coords_periodicity_set
 *
 *****************************************************************************/

void coords_periodicity_set(const int p[3]) {

  int ia;
  assert(initialised_ == 0);

  for (ia = 0; ia < 3; ia++) {
    periodic[ia] = p[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  coords_decomposition_set
 *
 *****************************************************************************/

void coords_decomposition_set(const int input[3]) {

  int ia;
  assert(initialised_ == 0);

  for (ia = 0; ia < 3; ia++) {
    pe_cartesian_size[ia] = input[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  coords_reorder_set
 *
 *****************************************************************************/

void coords_reorder_set(const int reorder_in) {

  assert(initialised_ == 0);

  reorder_ = reorder_in;

  return;
}

/*****************************************************************************
 *
 *  coords_minimum_distance
 *
 *  Returns the minimum image separation r1 -> r2 (in that direction)
 *  in periodic boundary conditions.
 *
 *****************************************************************************/

void coords_minimum_distance(const double r1[3], const double r2[3],
			     double r12[3]) {
  int ia;

  for (ia = 0; ia < 3; ia++) {
    r12[ia] = r2[ia] - r1[ia];
    if (r12[ia] >  0.5*ntotal_[ia]) r12[ia] -= 1.0*ntotal_[ia]*periodic[ia];
    if (r12[ia] < -0.5*ntotal_[ia]) r12[ia] += 1.0*ntotal_[ia]*periodic[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  coords_index_to_ijk
 *
 *  For given local index, return the corresponding local (ic,jc,kc)
 *  coordinates.
 *
 *****************************************************************************/

void coords_index_to_ijk(const int index, int coords[3]) {

  coords[X] = (1 - nhalo_) + index / xfac_;
  coords[Y] = (1 - nhalo_) + (index % xfac_) / yfac_;
  coords[Z] = (1 - nhalo_) + index % yfac_;

  assert(coords_index(coords[X], coords[Y], coords[Z]) == index);

  return;
}

/*****************************************************************************
 *
 *  coords_strides
 *
 *****************************************************************************/

int coords_strides(int * xs, int * ys, int * zs) {

  *xs = xfac_;
  *ys = yfac_;
  *zs = 1;

  return 0;
}

/*****************************************************************************
 *
 *  coords_active_region_radius_set
 *
 *****************************************************************************/

void coords_active_region_radius_set(const double r) {

  radius_ = r;
  return;
}

/*****************************************************************************
 *
 *  coords_active_region
 *
 *  Returns 1 in the 'region' and zero outside. The 'region' is a
 *  spherical volume of radius radius_, centred at the centre of
 *  the grid.
 *
 *****************************************************************************/

double coords_active_region(const int index) {

  int noffset[3];
  int coords[3];

  double x, y, z;
  double active;

  coords_nlocal_offset(noffset);
  coords_index_to_ijk(index, coords);

  x = 1.0*(noffset[X] + coords[X]) - (Lmin(X) + 0.5*L(X));
  y = 1.0*(noffset[Y] + coords[Y]) - (Lmin(Y) + 0.5*L(Y));
  z = 1.0*(noffset[Z] + coords[Z]) - (Lmin(Z) + 0.5*L(Z));

  active = 1.0;
  if ((x*x + y*y + z*z) > radius_*radius_) active = 0.0;

  return active;
}

/*****************************************************************************
 *
 *  coords_periodic_comm
 *
 *****************************************************************************/

int coords_periodic_comm(MPI_Comm * comm) {

  assert(initialised_);
  assert(comm);

  *comm = cart_periodic;

  return 0;
}

/*****************************************************************************
 *
 *  coords_cart_shift
 *
 *  A convenience to work out the rank of Cartesian neighbours in one
 *  go from the Cartesian communicator.
 *
 *  The shift is always +/- 1 in the given direction.
 *
 *****************************************************************************/

int coords_cart_shift(MPI_Comm comm, int dim, int direction, int * rank) {

  int crank;       /* Rank in the Cartesian communicator */
  int shift;       /* Shift is +/- 1 in direction dim */
  int coords[3];   /* Cartesian coordinates this rank */

  assert(initialised_);
  assert(dim == X || dim == Y || dim == Z);
  assert(direction == FORWARD || direction == BACKWARD);
  assert(rank);

  MPI_Comm_rank(comm, &crank);
  MPI_Cart_coords(comm, crank, 3, coords);

  if (direction == FORWARD) shift = +1;
  if (direction == BACKWARD) shift = -1;

  MPI_Cart_shift(comm, dim, shift, coords, rank);

  return 0;
}
