/*****************************************************************************
 *
 *  coords.c
 *
 *  The physical coordinate system and the MPI Cartesian Communicator.
 *
 *  $Id: coords.c,v 1.3.16.2 2009-11-13 14:33:50 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2008)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"

/* The effective state here is, with default values: */

static int n_total[3]                  = {64, 64, 64};
static int periodic[3]                 = {1, 1, 1};
static int pe_cartesian_size[3]        = {1, 1, 1};
static MPI_Comm cartesian_communicator = MPI_COMM_NULL;
static int reorder_                    = 1;
static int initialised_                = 0;

/* PENDING TODO: remove global scope */
int nhalo_ = 1;

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
  int tmp;
  int iperiodic[3];

  assert(initialised_ == 0);

  /* Set up the communicator and the Cartesian neighbour lists */

  for (n = 0; n < 3; n++) {
    iperiodic[n] = is_periodic(n);
  }

  if (is_ok_decomposition()) {
    /* The user decomposition is selected */
  }
  else {
    /* Look for a default */
    default_decomposition();
  }

  MPI_Cart_create(MPI_COMM_WORLD, 3, pe_cartesian_size, iperiodic, reorder_,
		  &cartesian_communicator);
  MPI_Comm_rank(cartesian_communicator, &pe_cartesian_rank);
  MPI_Cart_coords(cartesian_communicator, pe_cartesian_rank, 3,
		  pe_cartesian_coordinates);

  /* We use a temporary here, as MPI_Cart_shift can change the actual
   * argument for the rank of the source of the recieve, but we only
   * want destination of send. */

  for (n = 0; n < 3; n++) {
    tmp = pe_cartesian_coordinates[n];
    MPI_Cart_shift(cartesian_communicator, n, -1, &tmp,
		   &pe_cartesian_neighbours[BACKWARD][n]);
    tmp = pe_cartesian_coordinates[n];
    MPI_Cart_shift(cartesian_communicator, n, +1, &tmp,
		   &pe_cartesian_neighbours[FORWARD][n]);
  }

  /* Set local number of lattice sites and offsets. */

  for (n = 0; n < 3; n++) {
    n_local[n] = N_total(n) / pe_cartesian_size[n];
  }

  xfac_ = (n_local[Y] + 2*nhalo_)*(n_local[Z] + 2*nhalo_);
  yfac_ = (n_local[Z] + 2*nhalo_);

  info("\n");
  info("Initialised coordinate system:\n");
  info("Actual system size:     %d %d %d\n",
       n_total[X], n_total[Y], n_total[Z]);
  info("Actual decomposition:   %d %d %d\n",
       cart_size(X), cart_size(Y), cart_size(Z));
  info("Local domain:           %d %d %d\n",
       n_local[X], n_local[Y], n_local[Z]);
  info("Periodic:               %d %d %d\n",
       iperiodic[X], iperiodic[Y], iperiodic[Z]);
  info("Width of halo region:   %d\n", nhalo_);
  info("Reorder:                %s\n\n", reorder_ ? "true" : "false");

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

  for (ia = 0; ia < 3; ia++) {
    n_total[ia] = 64;
    pe_cartesian_size[ia] = 1;
    periodic[ia] = 1;
  }
  initialised_ = 0;

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
  return n_total[dim];
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
  return ((double) n_total[dim]);
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
 *  get_N_local
 *
 *  These quantities are used in performance-critical regions, so
 *  the strategy at the moment is to unload the 3-vector into a
 *  local array via these functions when required.
 *
 *****************************************************************************/

void get_N_local(int n[]) {

  int i;
  assert(initialised_);

  for (i = 0; i < 3; i++) {
    n[i] = n_local[i];
  }

  return;
}

/*****************************************************************************
 *
 *  get_N_offset
 *
 *****************************************************************************/

void get_N_offset(int n[]) {

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
  if (n_total[X] == 1) pe0[X] = 1;
  if (n_total[Y] == 1) pe0[Y] = 1;
  if (n_total[Z] == 1) pe0[Z] = 1;

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
 *  get_site_index
 *
 *  Compute the one-dimensional index from coordinates ic, jc, kc.
 *
 *****************************************************************************/

int get_site_index(const int ic, const int jc, const int kc) {

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
 *  coords_ntotal_set
 *
 *****************************************************************************/

void coords_ntotal_set(const int ntotal[3]) {

  int ia;
  assert(initialised_ == 0);

  for (ia = 0; ia < 3; ia++) {
    n_total[ia] = ntotal[ia];
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
