/*****************************************************************************
 *
 *  cartesian.c
 *
 *  The Cartesian communicator.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "cartesian.h"

static void default_decomposition(void);
static int  is_ok_decomposition(void);

#ifdef _MPI_
static MPI_Comm cartesian_communicator;
#endif

static int pe_cartesian_rank             = 0;
static int pe_cartesian_size[3]          = {1, 1, 1};
static int pe_cartesian_coordinates[3]   = {0, 0, 0};
static int pe_cartesian_neighbours[2][3] = {{0, 0, 0}, {0, 0, 0}};

static int n_local[3];
static int n_offset[3];

/*****************************************************************************
 *
 *  cartesian_init
 *
 *  Initialise the decomposition.
 *
 *****************************************************************************/

void cart_init() {

  int n;

#ifdef _MPI_

  int reorder = 1;
  int periodic[3];

  /* Look for a user-defined decomposition */

  n = RUN_get_int_parameter_vector("grid", pe_cartesian_size);

  if (n != 0 && is_ok_decomposition()) {
    /* The user decomposition is selected */
    info("(User) Processor decomposition is (%d, %d, %d)\n",
	 pe_cartesian_size[X], pe_cartesian_size[Y], pe_cartesian_size[Z]);
  }
  else {
    /* Look for a default */
    default_decomposition();
    info("(Default) Processor decomposition is (%d, %d, %d)\n",
	 pe_cartesian_size[X], pe_cartesian_size[Y], pe_cartesian_size[Z]);
  }

  /* Set up the communicator and the Cartesian neighbour lists */

  n = RUN_get_int_parameter("reorder", &reorder);

  for (n = 0; n < 3; n++) {
    periodic[n] = is_periodic(n);
  }

  info("Cartesian Communicator:\n");
  info("Periodic = (%d, %d, %d)\n", periodic[X], periodic[Y], periodic[Z]);
  info("Reorder is %s\n", reorder ? "true" : "false");

  MPI_Cart_create(MPI_COMM_WORLD, 3, pe_cartesian_size, periodic, reorder,
		  &cartesian_communicator);
  MPI_Comm_rank(cartesian_communicator, &pe_cartesian_rank);
  MPI_Cart_coords(cartesian_communicator, pe_cartesian_rank, 3,
		  pe_cartesian_coordinates);

  /* We use reorder as a temporary here, as MPI_Cart_shift can
   * change the actual argument for the rank of the source of
   * the recieve, but we only want destination of send. */

  for (n = 0; n < 3; n++) {
    reorder = pe_cartesian_coordinates[n];
    MPI_Cart_shift(cartesian_communicator, n, -1, &reorder,
		   &pe_cartesian_neighbours[BACKWARD][n]);
    reorder = pe_cartesian_coordinates[n];
    MPI_Cart_shift(cartesian_communicator, n, +1, &reorder,
		   &pe_cartesian_neighbours[FORWARD][n]);
  }

#endif

  /* Set local number of lattice sites and offsets. */

  for (n = 0; n < 3; n++) {
    n_local[n] = N_total(n) / pe_cartesian_size[n];
    n_offset[n] = pe_cartesian_coordinates[n]*n_local[n];
  }

  info("The local domain is (%d, %d, %d)\n",
       n_local[X], n_local[Y], n_local[Z]);

  return;
}

/*****************************************************************************
 *
 *  cart_rank
 *  cart_size
 *  cart_coords
 *  cart_neighb
 *  cart_comm
 *
 *****************************************************************************/

int cart_rank() {
  return pe_cartesian_rank;
}

int cart_size(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return pe_cartesian_size[dim];
}

int cart_coords(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return pe_cartesian_coordinates[dim];
}

int cart_neighb(const int dir, const int dim) {
  assert(dir == FORWARD || dir == BACKWARD);
  assert(dim == X || dim == Y || dim == Z);
  return pe_cartesian_neighbours[dir][dim];
}

#ifdef _MPI_
MPI_Comm cart_comm() {
  return cartesian_communicator;
}
#endif

/*****************************************************************************
 *
 *  N_local
 *  N_offset
 *
 *  These quantities are used in performance-critical regions, so
 *  the strategy at the moment is to unload the 3-vector into a
 *  local array via these functions when required.
 *
 *****************************************************************************/

void get_N_local(int n[]) {
  int i;

  for (i = 0; i < 3; i++) {
    n[i] = n_local[i];
  }

  return;
}

void get_N_offset(int n[]) {
  int i;

  for (i = 0; i < 3; i++) {
    n[i] = n_offset[i];
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

void default_decomposition() {

#ifdef _MPI_
  int pe0[3] = {0, 0, 0};

  MPI_Dims_create(pe_size(), 3, pe0);

  pe_cartesian_size[X] = pe0[X];
  pe_cartesian_size[Y] = pe0[Y];
  pe_cartesian_size[Z] = pe0[Z];
#endif
  
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

int is_ok_decomposition() {

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
