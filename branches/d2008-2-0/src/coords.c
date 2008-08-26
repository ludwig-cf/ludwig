/*****************************************************************************
 *
 *  coords.c
 *
 *  The physical coordinate system and the MPI Cartesian Communicator.
 *
 *  $Id: coords.c,v 1.3 2008-08-24 17:37:59 kevin Exp $
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
#include "runtime.h"
#include "coords.h"

/* Change in halo size requires recompilation. nhalo_ is public. */
const int nhalo_ = 1;

static int n_total[3] = {64, 64, 64};
static int n_local[3];
static int n_offset[3];

static int xfac_;
static int yfac_;

static int periodic[3]                   = {1, 1, 1};
static int pe_cartesian_rank             = 0;
static int pe_cartesian_size[3]          = {1, 1, 1};
static int pe_cartesian_coordinates[3]   = {0, 0, 0};
static int pe_cartesian_neighbours[2][3] = {{0, 0, 0}, {0, 0, 0}};

static double length[3] = {64.0, 64.0, 64.0};
static double lmin[3] = {0.5, 0.5, 0.5};

static void cart_init(void);
static void default_decomposition(void);
static int  is_ok_decomposition(void);

static MPI_Comm cartesian_communicator = MPI_COMM_NULL;

/*****************************************************************************
 *
 *  coords_init
 *
 *  Set the physical lattice size, and periodic flags (always periodic
 *  by default.
 *
 *****************************************************************************/

void coords_init() {

  int n;

  /* Look for "size" in the user input and set the lattice size. */

  n = RUN_get_int_parameter_vector("size", n_total);

  info((n == 0) ? "[Default] " : "[User   ] "); 
  info("Lattice size is (%d, %d, %d)\n", n_total[X], n_total[Y], n_total[Z]);

  for (n = 0; n < 3; n++) {
    length[n] = (double) n_total[n];
  }

  /* Look for the "periodicity" in the user input. (This is
   * not reported at the moment.) */

  n = RUN_get_int_parameter_vector("periodicity", periodic);

  info((n == 0) ? "[Default] " : "[User   ] "); 
  info("periodic boundaries set to (%d, %d, %d)\n", periodic[X], periodic[Y],
       periodic[Z]);

  cart_init();

  return;
}

/*****************************************************************************
 *
 *  cartesian_init
 *
 *  Initialise the decomposition and the Cartesian Communicator.
 *
 *****************************************************************************/

void cart_init() {

  int n;
  int reorder = 1;
  int periodic[3];

  /* Look for a user-defined decomposition */

  n = RUN_get_int_parameter_vector("grid", pe_cartesian_size);

  if (n != 0 && is_ok_decomposition()) {
    /* The user decomposition is selected */
    info("[User   ] Processor decomposition is (%d, %d, %d)\n",
	 pe_cartesian_size[X], pe_cartesian_size[Y], pe_cartesian_size[Z]);
  }
  else {
    /* Look for a default */
    default_decomposition();
    info("[Default] Processor decomposition is (%d, %d, %d)\n",
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

  /* Set local number of lattice sites and offsets. */

  for (n = 0; n < 3; n++) {
    n_local[n] = N_total(n) / pe_cartesian_size[n];
    n_offset[n] = pe_cartesian_coordinates[n]*n_local[n];
  }

  xfac_ = (n_local[Y] + 2*nhalo_)*(n_local[Z] + 2*nhalo_);
  yfac_ = (n_local[Z] + 2*nhalo_);

  info("[Compute] local domain as (%d, %d, %d)\n",
       n_local[X], n_local[Y], n_local[Z]);

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
  return length[dim];
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

  assert(ic >= 1-nhalo_);
  assert(jc >= 1-nhalo_);
  assert(kc >= 1-nhalo_);
  assert(ic <= n_local[X] + nhalo_);
  assert(jc <= n_local[Y] + nhalo_);
  assert(kc <= n_local[Z] + nhalo_);

  return (xfac_*(nhalo_ + ic - 1) + yfac_*(nhalo_ + jc -1) + nhalo_ + kc - 1);
}
