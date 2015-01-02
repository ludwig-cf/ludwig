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
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "coords.h"

struct coords_ro_s {
  int nhalo;
  /* etc */
};


struct coords_s {
  pe_t * pe;                       /* Retain a reference to pe */
  int nref;                        /* Reference count */

  /* kernel read-only type */
  /* coords_ro_t * host; */
  /* coords_ro_t * target; */

  /* Potential target data */
  int nhalo;                       /* Width of halo region */
  int nsites;                      /* Total sites (incl. halo) */
  int ntotal[3];                   /* System (physical) size */
  int nlocal[3];                   /* Local system size */
  int noffset[3];                  /* Local system offset */
  int str[3];                      /* Memory strides */
  int periodic[3];                 /* Periodic boundary (non-periodic = 0) */
  double lenmin[3];                /* System L_min */

  /* MPI data */
  int mpi_cartsz[3];               /* Cartesian size */
  int reorder;                     /* MPI reorder flag */
  int mpi_cartrank;                /* MPI Cartesian rank */
  int mpi_cartcoords[3];           /* Cartesian coordinates lookup */
  int mpi_cart_neighbours[2][3];   /* Ranks of Cartesian neighbours lookup */

  MPI_Comm commcart;               /* Cartesian communicator */
  MPI_Comm commperiodic;           /* Cartesian periodic communicator */

  coords_t * target;               /* Host pointer to target memory */
};

static void default_decomposition(void);
static int  is_ok_decomposition(void);

static coords_t * cs;

/*****************************************************************************
 *
 *  coords_create
 *
 *****************************************************************************/

int coords_create(pe_t * pe, coords_t ** pcoord) {

  assert(cs == NULL);
  assert(pe);

  cs = (coords_t *) calloc(1, sizeof(coords_t));
  if (cs == NULL) fatal("calloc(coords_t) failed\n");

  cs->pe = pe;
  pe_retain(cs->pe);

  /* Default values for non-zero quatities. */

  cs->ntotal[X]   = 64;
  cs->ntotal[Y]   = 64;
  cs->ntotal[Z]   = 64;
  cs->periodic[X] = 1;
  cs->periodic[Y] = 1;
  cs->periodic[Z] = 1;

  cs->mpi_cartsz[X] = 1;
  cs->mpi_cartsz[Y] = 1;
  cs->mpi_cartsz[Z] = 1;

  cs->nhalo = 1;
  cs->reorder = 1;
  cs->commcart = MPI_COMM_NULL;
  cs->commperiodic = MPI_COMM_NULL;
  cs->lenmin[X] = 0.5; cs->lenmin[Y] = 0.5; cs->lenmin[Z] = 0.5;

  cs->nref = 1;
  *pcoord = cs;

  return 0;
}

/*****************************************************************************
 *
 *  coords_retain
 *
 *****************************************************************************/

int coords_retain(coords_t * coord) {

  assert(coord);

  coord->nref += 1;

  return 0;
}

/*****************************************************************************
 *
 *  coords_free
 *
 *****************************************************************************/

int coords_free(coords_t ** coord) {

  assert(cs); /* Remove static reference */

  (*coord)->nref -= 1;

  if ((*coord)->nref <= 0) {

    MPI_Comm_free(&cs->commcart);
    MPI_Comm_free(&cs->commperiodic);
    pe_free(&cs->pe);

    free(cs);
    cs = NULL;
  }

  *coord = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  coords_commit
 *
 *****************************************************************************/

int coords_commit(coords_t * cs) {

  int n;
  int iperiodic[3] = {1, 1, 1};
  MPI_Comm comm;

  assert(cs);
  pe_mpi_comm(cs->pe, &comm);

  if (is_ok_decomposition()) {
    /* The user decomposition is selected */
  }
  else {
    /* Look for a default */
    default_decomposition();
  }

  /* A communicator which is always periodic: */

  MPI_Cart_create(comm, 3, cs->mpi_cartsz, iperiodic, cs->reorder,
		  &cs->commperiodic);

  /* Set up the communicator and the Cartesian neighbour lists for
   * the requested communicator. */

  iperiodic[X] = cs->periodic[X];
  iperiodic[Y] = cs->periodic[Y];
  iperiodic[Z] = cs->periodic[Z];

  MPI_Cart_create(comm, 3, cs->mpi_cartsz, iperiodic, cs->reorder,
		  &cs->commcart);
  MPI_Comm_rank(cs->commcart, &cs->mpi_cartrank);
  MPI_Cart_coords(cs->commcart, cs->mpi_cartrank, 3, cs->mpi_cartcoords);

  for (n = 0; n < 3; n++) {
    MPI_Cart_shift(cs->commcart, n, 1, &(cs->mpi_cart_neighbours[BACKWARD][n]),
		   &(cs->mpi_cart_neighbours[FORWARD][n]));
  }

  /* Set local number of lattice sites and offsets. */

  cs->nlocal[X] = cs->ntotal[X] / cs->mpi_cartsz[X];
  cs->nlocal[Y] = cs->ntotal[Y] / cs->mpi_cartsz[Y];
  cs->nlocal[Z] = cs->ntotal[Z] / cs->mpi_cartsz[Z];

  cs->noffset[X] = cs->mpi_cartcoords[X]*cs->nlocal[X];
  cs->noffset[Y] = cs->mpi_cartcoords[Y]*cs->nlocal[Y];
  cs->noffset[Z] = cs->mpi_cartcoords[Z]*cs->nlocal[Z];

  cs->str[Z] = 1;
  cs->str[Y] = cs->str[Z]*(cs->nlocal[Z] + 2*cs->nhalo);
  cs->str[X] = cs->str[Y]*(cs->nlocal[Y] + 2*cs->nhalo);

  cs->nsites = cs->str[X]*(cs->nlocal[X] + 2*cs->nhalo);

  return 0;
}

/*****************************************************************************
 *
 *  coords_info
 *
 *****************************************************************************/

int coords_info(coords_t * cs) {

  assert(cs);

  info("System size:    %d %d %d\n",
       cs->ntotal[X], cs->ntotal[Y], cs->ntotal[Z]);
  info("Decomposition:  %d %d %d\n",
       cs->mpi_cartsz[X], cs->mpi_cartsz[Y], cs->mpi_cartsz[Z]);
  info("Local domain:   %d %d %d\n",
       cs->nlocal[X], cs->nlocal[Y], cs->nlocal[Z]);
  info("Periodic:       %d %d %d\n",
       cs->periodic[X], cs->periodic[Y], cs->periodic[Z]);
  info("Halo nhalo:     %d\n", cs->nhalo);
  info("Reorder:        %s\n", cs->reorder ? "true" : "false");
  info("Initialised:    %d\n", 1);

  return 0;
}

/*****************************************************************************
 *
 *  coords_cartsz
 *
 *****************************************************************************/

int coords_cartsz(coords_t * cs, int sz[3]) {

  assert(cs);

  sz[X] = cs->mpi_cartsz[X];
  sz[Y] = cs->mpi_cartsz[Y];
  sz[Z] = cs->mpi_cartsz[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_cart_coords
 *
 *****************************************************************************/

int coords_cart_coords(coords_t * cs, int coords[3]) {

  assert(cs);

  coords[X] = cs->mpi_cartcoords[X];
  coords[Y] = cs->mpi_cartcoords[Y];
  coords[Z] = cs->mpi_cartcoords[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_cart_neighb
 *
 *****************************************************************************/

int coords_cart_neighb(coords_t * cs, int forwback, int dim) {

  int p;

  assert(cs);
  assert(forwback == FORWARD || forwback == BACKWARD);
  assert(dim == X || dim == Y || dim == Z);

  p = cs->mpi_cart_neighbours[forwback][dim];

  return p;
}

/*****************************************************************************
 *
 *  coords_cart_comm
 *
 *****************************************************************************/

int coords_cart_comm(coords_t * cs, MPI_Comm * comm) {

  assert(cs);
  assert(comm);

  *comm = cs->commcart;

  return 0;
}

/*****************************************************************************
 *
 *  coords_periodic
 *
 *****************************************************************************/

int coords_periodic(coords_t * cs, int periodic[3]) {

  assert(cs);

  periodic[X] = cs->periodic[X];
  periodic[Y] = cs->periodic[Y];
  periodic[Z] = cs->periodic[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_ltot
 *
 *****************************************************************************/

int coords_ltot(coords_t * cs, double ltotal[3]) {

  assert(cs);

  ltotal[X] = (double) cs->ntotal[X];
  ltotal[Y] = (double) cs->ntotal[Y];
  ltotal[Z] = (double) cs->ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_lmin
 *
 *****************************************************************************/

int coords_lmin(coords_t * cs, double lmin[3]) {

  assert(cs);

  lmin[X] = cs->lenmin[X];
  lmin[Y] = cs->lenmin[Y];
  lmin[Z] = cs->lenmin[Z];

  return 0;
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

  assert(cs);

  n[X] = cs->nlocal[X];
  n[Y] = cs->nlocal[Y];
  n[Z] = cs->nlocal[Z];

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

int coords_nsites(coords_t * cs, int * nsites) {

  assert(cs);

  *nsites = cs->nsites;

  return 0;
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

  assert(cs);
  n[X] = cs->noffset[X];
  n[Y] = cs->noffset[Y];
  n[Z] = cs->noffset[Z];

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
  if (cs->ntotal[X] == 1) pe0[X] = 1;
  if (cs->ntotal[Y] == 1) pe0[Y] = 1;
  if (cs->ntotal[Z] == 1) pe0[Z] = 1;

  MPI_Dims_create(pe_mpi_size(cs->pe), 3, pe0);

  cs->mpi_cartsz[X] = pe0[X];
  cs->mpi_cartsz[Y] = pe0[Y];
  cs->mpi_cartsz[Z] = pe0[Z];
  
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

  if (cs->ntotal[X] % cs->mpi_cartsz[X]) ok = 0;
  if (cs->ntotal[Y] % cs->mpi_cartsz[Y]) ok = 0;
  if (cs->ntotal[Z] % cs->mpi_cartsz[Z]) ok = 0;

  /*  The Cartesian decomposition must use all processors in COMM_WORLD. */
  nnodes = cs->mpi_cartsz[X]*cs->mpi_cartsz[Y]*cs->mpi_cartsz[Z];
  if (nnodes != pe_mpi_size(cs->pe)) ok = 0;

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
  assert(cs);

  assert(ic >= 1 - cs->nhalo);
  assert(jc >= 1 - cs->nhalo);
  assert(kc >= 1 - cs->nhalo);
  assert(ic <= cs->nlocal[X] + cs->nhalo);
  assert(jc <= cs->nlocal[Y] + cs->nhalo);
  assert(kc <= cs->nlocal[Z] + cs->nhalo);

  return (cs->str[X]*(cs->nhalo + ic - 1) +
	  cs->str[Y]*(cs->nhalo + jc - 1) +
	  cs->str[Z]*(cs->nhalo + kc - 1));
}

/*****************************************************************************
 *
 *  coords_nhalo_set
 *
 *****************************************************************************/

int coords_nhalo_set(coords_t * cs, int nhalo) {

  assert(nhalo > 0);
  assert(cs);

  cs->nhalo = nhalo;

  return 0;
}

/*****************************************************************************
 *
 *  coords_nhalo
 *
 *****************************************************************************/

int coords_nhalo(void) {

  assert(cs);
  return cs->nhalo;
}

/*****************************************************************************
 *
 *  coords_ntotal
 *
 *****************************************************************************/

int coords_ntotal(coords_t * cs, int ntotal[3]) {

  assert(cs);

  ntotal[X] = cs->ntotal[X];
  ntotal[Y] = cs->ntotal[Y];
  ntotal[Z] = cs->ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_ntotal_set
 *
 *****************************************************************************/

int coords_ntotal_set(coords_t * cs, const int ntotal[3]) {

  assert(cs);

  cs->ntotal[X] = ntotal[X];
  cs->ntotal[Y] = ntotal[Y];
  cs->ntotal[Z] = ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_periodicity_set
 *
 *****************************************************************************/

int coords_periodicity_set(coords_t * cs, const int period[3]) {

  assert(cs);

  cs->periodic[X] = period[X];
  cs->periodic[Y] = period[Y];
  cs->periodic[Z] = period[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_decomposition_set
 *
 *****************************************************************************/

int coords_decomposition_set(coords_t * cs, const int irequest[3]) {

  assert(cs);

  cs->mpi_cartsz[X] = irequest[X];
  cs->mpi_cartsz[Y] = irequest[Y];
  cs->mpi_cartsz[Z] = irequest[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_reorder_set
 *
 *****************************************************************************/

int coords_reorder_set(coords_t * cs, int reorder) {

  assert(cs);

  cs->reorder = reorder;

  return 0;
}

/*****************************************************************************
 *
 *  coords_minimum_distance
 *
 *  Returns the minimum image separation r1 -> r2 (in that direction)
 *  in periodic boundary conditions.
 *
 *****************************************************************************/

int coords_minimum_distance(coords_t * cs, const double r1[3],
			    const double r2[3],
			    double r12[3]) {

  assert(cs);

  r12[X] = r2[X] - r1[X];
  r12[Y] = r2[Y] - r1[Y];
  r12[Z] = r2[Z] - r1[Z];

  if (r12[X] >  0.5*cs->ntotal[X]) r12[X] -= 1.0*cs->ntotal[X]*cs->periodic[X];
  if (r12[X] < -0.5*cs->ntotal[X]) r12[X] += 1.0*cs->ntotal[X]*cs->periodic[X];
  if (r12[Y] >  0.5*cs->ntotal[Y]) r12[Y] -= 1.0*cs->ntotal[Y]*cs->periodic[Y];
  if (r12[Y] < -0.5*cs->ntotal[Y]) r12[Y] += 1.0*cs->ntotal[Y]*cs->periodic[Y];
  if (r12[Z] >  0.5*cs->ntotal[Z]) r12[Z] -= 1.0*cs->ntotal[Z]*cs->periodic[Z];
  if (r12[Z] < -0.5*cs->ntotal[Z]) r12[Z] += 1.0*cs->ntotal[Z]*cs->periodic[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_index_to_ijk
 *
 *  For given local index, return the corresponding local (ic,jc,kc)
 *  coordinates.
 *
 *****************************************************************************/

int coords_index_to_ijk(coords_t * cs, int index, int coords[3]) {

  assert(cs);

  coords[X] = (1 - cs->nhalo) + index / cs->str[X];
  coords[Y] = (1 - cs->nhalo) + (index % cs->str[X]) / cs->str[Y];
  coords[Z] = (1 - cs->nhalo) + index % cs->str[Y];

  assert(coords_index(coords[X], coords[Y], coords[Z]) == index);

  return 0;
}

/*****************************************************************************
 *
 *  coords_strides
 *
 *****************************************************************************/

int coords_strides(coords_t * cs, int * xs, int * ys, int * zs) {

  assert(cs);

  *xs = cs->str[X];
  *ys = cs->str[Y];
  *zs = cs->str[Z];

  return 0;
}

/*****************************************************************************
 *
 *  coords_periodic_comm
 *
 *****************************************************************************/

int coords_periodic_comm(coords_t * cs, MPI_Comm * comm) {

  assert(cs);

  *comm = cs->commperiodic;

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
