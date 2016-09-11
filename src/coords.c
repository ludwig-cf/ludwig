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
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "coords.h"

typedef struct coords_param_s cs_param_t;

struct coords_param_s {
  int nhalo;                       /* Width of halo region */
  int nsites;                      /* Total sites (incl. halo) */
  int ntotal[3];                   /* System (physical) size */
  int nlocal[3];                   /* Local system size */
  int noffset[3];                  /* Local system offset */
  int str[3];                      /* Memory strides */
  int periodic[3];                 /* Periodic boundary (non-periodic = 0) */

  int mpi_cartsz[3];               /* Cartesian size */
  int mpi_cartcoords[3];           /* Cartesian coordinates lookup */

  double lmin[3];                  /* System L_min */
};

struct coords_s {
  pe_t * pe;                       /* Retain a reference to pe */
  int nref;                        /* Reference count */

  cs_param_t * cp;                 /* Constants */

  /* Host data */
  int mpi_cartrank;                /* MPI Cartesian rank */
  int reorder;                     /* MPI reorder flag */
  int mpi_cart_neighbours[2][3];   /* Ranks of Cartesian neighbours lookup */

  MPI_Comm commcart;               /* Cartesian communicator */
  MPI_Comm commperiodic;           /* Cartesian periodic communicator */

  cs_t * target;                   /* Host pointer to target memory */
};

static int cs_default_decomposition(cs_t * cs);
static int cs_is_ok_decomposition(cs_t * cs);


static cs_t * stat_ref = NULL;
/* static __constant__ cs_param_t const_param;*/

/*****************************************************************************
 *
 *  cs_create
 *
 *****************************************************************************/

int cs_create(pe_t * pe, cs_t ** pcs) {

  cs_t * cs = NULL;

  assert(pe);
  assert(pcs);

  cs = (cs_t *) calloc(1, sizeof(cs_t));
  if (cs == NULL) fatal("calloc(cs_t) failed\n");
  cs->cp = (cs_param_t *) calloc(1, sizeof(cs_param_t));
  if (cs->cp == NULL) fatal("calloc(cs_param_t) failed\n");

  cs->pe = pe;
  pe_retain(cs->pe);

  /* Default values for non-zero quatities. */

  cs->cp->ntotal[X]   = 64;
  cs->cp->ntotal[Y]   = 64;
  cs->cp->ntotal[Z]   = 64;
  cs->cp->periodic[X] = 1;
  cs->cp->periodic[Y] = 1;
  cs->cp->periodic[Z] = 1;

  cs->cp->mpi_cartsz[X] = 1;
  cs->cp->mpi_cartsz[Y] = 1;
  cs->cp->mpi_cartsz[Z] = 1;

  cs->cp->nhalo = 1;
  cs->reorder = 1;
  cs->commcart = MPI_COMM_NULL;
  cs->commperiodic = MPI_COMM_NULL;
  cs->cp->lmin[X] = 0.5; cs->cp->lmin[Y] = 0.5; cs->cp->lmin[Z] = 0.5;

  cs->nref = 1;
  *pcs = cs;
  stat_ref = cs;

  return 0;
}

/*****************************************************************************
 *
 *  cs_retain
 *
 *****************************************************************************/

int cs_retain(cs_t * cs) {

  assert(cs);

  cs->nref += 1;

  return 0;
}

/*****************************************************************************
 *
 *  cs_free
 *
 *****************************************************************************/

int cs_free(cs_t * cs) {

  assert(cs);

  cs->nref -= 1;

  if (cs->nref <= 0) {
    MPI_Comm_free(&cs->commcart);
    MPI_Comm_free(&cs->commperiodic);
    pe_free(cs->pe);
    free(cs->cp);
    free(cs);
    stat_ref = NULL;
  }

  return 0;
}

/*****************************************************************************
 *
 *  cs_target
 *
 *  Returns host pointer to device copy.
 *
 *****************************************************************************/

__host__ int cs_target(cs_t * cs, cs_t ** target) {

  assert(cs);
  assert(target);

  *target = cs->target;

  return 0;
}

/*****************************************************************************
 *
 *  cs_ref
 *
 *****************************************************************************/

__host__ int cs_ref(cs_t ** ref) {

  assert(stat_ref);
  assert(ref);

  *ref = stat_ref;

  return 0;
}

/*****************************************************************************
 *
 *  cs_init
 *
 *****************************************************************************/

int cs_init(cs_t * cs) {

  int n;
  int ndevice;
  int iperiodic[3] = {1, 1, 1};
  MPI_Comm comm;

  assert(cs);
  pe_mpi_comm(cs->pe, &comm);

  if (cs_is_ok_decomposition(cs)) {
    /* The user decomposition is selected */
  }
  else {
    /* Look for a default */
    cs_default_decomposition(cs);
  }

  /* A communicator which is always periodic: */

  MPI_Cart_create(comm, 3, cs->cp->mpi_cartsz, iperiodic, cs->reorder,
		  &cs->commperiodic);

  /* Set up the communicator and the Cartesian neighbour lists for
   * the requested communicator. */

  iperiodic[X] = cs->cp->periodic[X];
  iperiodic[Y] = cs->cp->periodic[Y];
  iperiodic[Z] = cs->cp->periodic[Z];

  MPI_Cart_create(comm, 3, cs->cp->mpi_cartsz, iperiodic, cs->reorder,
		  &cs->commcart);
  MPI_Comm_rank(cs->commcart, &cs->mpi_cartrank);
  MPI_Cart_coords(cs->commcart, cs->mpi_cartrank, 3, cs->cp->mpi_cartcoords);

  for (n = 0; n < 3; n++) {
    MPI_Cart_shift(cs->commcart, n, 1,
		   &(cs->mpi_cart_neighbours[BACKWARD][n]),
		   &(cs->mpi_cart_neighbours[FORWARD][n]));
  }

  /* Set local number of lattice sites and offsets. */

  cs->cp->nlocal[X] = cs->cp->ntotal[X] / cs->cp->mpi_cartsz[X];
  cs->cp->nlocal[Y] = cs->cp->ntotal[Y] / cs->cp->mpi_cartsz[Y];
  cs->cp->nlocal[Z] = cs->cp->ntotal[Z] / cs->cp->mpi_cartsz[Z];

  cs->cp->noffset[X] = cs->cp->mpi_cartcoords[X]*cs->cp->nlocal[X];
  cs->cp->noffset[Y] = cs->cp->mpi_cartcoords[Y]*cs->cp->nlocal[Y];
  cs->cp->noffset[Z] = cs->cp->mpi_cartcoords[Z]*cs->cp->nlocal[Z];

  cs->cp->str[Z] = 1;
  cs->cp->str[Y] = cs->cp->str[Z]*(cs->cp->nlocal[Z] + 2*cs->cp->nhalo);
  cs->cp->str[X] = cs->cp->str[Y]*(cs->cp->nlocal[Y] + 2*cs->cp->nhalo);

  cs->cp->nsites = cs->cp->str[X]*(cs->cp->nlocal[X] + 2*cs->cp->nhalo);

  /* Device side */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    cs->target = cs;
  }
  else {
    assert(0);
    /* Required. */
  }

  return 0;
}

/*****************************************************************************
 *
 *  cs_info
 *
 *****************************************************************************/

int cs_info(cs_t * cs) {

  assert(cs);

  pe_info(cs->pe, "\n");
  pe_info(cs->pe, "System details\n");
  pe_info(cs->pe, "--------------\n");

  pe_info(cs->pe, "System size:    %d %d %d\n",
	  cs->cp->ntotal[X], cs->cp->ntotal[Y], cs->cp->ntotal[Z]);
  pe_info(cs->pe, "Decomposition:  %d %d %d\n",
	  cs->cp->mpi_cartsz[X], cs->cp->mpi_cartsz[Y], cs->cp->mpi_cartsz[Z]);
  pe_info(cs->pe, "Local domain:   %d %d %d\n",
	  cs->cp->nlocal[X], cs->cp->nlocal[Y], cs->cp->nlocal[Z]);
  pe_info(cs->pe, "Periodic:       %d %d %d\n",
	  cs->cp->periodic[X], cs->cp->periodic[Y], cs->cp->periodic[Z]);
  pe_info(cs->pe, "Halo nhalo:     %d\n", cs->cp->nhalo);
  pe_info(cs->pe, "Reorder:        %s\n", cs->reorder ? "true" : "false");
  pe_info(cs->pe, "Initialised:    %d\n", 1);

  return 0;
}

/*****************************************************************************
 *
 *  cs_cartsz
 *
 *****************************************************************************/

int cs_cartsz(cs_t * cs, int sz[3]) {

  assert(cs);

  sz[X] = cs->cp->mpi_cartsz[X];
  sz[Y] = cs->cp->mpi_cartsz[Y];
  sz[Z] = cs->cp->mpi_cartsz[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_cart_coords
 *
 *****************************************************************************/

int cs_cart_coords(cs_t * cs, int coords[3]) {

  assert(cs);

  coords[X] = cs->cp->mpi_cartcoords[X];
  coords[Y] = cs->cp->mpi_cartcoords[Y];
  coords[Z] = cs->cp->mpi_cartcoords[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_cart_neighb
 *
 *****************************************************************************/

int cs_cart_neighb(cs_t * cs, int forwback, int dim) {

  int p;

  assert(cs);
  assert(forwback == FORWARD || forwback == BACKWARD);
  assert(dim == X || dim == Y || dim == Z);

  p = cs->mpi_cart_neighbours[forwback][dim];

  return p;
}

/*****************************************************************************
 *
 *  cs_cart_comm
 *
 *****************************************************************************/

int cs_cart_comm(cs_t * cs, MPI_Comm * comm) {

  assert(cs);
  assert(comm);

  *comm = cs->commcart;

  return 0;
}

/*****************************************************************************
 *
 *  cs_periodic
 *
 *****************************************************************************/

int cs_periodic(cs_t * cs, int periodic[3]) {

  assert(cs);

  periodic[X] = cs->cp->periodic[X];
  periodic[Y] = cs->cp->periodic[Y];
  periodic[Z] = cs->cp->periodic[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_ltot
 *
 *****************************************************************************/

int cs_ltot(cs_t * cs, double ltotal[3]) {

  assert(cs);

  ltotal[X] = (double) cs->cp->ntotal[X];
  ltotal[Y] = (double) cs->cp->ntotal[Y];
  ltotal[Z] = (double) cs->cp->ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_lmin
 *
 *****************************************************************************/

int cs_lmin(cs_t * cs, double lmin[3]) {

  assert(cs);

  lmin[X] = cs->cp->lmin[X];
  lmin[Y] = cs->cp->lmin[Y];
  lmin[Z] = cs->cp->lmin[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_nlocal
 *
 *  These quantities are used in performance-critical regions, so
 *  the strategy at the moment is to unload the 3-vector into a
 *  local array via these functions when required.
 *
 *****************************************************************************/

int cs_nlocal(cs_t * cs, int n[3]) {

  assert(cs);

  n[X] = cs->cp->nlocal[X];
  n[Y] = cs->cp->nlocal[Y];
  n[Z] = cs->cp->nlocal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_nsites
 *
 *  Return the total number of lattice sites, including the
 *  halo regions.
 *
 *****************************************************************************/

int cs_nsites(cs_t * cs, int * nsites) {

  assert(cs);

  *nsites = cs->cp->nsites;

  return 0;
}

/*****************************************************************************
 *
 *  cs_nlocal_offset
 *
 *  For the local domain, return the location of the first latttice
 *  site in the global domain.
 *
 *****************************************************************************/

int cs_nlocal_offset(cs_t * cs, int n[3]) {

  assert(cs);

  n[X] = cs->cp->noffset[X];
  n[Y] = cs->cp->noffset[Y];
  n[Z] = cs->cp->noffset[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_default_decomposition
 *
 *  This does not, at the moment, take account of the system size,
 *  so user-defined decomposition prefered for high aspect ratio systems.
 *
 *****************************************************************************/

static int cs_default_decomposition(cs_t * cs) {

  int pe0[3] = {0, 0, 0};

  assert(cs);

  /* Trap 2-d systems */
  if (cs->cp->ntotal[X] == 1) pe0[X] = 1;
  if (cs->cp->ntotal[Y] == 1) pe0[Y] = 1;
  if (cs->cp->ntotal[Z] == 1) pe0[Z] = 1;

  MPI_Dims_create(pe_mpi_size(cs->pe), 3, pe0);

  cs->cp->mpi_cartsz[X] = pe0[X];
  cs->cp->mpi_cartsz[Y] = pe0[Y];
  cs->cp->mpi_cartsz[Z] = pe0[Z];
  
  if (cs_is_ok_decomposition(cs) == 0) {
    fatal("No default decomposition available!\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  cs_is_ok_decomposition
 *
 *  Sanity check for the current processor / lattice decomposition.
 *
 *****************************************************************************/

static int cs_is_ok_decomposition(cs_t * cs) {

  int ok = 1;
  int nnodes;

  assert(cs);

  if (cs->cp->ntotal[X] % cs->cp->mpi_cartsz[X]) ok = 0;
  if (cs->cp->ntotal[Y] % cs->cp->mpi_cartsz[Y]) ok = 0;
  if (cs->cp->ntotal[Z] % cs->cp->mpi_cartsz[Z]) ok = 0;

  /*  The Cartesian decomposition must use all processors in COMM_WORLD. */
  nnodes = cs->cp->mpi_cartsz[X]*cs->cp->mpi_cartsz[Y]*cs->cp->mpi_cartsz[Z];
  if (nnodes != pe_mpi_size(cs->pe)) ok = 0;

  return ok;
}

/*****************************************************************************
 *
 *  cs_index
 *
 *  Compute the one-dimensional index from coordinates ic, jc, kc.
 *
 *****************************************************************************/

int cs_index(cs_t * cs,  int ic, int jc, int kc) {

  assert(cs);

  assert(ic >= 1 - cs->cp->nhalo);
  assert(jc >= 1 - cs->cp->nhalo);
  assert(kc >= 1 - cs->cp->nhalo);
  assert(ic <= cs->cp->nlocal[X] + cs->cp->nhalo);
  assert(jc <= cs->cp->nlocal[Y] + cs->cp->nhalo);
  assert(kc <= cs->cp->nlocal[Z] + cs->cp->nhalo);

  return (cs->cp->str[X]*(cs->cp->nhalo + ic - 1) +
	  cs->cp->str[Y]*(cs->cp->nhalo + jc - 1) +
	  cs->cp->str[Z]*(cs->cp->nhalo + kc - 1));
}

/*****************************************************************************
 *
 *  cs_nhalo_set
 *
 *****************************************************************************/

int cs_nhalo_set(cs_t * cs, int nhalo) {

  assert(nhalo > 0);
  assert(cs);

  cs->cp->nhalo = nhalo;

  return 0;
}

/*****************************************************************************
 *
 *  cs_nhalo
 *
 *****************************************************************************/

int cs_nhalo(cs_t * cs, int * nhalo) {

  assert(cs);
  assert(nhalo);

  *nhalo = cs->cp->nhalo;

  return 0;
}

/*****************************************************************************
 *
 *  cs_ntotal
 *
 *****************************************************************************/

int cs_ntotal(cs_t * cs, int ntotal[3]) {

  assert(cs);

  ntotal[X] = cs->cp->ntotal[X];
  ntotal[Y] = cs->cp->ntotal[Y];
  ntotal[Z] = cs->cp->ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_ntotal_set
 *
 *****************************************************************************/

int cs_ntotal_set(cs_t * cs, const int ntotal[3]) {

  assert(cs);

  cs->cp->ntotal[X] = ntotal[X];
  cs->cp->ntotal[Y] = ntotal[Y];
  cs->cp->ntotal[Z] = ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_periodicity_set
 *
 *****************************************************************************/

int cs_periodicity_set(cs_t * cs, const int period[3]) {

  assert(cs);

  cs->cp->periodic[X] = period[X];
  cs->cp->periodic[Y] = period[Y];
  cs->cp->periodic[Z] = period[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_decomposition_set
 *
 *****************************************************************************/

int cs_decomposition_set(cs_t * cs, const int irequest[3]) {

  assert(cs);

  cs->cp->mpi_cartsz[X] = irequest[X];
  cs->cp->mpi_cartsz[Y] = irequest[Y];
  cs->cp->mpi_cartsz[Z] = irequest[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_reorder_set
 *
 *****************************************************************************/

int cs_reorder_set(cs_t * cs, int reorder) {

  assert(cs);

  cs->reorder = reorder;

  return 0;
}

/*****************************************************************************
 *
 *  cs_minimum_distance
 *
 *  Returns the minimum image separation r1 -> r2 (in that direction)
 *  in periodic boundary conditions.
 *
 *****************************************************************************/

int cs_minimum_distance(cs_t * cs, const double r1[3],
			    const double r2[3],
			    double r12[3]) {

  assert(cs);

  r12[X] = r2[X] - r1[X];
  r12[Y] = r2[Y] - r1[Y];
  r12[Z] = r2[Z] - r1[Z];

  if (r12[X] >  0.5*cs->cp->ntotal[X]) r12[X] -= 1.0*cs->cp->ntotal[X]*cs->cp->periodic[X];
  if (r12[X] < -0.5*cs->cp->ntotal[X]) r12[X] += 1.0*cs->cp->ntotal[X]*cs->cp->periodic[X];
  if (r12[Y] >  0.5*cs->cp->ntotal[Y]) r12[Y] -= 1.0*cs->cp->ntotal[Y]*cs->cp->periodic[Y];
  if (r12[Y] < -0.5*cs->cp->ntotal[Y]) r12[Y] += 1.0*cs->cp->ntotal[Y]*cs->cp->periodic[Y];
  if (r12[Z] >  0.5*cs->cp->ntotal[Z]) r12[Z] -= 1.0*cs->cp->ntotal[Z]*cs->cp->periodic[Z];
  if (r12[Z] < -0.5*cs->cp->ntotal[Z]) r12[Z] += 1.0*cs->cp->ntotal[Z]*cs->cp->periodic[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_index_to_ijk
 *
 *  For given local index, return the corresponding local (ic,jc,kc)
 *  coordinates.
 *
 *****************************************************************************/

int cs_index_to_ijk(cs_t * cs, int index, int coords[3]) {

  assert(cs);

  coords[X] = (1 - cs->cp->nhalo) + index / cs->cp->str[X];
  coords[Y] = (1 - cs->cp->nhalo) + (index % cs->cp->str[X]) / cs->cp->str[Y];
  coords[Z] = (1 - cs->cp->nhalo) + index % cs->cp->str[Y];

  assert(cs_index(cs, coords[X], coords[Y], coords[Z]) == index);

  return 0;
}

/*****************************************************************************
 *
 *  cs_strides
 *
 *****************************************************************************/

int cs_strides(cs_t * cs, int * xs, int * ys, int * zs) {

  assert(cs);

  *xs = cs->cp->str[X];
  *ys = cs->cp->str[Y];
  *zs = cs->cp->str[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_periodic_comm
 *
 *****************************************************************************/

int cs_periodic_comm(cs_t * cs, MPI_Comm * comm) {

  assert(cs);

  *comm = cs->commperiodic;

  return 0;
}

/*****************************************************************************
 *
 *  cs_cart_shift
 *
 *  A convenience to work out the rank of Cartesian neighbours in one
 *  go from the Cartesian communicator.
 *
 *  The shift is always +/- 1 in the given direction.
 *
 *****************************************************************************/

int cs_cart_shift(MPI_Comm comm, int dim, int direction, int * rank) {

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

/*****************************************************************************
 *
 *  cs_pe_rank
 *
 *  This can be useful if you do not want side effects from any possible
 *  reordering of Cartesian rank.
 *
 *****************************************************************************/

int cs_pe_rank(cs_t * cs) {

  assert(cs);

  return pe_mpi_rank(cs->pe);
}

/*****************************************************************************
 *
 *  cs_nall
 *
 *****************************************************************************/

__host__ __device__ int cs_nall(cs_t * cs, int nall[3]) {

  assert(cs);

  nall[X] = cs->cp->nlocal[X] + 2*cs->cp->nhalo;
  nall[Y] = cs->cp->nlocal[Y] + 2*cs->cp->nhalo;
  nall[Z] = cs->cp->nlocal[Z] + 2*cs->cp->nhalo;

  return 0;
}

/*****************************************************************************
 *
 * Static interface schuedled for deletion.
 *
 *****************************************************************************/

/*****************************************************************************
 *
 *  coords_info()
 *
 *****************************************************************************/

void coords_info(void) {

  assert(stat_ref);

  cs_info(stat_ref);

  return;
}

/*****************************************************************************
 *
 *  cart_rank access function
 *
 *****************************************************************************/

int cart_rank() {
  assert(stat_ref);
  return stat_ref->mpi_cartrank;
}

/*****************************************************************************
 *
 *  cart_size access function
 *
 *****************************************************************************/

int cart_size(const int dim) {
  assert(stat_ref);
  return stat_ref->cp->mpi_cartsz[dim];
}

/*****************************************************************************
 *
 *  cart_coords access function
 *
 *****************************************************************************/

int cart_coords(const int dim) {
  assert(stat_ref);
  return stat_ref->cp->mpi_cartcoords[dim];
}

/*****************************************************************************
 *
 *  cart_neighb access function
 *
 *****************************************************************************/

int cart_neighb(const int dir, const int dim) {
  assert(stat_ref);
  return stat_ref->mpi_cart_neighbours[dir][dim];
}

/*****************************************************************************
 *
 *  Cartesian communicator
 *
 *****************************************************************************/

MPI_Comm cart_comm() {
  assert(stat_ref);
  return stat_ref->commcart;
}

/*****************************************************************************
 *
 *  N_total access function
 *
 *****************************************************************************/

int N_total(const int dim) {
  assert(stat_ref);
  assert(dim == X || dim == Y || dim == Z);
  return stat_ref->cp->ntotal[dim];
}

/*****************************************************************************
 *
 *  is_periodic
 *
 *****************************************************************************/

int is_periodic(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  assert(stat_ref);
  return stat_ref->cp->periodic[dim];
}

/*****************************************************************************
 *
 *  L access function
 *
 *****************************************************************************/

double L(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  assert(stat_ref);
  return ((double) stat_ref->cp->ntotal[dim]);
}

/*****************************************************************************
 *
 *  Lmin access function
 *
 *****************************************************************************/

double Lmin(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  assert(stat_ref);
  return stat_ref->cp->lmin[dim];
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

  assert(stat_ref);
  cs_nlocal(stat_ref, n);

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

  assert(stat_ref);

  return stat_ref->cp->nsites;
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

  assert(stat_ref);

  cs_nlocal_offset(stat_ref, n);

  return;
}

/*****************************************************************************
 *
 *  coords_index
 *
 *  Compute the one-dimensional index from coordinates ic, jc, kc.
 *
 *****************************************************************************/

int coords_index(const int ic, const int jc, const int kc) {

  assert(stat_ref);

  return cs_index(stat_ref, ic, jc, kc);
}


/*****************************************************************************
 *
 *  coords_nhalo
 *
 *****************************************************************************/

int coords_nhalo(void) {
  assert(stat_ref);
  return stat_ref->cp->nhalo;
}

/*****************************************************************************
 *
 *  coords_ntotal
 *
 *****************************************************************************/

int coords_ntotal(int ntotal[3]) {

  assert(stat_ref);
  cs_ntotal(stat_ref, ntotal);

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

void coords_minimum_distance(const double r1[3], const double r2[3],
			     double r12[3]) {

  assert(stat_ref);
  cs_minimum_distance(stat_ref, r1, r2, r12);
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

  assert(stat_ref);
  cs_index_to_ijk(stat_ref, index, coords);

  return;
}

/*****************************************************************************
 *
 *  coords_strides
 *
 *****************************************************************************/

int coords_strides(int * xs, int * ys, int * zs) {

  assert(stat_ref);
  cs_strides(stat_ref, xs, ys, zs);

  return 0;
}

/*****************************************************************************
 *
 *  coords_nall
 *
 *****************************************************************************/

int coords_nall(int nall[3]) {

  assert(stat_ref);
  cs_nall(stat_ref, nall);

  return 0;
}
