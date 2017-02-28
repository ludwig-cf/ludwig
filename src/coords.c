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
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "coords_s.h"

static __host__ int cs_default_decomposition(cs_t * cs);
static __host__ int cs_is_ok_decomposition(cs_t * cs);

static __constant__ cs_param_t const_param;

/*****************************************************************************
 *
 *  cs_create
 *
 *****************************************************************************/

__host__ int cs_create(pe_t * pe, cs_t ** pcs) {

  cs_t * cs = NULL;

  assert(pe);
  assert(pcs);

  cs = (cs_t *) calloc(1, sizeof(cs_t));
  if (cs == NULL) pe_fatal(pe, "calloc(cs_t) failed\n");
  cs->param = (cs_param_t *) calloc(1, sizeof(cs_param_t));
  if (cs->param == NULL) pe_fatal(pe, "calloc(cs_param_t) failed\n");

  cs->pe = pe;
  pe_retain(cs->pe);

  /* Default values for non-zero quatities. */

  cs->param->ntotal[X]   = 64;
  cs->param->ntotal[Y]   = 64;
  cs->param->ntotal[Z]   = 64;
  cs->param->periodic[X] = 1;
  cs->param->periodic[Y] = 1;
  cs->param->periodic[Z] = 1;

  cs->param->mpi_cartsz[X] = 1;
  cs->param->mpi_cartsz[Y] = 1;
  cs->param->mpi_cartsz[Z] = 1;

  cs->param->nhalo = 1;
  cs->reorder = 1;
  cs->commcart = MPI_COMM_NULL;
  cs->commperiodic = MPI_COMM_NULL;
  cs->param->lmin[X] = 0.5;
  cs->param->lmin[Y] = 0.5;
  cs->param->lmin[Z] = 0.5;

  cs->nref = 1;
  *pcs = cs;

  return 0;
}

/*****************************************************************************
 *
 *  cs_retain
 *
 *****************************************************************************/

__host__ int cs_retain(cs_t * cs) {

  assert(cs);

  cs->nref += 1;

  return 0;
}

/*****************************************************************************
 *
 *  cs_free
 *
 *****************************************************************************/

__host__ int cs_free(cs_t * cs) {

  assert(cs);

  cs->nref -= 1;

  if (cs->nref <= 0) {

    if (cs->target != cs) targetFree(cs->target);

    MPI_Comm_free(&cs->commcart);
    MPI_Comm_free(&cs->commperiodic);
    pe_free(cs->pe);
    free(cs->param);
    free(cs);
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
 *  cs_init
 *
 *****************************************************************************/

__host__ int cs_init(cs_t * cs) {

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

  MPI_Cart_create(comm, 3, cs->param->mpi_cartsz, iperiodic, cs->reorder,
		  &cs->commperiodic);

  /* Set up the communicator and the Cartesian neighbour lists for
   * the requested communicator. */

  iperiodic[X] = cs->param->periodic[X];
  iperiodic[Y] = cs->param->periodic[Y];
  iperiodic[Z] = cs->param->periodic[Z];

  MPI_Cart_create(comm, 3, cs->param->mpi_cartsz, iperiodic, cs->reorder,
		  &cs->commcart);
  MPI_Comm_rank(cs->commcart, &cs->mpi_cartrank);
  MPI_Cart_coords(cs->commcart, cs->mpi_cartrank, 3, cs->param->mpi_cartcoords);

  for (n = 0; n < 3; n++) {
    MPI_Cart_shift(cs->commcart, n, 1,
		   &(cs->mpi_cart_neighbours[BACKWARD][n]),
		   &(cs->mpi_cart_neighbours[FORWARD][n]));
  }

  /* Set local number of lattice sites and offsets. */

  cs->param->nlocal[X] = cs->param->ntotal[X] / cs->param->mpi_cartsz[X];
  cs->param->nlocal[Y] = cs->param->ntotal[Y] / cs->param->mpi_cartsz[Y];
  cs->param->nlocal[Z] = cs->param->ntotal[Z] / cs->param->mpi_cartsz[Z];

  cs->param->noffset[X] = cs->param->mpi_cartcoords[X]*cs->param->nlocal[X];
  cs->param->noffset[Y] = cs->param->mpi_cartcoords[Y]*cs->param->nlocal[Y];
  cs->param->noffset[Z] = cs->param->mpi_cartcoords[Z]*cs->param->nlocal[Z];

  cs->param->str[Z] = 1;
  cs->param->str[Y] = cs->param->str[Z]*(cs->param->nlocal[Z] + 2*cs->param->nhalo);
  cs->param->str[X] = cs->param->str[Y]*(cs->param->nlocal[Y] + 2*cs->param->nhalo);

  cs->param->nsites = cs->param->str[X]*(cs->param->nlocal[X] + 2*cs->param->nhalo);

  /* Device side */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    cs->target = cs;
  }
  else {
    cs_param_t * tmp;
    targetCalloc((void **) &cs->target, sizeof(cs_t));
    targetConstAddress((void **) &tmp, const_param);
    copyToTarget(&cs->target->param, (const void *) &tmp, sizeof(cs_param_t *));
    cs_commit(cs);
  }

  return 0;
}

/*****************************************************************************
 *
 *  cs_commit
 *
 *****************************************************************************/

__host__ int cs_commit(cs_t * cs) {

  assert(cs);

  copyConstToTarget(&const_param, cs->param, sizeof(cs_param_t));

  return 0;
}

/*****************************************************************************
 *
 *  cs_info
 *
 *****************************************************************************/

__host__ int cs_info(cs_t * cs) {

  assert(cs);

  pe_info(cs->pe, "\n");
  pe_info(cs->pe, "System details\n");
  pe_info(cs->pe, "--------------\n");

  pe_info(cs->pe, "System size:    %d %d %d\n",
	  cs->param->ntotal[X], cs->param->ntotal[Y], cs->param->ntotal[Z]);
  pe_info(cs->pe, "Decomposition:  %d %d %d\n",
	  cs->param->mpi_cartsz[X], cs->param->mpi_cartsz[Y], cs->param->mpi_cartsz[Z]);
  pe_info(cs->pe, "Local domain:   %d %d %d\n",
	  cs->param->nlocal[X], cs->param->nlocal[Y], cs->param->nlocal[Z]);
  pe_info(cs->pe, "Periodic:       %d %d %d\n",
	  cs->param->periodic[X], cs->param->periodic[Y], cs->param->periodic[Z]);
  pe_info(cs->pe, "Halo nhalo:     %d\n", cs->param->nhalo);
  pe_info(cs->pe, "Reorder:        %s\n", cs->reorder ? "true" : "false");
  pe_info(cs->pe, "Initialised:    %d\n", 1);

  return 0;
}

/*****************************************************************************
 *
 *  cs_cartsz
 *
 *****************************************************************************/

__host__ __device__
int cs_cartsz(cs_t * cs, int sz[3]) {

  assert(cs);

  sz[X] = cs->param->mpi_cartsz[X];
  sz[Y] = cs->param->mpi_cartsz[Y];
  sz[Z] = cs->param->mpi_cartsz[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_cart_coords
 *
 *****************************************************************************/

__host__ __device__
int cs_cart_coords(cs_t * cs, int coords[3]) {

  assert(cs);

  coords[X] = cs->param->mpi_cartcoords[X];
  coords[Y] = cs->param->mpi_cartcoords[Y];
  coords[Z] = cs->param->mpi_cartcoords[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_cart_neighb
 *
 *****************************************************************************/

__host__ int cs_cart_neighb(cs_t * cs, int forwback, int dim) {

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

__host__ int cs_cart_comm(cs_t * cs, MPI_Comm * comm) {

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

__host__ __device__
int cs_periodic(cs_t * cs, int periodic[3]) {

  assert(cs);

  periodic[X] = cs->param->periodic[X];
  periodic[Y] = cs->param->periodic[Y];
  periodic[Z] = cs->param->periodic[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_ltot
 *
 *****************************************************************************/

__host__ __device__
int cs_ltot(cs_t * cs, double ltotal[3]) {

  assert(cs);

  ltotal[X] = (double) cs->param->ntotal[X];
  ltotal[Y] = (double) cs->param->ntotal[Y];
  ltotal[Z] = (double) cs->param->ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_lmin
 *
 *****************************************************************************/

__host__ __device__
int cs_lmin(cs_t * cs, double lmin[3]) {

  assert(cs);

  lmin[X] = cs->param->lmin[X];
  lmin[Y] = cs->param->lmin[Y];
  lmin[Z] = cs->param->lmin[Z];

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

__host__ __device__
int cs_nlocal(cs_t * cs, int n[3]) {

  assert(cs);

  n[X] = cs->param->nlocal[X];
  n[Y] = cs->param->nlocal[Y];
  n[Z] = cs->param->nlocal[Z];

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

__host__ __device__
int cs_nsites(cs_t * cs, int * nsites) {

  assert(cs);

  *nsites = cs->param->nsites;

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

__host__ __device__
int cs_nlocal_offset(cs_t * cs, int n[3]) {

  assert(cs);

  n[X] = cs->param->noffset[X];
  n[Y] = cs->param->noffset[Y];
  n[Z] = cs->param->noffset[Z];

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

static __host__ int cs_default_decomposition(cs_t * cs) {

  int pe0[3] = {0, 0, 0};

  assert(cs);

  /* Trap 2-d systems */
  if (cs->param->ntotal[X] == 1) pe0[X] = 1;
  if (cs->param->ntotal[Y] == 1) pe0[Y] = 1;
  if (cs->param->ntotal[Z] == 1) pe0[Z] = 1;

  MPI_Dims_create(pe_mpi_size(cs->pe), 3, pe0);

  cs->param->mpi_cartsz[X] = pe0[X];
  cs->param->mpi_cartsz[Y] = pe0[Y];
  cs->param->mpi_cartsz[Z] = pe0[Z];
  
  if (cs_is_ok_decomposition(cs) == 0) {
    pe_fatal(cs->pe, "No default decomposition available!\n");
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

static __host__ int cs_is_ok_decomposition(cs_t * cs) {

  int ok = 1;
  int nnodes;

  assert(cs);

  if (cs->param->ntotal[X] % cs->param->mpi_cartsz[X]) ok = 0;
  if (cs->param->ntotal[Y] % cs->param->mpi_cartsz[Y]) ok = 0;
  if (cs->param->ntotal[Z] % cs->param->mpi_cartsz[Z]) ok = 0;

  /*  The Cartesian decomposition must use all processors in COMM_WORLD. */
  nnodes = cs->param->mpi_cartsz[X]*cs->param->mpi_cartsz[Y]*cs->param->mpi_cartsz[Z];
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

__host__ __device__
int cs_index(cs_t * cs,  int ic, int jc, int kc) {

  assert(cs);

  assert(ic >= 1 - cs->param->nhalo);
  assert(jc >= 1 - cs->param->nhalo);
  assert(kc >= 1 - cs->param->nhalo);
  assert(ic <= cs->param->nlocal[X] + cs->param->nhalo);
  assert(jc <= cs->param->nlocal[Y] + cs->param->nhalo);
  assert(kc <= cs->param->nlocal[Z] + cs->param->nhalo);

  return (cs->param->str[X]*(cs->param->nhalo + ic - 1) +
	  cs->param->str[Y]*(cs->param->nhalo + jc - 1) +
	  cs->param->str[Z]*(cs->param->nhalo + kc - 1));
}

/*****************************************************************************
 *
 *  cs_nhalo_set
 *
 *****************************************************************************/

__host__ int cs_nhalo_set(cs_t * cs, int nhalo) {

  assert(nhalo > 0);
  assert(cs);

  cs->param->nhalo = nhalo;

  return 0;
}

/*****************************************************************************
 *
 *  cs_nhalo
 *
 *****************************************************************************/

__host__ __device__
int cs_nhalo(cs_t * cs, int * nhalo) {

  assert(cs);
  assert(nhalo);

  *nhalo = cs->param->nhalo;

  return 0;
}

/*****************************************************************************
 *
 *  cs_ntotal
 *
 *****************************************************************************/

__host__ __device__
int cs_ntotal(cs_t * cs, int ntotal[3]) {

  assert(cs);

  ntotal[X] = cs->param->ntotal[X];
  ntotal[Y] = cs->param->ntotal[Y];
  ntotal[Z] = cs->param->ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_ntotal_set
 *
 *****************************************************************************/

__host__ int cs_ntotal_set(cs_t * cs, const int ntotal[3]) {

  assert(cs);

  cs->param->ntotal[X] = ntotal[X];
  cs->param->ntotal[Y] = ntotal[Y];
  cs->param->ntotal[Z] = ntotal[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_periodicity_set
 *
 *****************************************************************************/

__host__ int cs_periodicity_set(cs_t * cs, const int period[3]) {

  assert(cs);

  cs->param->periodic[X] = period[X];
  cs->param->periodic[Y] = period[Y];
  cs->param->periodic[Z] = period[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_decomposition_set
 *
 *****************************************************************************/

__host__ int cs_decomposition_set(cs_t * cs, const int irequest[3]) {

  assert(cs);

  cs->param->mpi_cartsz[X] = irequest[X];
  cs->param->mpi_cartsz[Y] = irequest[Y];
  cs->param->mpi_cartsz[Z] = irequest[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_reorder_set
 *
 *****************************************************************************/

__host__ int cs_reorder_set(cs_t * cs, int reorder) {

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

__host__ __device__
int cs_minimum_distance(cs_t * cs, const double r1[3],
			    const double r2[3],
			    double r12[3]) {

  assert(cs);

  r12[X] = r2[X] - r1[X];
  r12[Y] = r2[Y] - r1[Y];
  r12[Z] = r2[Z] - r1[Z];

  if (r12[X] >  0.5*cs->param->ntotal[X]) r12[X] -= 1.0*cs->param->ntotal[X]*cs->param->periodic[X];
  if (r12[X] < -0.5*cs->param->ntotal[X]) r12[X] += 1.0*cs->param->ntotal[X]*cs->param->periodic[X];
  if (r12[Y] >  0.5*cs->param->ntotal[Y]) r12[Y] -= 1.0*cs->param->ntotal[Y]*cs->param->periodic[Y];
  if (r12[Y] < -0.5*cs->param->ntotal[Y]) r12[Y] += 1.0*cs->param->ntotal[Y]*cs->param->periodic[Y];
  if (r12[Z] >  0.5*cs->param->ntotal[Z]) r12[Z] -= 1.0*cs->param->ntotal[Z]*cs->param->periodic[Z];
  if (r12[Z] < -0.5*cs->param->ntotal[Z]) r12[Z] += 1.0*cs->param->ntotal[Z]*cs->param->periodic[Z];

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

__host__ __device__
int cs_index_to_ijk(cs_t * cs, int index, int coords[3]) {

  assert(cs);

  coords[X] = (1 - cs->param->nhalo) + index / cs->param->str[X];
  coords[Y] = (1 - cs->param->nhalo) + (index % cs->param->str[X]) / cs->param->str[Y];
  coords[Z] = (1 - cs->param->nhalo) + index % cs->param->str[Y];

  assert(cs_index(cs, coords[X], coords[Y], coords[Z]) == index);

  return 0;
}

/*****************************************************************************
 *
 *  cs_strides
 *
 *****************************************************************************/

__host__ __device__
int cs_strides(cs_t * cs, int * xs, int * ys, int * zs) {

  assert(cs);

  *xs = cs->param->str[X];
  *ys = cs->param->str[Y];
  *zs = cs->param->str[Z];

  return 0;
}

/*****************************************************************************
 *
 *  cs_periodic_comm
 *
 *****************************************************************************/

__host__ int cs_periodic_comm(cs_t * cs, MPI_Comm * comm) {

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

__host__ int cs_cart_shift(MPI_Comm comm, int dim, int direction, int * rank) {

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

__host__ int cs_pe_rank(cs_t * cs) {

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

  nall[X] = cs->param->nlocal[X] + 2*cs->param->nhalo;
  nall[Y] = cs->param->nlocal[Y] + 2*cs->param->nhalo;
  nall[Z] = cs->param->nlocal[Z] + 2*cs->param->nhalo;

  return 0;
}

/*****************************************************************************
 *
 *  cs_cart_rank access function
 *
 *****************************************************************************/

__host__ int cs_cart_rank(cs_t * cs) {
  assert(cs);
  return cs->mpi_cartrank;
}
