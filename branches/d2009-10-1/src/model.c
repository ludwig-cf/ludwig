/*****************************************************************************
 *
 *  model.c
 *
 *  This encapsulates data/operations related to distributions.
 *  However, the implementation of the distribution is exposed
 *  for performance-critical operations ehich, for space
 *  considerations, are not included in this file.
 *
 *  The LB model is either D2Q9, D3Q15 or D3Q19, as included in model.h.
 *
 *  $Id: model.c,v 1.17.4.10 2010-07-07 10:44:38 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "io_harness.h"
#include "model.h"

const double cs2  = (1.0/3.0);
const double rcs2 = 3.0;

struct io_info_t * io_info_distribution_; 
double * f_;

static int ndist_ = 1;
static int nsite_ = 0;
static int initialised_ = 0;

static MPI_Datatype plane_xy_full_;
static MPI_Datatype plane_xz_full_;
static MPI_Datatype plane_yz_full_;
static MPI_Datatype plane_xy_reduced_[2];
static MPI_Datatype plane_xz_reduced_[2];
static MPI_Datatype plane_yz_reduced_[2];
static MPI_Datatype plane_xy_[2];
static MPI_Datatype plane_xz_[2];
static MPI_Datatype plane_yz_[2];

static MPI_Datatype site_x_[2];
static MPI_Datatype site_y_[2];
static MPI_Datatype site_z_[2];

static void distribution_io_info_init(void);
static void distribution_mpi_init(void);
static void distribution_set_types(const int, MPI_Datatype *);
static void distribution_set_blocks(const int, int *, const int, const int *);
static void distribution_set_displacements(const int, MPI_Aint *, const int,
					   const int *);
static int distributions_read(FILE *, const int, const int, const int);
static int distributions_write(FILE *, const int, const int, const int);

/***************************************************************************
 *
 *  init_site
 *
 *  Irrespective of the value of nhalo associated with coords.c,
 *  we only ever at the moment pass one plane worth of distribution
 *  values. This is nhalolocal.
 *
 ***************************************************************************/
 
void init_site() {

  int nlocal[3];
  int nx, ny, nz;
  int nhalolocal = 1;
  int ndata;
  int nhalo;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nx = nlocal[X] + 2*nhalo;
  ny = nlocal[Y] + 2*nhalo;
  nz = nlocal[Z] + 2*nhalo;
  nsite_ = nx*ny*nz;

  /* The total number of distribution data is then... */

  ndata = nsite_*ndist_*NVEL;

  f_ = (double  *) malloc(ndata*sizeof(double));
  if (f_ == NULL) fatal("malloc(distributions) failed\n");

  /* Set up the MPI Datatypes used for site, and its corresponding
   * halo messages:
   *
   * in XY plane nx*ny blocks of 1 site with stride nz;
   * in XZ plane nx blocks of nz sites with stride ny*nz;
   * in YZ plane one contiguous block of ny*nz sites. */

  MPI_Type_vector(nx*ny, ndist_*NVEL*nhalolocal, ndist_*NVEL*nz, MPI_DOUBLE,
		  &plane_xy_full_);
  MPI_Type_commit(&plane_xy_full_);

  MPI_Type_vector(nx, ndist_*NVEL*nz*nhalolocal, ndist_*NVEL*ny*nz, MPI_DOUBLE,
		  &plane_xz_full_);
  MPI_Type_commit(&plane_xz_full_);

  MPI_Type_vector(1, ndist_*NVEL*ny*nz*nhalolocal, 1, MPI_DOUBLE,
		  &plane_yz_full_);
  MPI_Type_commit(&plane_yz_full_);

  distribution_mpi_init();
  distribution_io_info_init();
  initialised_ = 1;

  distribution_halo_set_complete();
  distribution_init_f();

  return;
}

/*****************************************************************************
 *
 *  distribution_init_f
 *
 *  Fluid uniformly at rest.
 *
 *****************************************************************************/

void distribution_init_f(void) {

  int nlocal[3];
  int ic, jc, kc, index;

  assert(initialised_);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	/* should be rho0 */
	distribution_zeroth_moment_set_equilibrium(index, 0, 1.0);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_mpi_init
 *
 *  Commit the various datatypes required for halo swaps.
 *
 *****************************************************************************/

static void distribution_mpi_init() {

  int count;
  int nlocal[3];
  int nx, ny, nz;
  int * blocklen;
  int nhalo;
  MPI_Aint * disp_fwd;
  MPI_Aint * disp_bwd;
  MPI_Datatype * types;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nx = nlocal[X] + 2*nhalo;
  ny = nlocal[Y] + 2*nhalo;
  nz = nlocal[Z] + 2*nhalo;

  /* X direction */

  count = ndist_*CVXBLOCK + 2;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  distribution_set_types(count, types);
  distribution_set_blocks(count, blocklen, CVXBLOCK, xblocklen_cv);
  distribution_set_displacements(count, disp_fwd, CVXBLOCK, xdisp_fwd_cv);
  distribution_set_displacements(count, disp_bwd, CVXBLOCK, xdisp_bwd_cv);

  MPI_Type_struct(count, blocklen, disp_fwd, types, &site_x_[FORWARD]);
  MPI_Type_struct(count, blocklen, disp_bwd, types, &site_x_[BACKWARD]);
  MPI_Type_commit(&site_x_[FORWARD]);
  MPI_Type_commit(&site_x_[BACKWARD]);

  MPI_Type_contiguous(ny*nz, site_x_[FORWARD], &plane_yz_reduced_[FORWARD]);
  MPI_Type_contiguous(ny*nz, site_x_[BACKWARD], &plane_yz_reduced_[BACKWARD]);
  MPI_Type_commit(&plane_yz_reduced_[FORWARD]);
  MPI_Type_commit(&plane_yz_reduced_[BACKWARD]);

  free(blocklen);
  free(disp_fwd);
  free(disp_bwd);
  free(types);

  /* Y direction */

  count = ndist_*CVYBLOCK + 2;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  distribution_set_types(count, types);
  distribution_set_blocks(count, blocklen, CVYBLOCK, yblocklen_cv);
  distribution_set_displacements(count, disp_fwd, CVYBLOCK, ydisp_fwd_cv);
  distribution_set_displacements(count, disp_bwd, CVYBLOCK, ydisp_bwd_cv);

  MPI_Type_struct(count, blocklen, disp_fwd, types, &site_y_[FORWARD]);
  MPI_Type_struct(count, blocklen, disp_bwd, types, &site_y_[BACKWARD]);
  MPI_Type_commit(&site_y_[FORWARD]);
  MPI_Type_commit(&site_y_[BACKWARD]);

  MPI_Type_vector(nx, nz, ny*nz, site_y_[FORWARD],
		  &plane_xz_reduced_[FORWARD]);
  MPI_Type_vector(nx, nz, ny*nz, site_y_[BACKWARD],
		  &plane_xz_reduced_[BACKWARD]);
  MPI_Type_commit(&plane_xz_reduced_[FORWARD]);
  MPI_Type_commit(&plane_xz_reduced_[BACKWARD]);

  free(blocklen);
  free(disp_fwd);
  free(disp_bwd);
  free(types);

  /* Z direction */

  count = ndist_*CVZBLOCK + 2;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  distribution_set_types(count, types);
  distribution_set_blocks(count, blocklen, CVZBLOCK, zblocklen_cv);
  distribution_set_displacements(count, disp_fwd, CVZBLOCK, zdisp_fwd_cv);
  distribution_set_displacements(count, disp_bwd, CVZBLOCK, zdisp_bwd_cv);

  MPI_Type_struct(count, blocklen, disp_fwd, types, &site_z_[FORWARD]);
  MPI_Type_struct(count, blocklen, disp_bwd, types, &site_z_[BACKWARD]);
  MPI_Type_commit(&site_z_[FORWARD]);
  MPI_Type_commit(&site_z_[BACKWARD]);

  MPI_Type_vector(nx*ny, 1, nz, site_z_[FORWARD], &plane_xy_reduced_[FORWARD]);
  MPI_Type_vector(nx*ny, 1, nz, site_z_[BACKWARD],
		  &plane_xy_reduced_[BACKWARD]);
  MPI_Type_commit(&plane_xy_reduced_[FORWARD]);
  MPI_Type_commit(&plane_xy_reduced_[BACKWARD]);

  free(blocklen);
  free(disp_fwd);
  free(disp_bwd);
  free(types);

  return;
}

/*****************************************************************************
 *
 *  distribution_set_types
 *
 *  Set the type signature for reduced halo struct.
 *
 *****************************************************************************/

static void distribution_set_types(const int ntype, MPI_Datatype * type) {

  int n;

  type[0] = MPI_LB;
  for (n = 1; n < ntype - 1; n++) {
    type[n] = MPI_DOUBLE;
  }
  type[ntype - 1] = MPI_UB;

  return;
}

/*****************************************************************************
 *
 *  distribution_set_blocks
 *
 *  Set the blocklengths for the reduced halo struct.
 *  This means 'expanding' the basic blocks for the current DdQn model
 *  to the current number of distributions.
 *
 *****************************************************************************/

static void distribution_set_blocks(const int nblock, int * block,
				    const int nbasic,
				    const int * basic_block){

  int n, p;

  block[0] = 1; /* For MPI_LB */

  for (n = 0; n < ndist_; n++) {
    for (p = 0; p < nbasic; p++) {
      block[1 + n*nbasic + p] = basic_block[p];
    }
  }

  block[nblock - 1] = 1; /* For MPI_UB */

  return;
}

/*****************************************************************************
 *
 *  distribution_set_displacements
 *
 *  Set the displacements for the reduced halo struct.
 *
 *****************************************************************************/

static void distribution_set_displacements(const int ndisp,
					   MPI_Aint * disp,
					   const int nbasic,
					   const int * disp_basic) {
  int n, p;
  MPI_Aint disp0, disp1;

  MPI_Address(f_, &disp0);

  disp[0] = 0; /* For MPI_LB */

  for (n = 0; n < ndist_; n++) {
    for (p = 0; p < nbasic; p++) {
      MPI_Address(f_ + n*NVEL + disp_basic[p], &disp1);
      disp[1 + n*nbasic + p] = disp1 - disp0;
    }
  }

  /* For MPI_UB */
  MPI_Address(f_ + ndist_*NVEL, &disp1);
  disp[ndisp - 1] = disp1 - disp0;

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

  int n;

  io_info_destroy(io_info_distribution_);
  free(f_);

  MPI_Type_free(&plane_xy_full_);
  MPI_Type_free(&plane_xz_full_);
  MPI_Type_free(&plane_yz_full_);

  for (n = 0; n < 2; n++) {
    MPI_Type_free(plane_xy_reduced_ + n);
    MPI_Type_free(plane_xz_reduced_ + n);
    MPI_Type_free(plane_yz_reduced_ + n);
    MPI_Type_free(site_x_ + n);
    MPI_Type_free(site_y_ + n);
    MPI_Type_free(site_z_ + n);
    plane_xy_[n] = MPI_DATATYPE_NULL;
    plane_xz_[n] = MPI_DATATYPE_NULL;
    plane_yz_[n] = MPI_DATATYPE_NULL;
  }

  initialised_ = 0;
  ndist_ = 1;

  return;
}

/*****************************************************************************
 *
 *  distribution_io_info_init
 *
 *  Initialise the io_info struct for the distributions.
 *
 *****************************************************************************/

static void distribution_io_info_init() {

  int n;
  int io_grid[3];
  char string[FILENAME_MAX];

  n = RUN_get_int_parameter_vector("io_grid_distribution", io_grid);

  if (n == 1) {
    io_info_distribution_ = io_info_create_with_grid(io_grid);
  }
  else {
    io_info_distribution_ = io_info_create();
  }

  sprintf(string, "%1d x Distribution: d%dq%d", ndist_, NDIM, NVEL);

  io_info_set_name(io_info_distribution_, string);
  io_info_set_read(io_info_distribution_, distributions_read);
  io_info_set_write(io_info_distribution_, distributions_write);
  io_info_set_bytesize(io_info_distribution_, ndist_*NVEL*sizeof(double));

  io_write_metadata("config", io_info_distribution_);

  return;
}

/*****************************************************************************
 *
 *  distribution_get_stress_at_site
 *
 *  Return the (deviatoric) stress at index.
 *
 *****************************************************************************/

void distribution_get_stress_at_site(int index, double s[3][3]) {

  int p, ia, ib;

  /* Want to ensure index is ok */
  assert(index >= 0  && index < nsite_);

  for (ia = 0; ia < NDIM; ia++) {
    for (ib = 0; ib < NDIM; ib++) {
      s[ia][ib] = 0.0;
    }
  }

  for (p = 0; p < NVEL; p++) {
    for (ia = 0; ia < NDIM; ia++) {
      for (ib = 0; ib < NDIM; ib++) {
	s[ia][ib] += f_[ndist_*NVEL*index + p]*q_[p][ia][ib];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_halo
 *
 *  Swap the distributions at the periodic/processor boundaries
 *  in each direction.
 *
 *****************************************************************************/

void distribution_halo() {

  int ic, jc, kc;
  int ihalo, ireal;
  int nhalo;
  int nlocal[3];

  const int tagf = 900;
  const int tagb = 901;

  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm = cart_comm();

  assert(initialised_);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  /* The x-direction (YZ plane) */

  if (cart_size(X) == 1) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	ihalo = ndist_*NVEL*coords_index(0, jc, kc);
	ireal = ndist_*NVEL*coords_index(nlocal[X], jc, kc);
	memcpy(f_ + ihalo, f_ + ireal, ndist_*NVEL*sizeof(double));

	ihalo = ndist_*NVEL*coords_index(nlocal[X]+1, jc, kc);
	ireal = ndist_*NVEL*coords_index(1, jc, kc);
	memcpy(f_ + ihalo, f_ + ireal, ndist_*NVEL*sizeof(double));
      }
    }
  }
  else {
    ihalo = ndist_*NVEL*coords_index(nlocal[X] + 1, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(f_ + ihalo, 1, plane_yz_[BACKWARD],
	      cart_neighb(FORWARD,X), tagb, comm, &request[0]);
    ihalo = ndist_*NVEL*coords_index(0, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(f_ + ihalo, 1, plane_yz_[FORWARD],
	      cart_neighb(BACKWARD,X), tagf, comm, &request[1]);
    ireal = ndist_*NVEL*coords_index(1, 1-nhalo, 1-nhalo);
    MPI_Issend(f_ + ireal, 1, plane_yz_[BACKWARD],
	       cart_neighb(BACKWARD,X), tagb, comm, &request[2]);
    ireal = ndist_*NVEL*coords_index(nlocal[X], 1-nhalo, 1-nhalo);
    MPI_Issend(f_ + ireal, 1, plane_yz_[FORWARD],
	       cart_neighb(FORWARD,X), tagf, comm, &request[3]);
    MPI_Waitall(4, request, status);
  }
  
  /* The y-direction (XZ plane) */

  if (cart_size(Y) == 1) {
    for (ic = 0; ic <= nlocal[X] + 1; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	ihalo = ndist_*NVEL*coords_index(ic, 0, kc);
	ireal = ndist_*NVEL*coords_index(ic, nlocal[Y], kc);
	memcpy(f_ + ihalo, f_ + ireal, ndist_*NVEL*sizeof(double));

	ihalo = ndist_*NVEL*coords_index(ic, nlocal[Y] + 1, kc);
	ireal = ndist_*NVEL*coords_index(ic, 1, kc);
	memcpy(f_ + ihalo, f_ + ireal, ndist_*NVEL*sizeof(double));
      }
    }
  }
  else {
    ihalo = ndist_*NVEL*coords_index(1 - nhalo, nlocal[Y] + 1, 1 - nhalo);
    MPI_Irecv(f_ + ihalo, 1, plane_xz_[BACKWARD],
	      cart_neighb(FORWARD,Y), tagb, comm, &request[0]);
    ihalo = ndist_*NVEL*coords_index(1 - nhalo, 0, 1 - nhalo);
    MPI_Irecv(f_ + ihalo, 1, plane_xz_[FORWARD], cart_neighb(BACKWARD,Y),
	      tagf, comm, &request[1]);
    ireal = ndist_*NVEL*coords_index(1 - nhalo, 1, 1 - nhalo);
    MPI_Issend(f_ + ireal, 1, plane_xz_[BACKWARD], cart_neighb(BACKWARD,Y),
	       tagb, comm, &request[2]);
    ireal = ndist_*NVEL*coords_index(1 - nhalo, nlocal[Y], 1 - nhalo);
    MPI_Issend(f_ + ireal, 1, plane_xz_[FORWARD], cart_neighb(FORWARD,Y),
	       tagf, comm, &request[3]);
    MPI_Waitall(4, request, status);
  }
  
  /* Finally, z-direction (XY plane) */

  if (cart_size(Z) == 1) {
    for (ic = 0; ic <= nlocal[X] + 1; ic++) {
      for (jc = 0; jc <= nlocal[Y] + 1; jc++) {

	ihalo = ndist_*NVEL*coords_index(ic, jc, 0);
	ireal = ndist_*NVEL*coords_index(ic, jc, nlocal[Z]);
	memcpy(f_ + ihalo, f_ + ireal, ndist_*NVEL*sizeof(double));

	ihalo = ndist_*NVEL*coords_index(ic, jc, nlocal[Z] + 1);
	ireal = ndist_*NVEL*coords_index(ic, jc, 1);
	memcpy(f_ + ihalo, f_ + ireal, ndist_*NVEL*sizeof(double));
      }
    }
  }
  else {

    ihalo = ndist_*NVEL*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z] + 1);
    MPI_Irecv(f_ + ihalo, 1, plane_xy_[BACKWARD], cart_neighb(FORWARD,Z),
	      tagb, comm, &request[0]);
    ihalo = ndist_*NVEL*coords_index(1 - nhalo, 1 - nhalo, 0);
    MPI_Irecv(f_ + ihalo, 1, plane_xy_[FORWARD], cart_neighb(BACKWARD,Z),
	      tagf, comm, &request[1]);
    ireal = ndist_*NVEL*coords_index(1 - nhalo, 1 - nhalo, 1);
    MPI_Issend(f_ + ireal, 1, plane_xy_[BACKWARD], cart_neighb(BACKWARD,Z),
	       tagb, comm, &request[2]);
    ireal = ndist_*NVEL*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z]);
    MPI_Issend(f_ + ireal, 1, plane_xy_[FORWARD], cart_neighb(FORWARD,Z),
	       tagf, comm, &request[3]);  
    MPI_Waitall(4, request, status);
  }
 
  return;
}

/*****************************************************************************
 *
 *  read_distributions
 *
 *  Read one lattice site (ic, jc, kc) worth of distributions.
 *
 *****************************************************************************/

static int distributions_read(FILE * fp, const int ic, const int jc,
			      const int kc) {

  int index, n;

  index = coords_index(ic, jc, kc);

  n = fread(f_ + ndist_*NVEL*index, sizeof(double), ndist_*NVEL, fp);

  if (n != ndist_*NVEL) {
    fatal("fread(distribution) failed at %d %d %d\n", ic, jc,kc);
  }

  return n;
}

/*****************************************************************************
 *
 *  distributions_write
 *
 *  Write one lattice site (ic, jc, kc) worth of distributions.
 *
 *****************************************************************************/

static int distributions_write(FILE * fp, const int ic , const int jc,
			       const int kc) {

  int index, n;

  index = coords_index(ic, jc, kc);

  n = fwrite(f_ + ndist_*NVEL*index, sizeof(double), ndist_*NVEL, fp);

  if (n != ndist_*NVEL) {
    fatal("fwrite(distribution) failde at %d %d %d\n", ic, jc, kc);
  }

  return n;
}

/*****************************************************************************
 *
 *  distribution_halo_set_complete
 *
 *  Set the actual halo datatype to the full type to swap
 *  all NVEL distributions.
 *
 *****************************************************************************/

void distribution_halo_set_complete(void) {

  assert(initialised_);

  plane_xy_[FORWARD]  = plane_xy_full_;
  plane_xy_[BACKWARD] = plane_xy_full_;
  plane_xz_[FORWARD]  = plane_xz_full_;
  plane_xz_[BACKWARD] = plane_xz_full_;
  plane_yz_[FORWARD]  = plane_yz_full_;
  plane_yz_[BACKWARD] = plane_yz_full_;

  return;
}

/*****************************************************************************
 *
 *  distribution_halo_set_reduced
 *
 *  Set the actual halo datatype to the reduced type to send only the
 *  propagating elements of the distribution in a given direction.
 *
 *****************************************************************************/

void distribution_halo_set_reduced(void) {

  assert(initialised_);

  plane_xy_[FORWARD]  = plane_xy_reduced_[FORWARD];
  plane_xy_[BACKWARD] = plane_xy_reduced_[BACKWARD];
  plane_xz_[FORWARD]  = plane_xz_reduced_[FORWARD];
  plane_xz_[BACKWARD] = plane_xz_reduced_[BACKWARD];
  plane_yz_[FORWARD]  = plane_yz_reduced_[FORWARD];
  plane_yz_[BACKWARD] = plane_yz_reduced_[BACKWARD];

  return;
}

/*****************************************************************************
 *
 *  distribution_ndist
 *
 *  Return the number of distribution functions.
 *
 *****************************************************************************/

int distribution_ndist(void) {

  assert(initialised_);
  return ndist_;
}

/*****************************************************************************
 *
 *  distribution_ndist_set
 *
 *  Set the number of distribution functions to be used.
 *
 *****************************************************************************/

void distribution_ndist_set(const int n) {

  assert(initialised_ == 0);
  assert(n > 0);
  ndist_ = n;

  return;
}

/*****************************************************************************
 *
 *  distribution_f
 *
 *  Get the distribution at site index, velocity p, distribution n.
 *
 *****************************************************************************/

double distribution_f(const int index, const int p, const int n) {

  assert(initialised_);
  assert(index >= 0 && index < nsite_);
  assert(p >= 0 && p < NVEL);
  assert(n >= 0 && n < ndist_);

  return f_[ndist_*NVEL*index + n*NVEL + p];
}

/*****************************************************************************
 *
 *  distribution_f_set
 *
 *  Set the distribution for site index, velocity p, distribution n.
 *
 *****************************************************************************/

void distribution_f_set(const int index, const int p, const int n,
			const double fvalue) {

  assert(initialised_);
  assert(index >= 0 && index < nsite_);
  assert(p >= 0 && p < NVEL);
  assert(n >= 0 && n < ndist_);

  f_[ndist_*NVEL*index + n*NVEL + p] = fvalue;

  return;
}

/*****************************************************************************
 *
 *  distribution_zeroth_moment
 *
 *  Return the zeroth moment of the distribution (rho for n = 0).
 *
 *****************************************************************************/

double distribution_zeroth_moment(const int index, const int n) {

  int p;
  double rho;

  assert(initialised_);
  assert(index >= 0 && index < nsite_);
  assert(n >= 0 && n < ndist_);

  rho = 0.0;

  for (p = 0; p < NVEL; p++) {
    rho += f_[ndist_*NVEL*index + n*NVEL + p];
  }

  return rho;
}

/*****************************************************************************
 *
 *  distribution_first_moment
 *
 *  Return the first moment of the distribution p.
 *
 *****************************************************************************/

void distribution_first_moment(const int index, const int ndist, double g[3]) {

  int p;
  int n;

  assert(initialised_);
  assert(index >= 0 && index < nsite_);
  assert(ndist >= 0 && ndist < ndist_);

  for (n = 0; n < 3; n++) {
    g[n] = 0.0;
  }

  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < NDIM; n++) {
      g[n] += cv[p][n]*f_[ndist_*NVEL*index + ndist*NVEL + p];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_zeroth_moment_set_equilibrium
 *
 *  Project the given value of rho onto the equilibrium distribution
 *  via
 *
 *    f_i = w_i rho
 *
 *****************************************************************************/

void distribution_zeroth_moment_set_equilibrium(const int index, const int n,
						const double rho) {
  int p;

  assert(initialised_);
  assert (n >= 0 && n < ndist_);
  assert(index >= 0 && index < nsite_);

  for (p = 0; p < NVEL; p++) {
    f_[ndist_*NVEL*index + n*NVEL + p] = wv[p]*rho;
  }

  return;
}
