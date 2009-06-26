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
 *  $Id: model.c,v 1.15 2009-06-26 08:44:33 kevin Exp $
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
#include "io_harness.h"
#include "timer.h"

const double cs2  = (1.0/3.0);
const double rcs2 = 3.0;

struct io_info_t * io_info_distribution_; 
Site  * site;

static int nsites_ = 0;
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

MPI_Datatype DT_Site; /* currently referenced in model_le.c */
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

  MPI_Type_vector(nx*ny, nhalolocal, nz, DT_Site, &plane_xy_full_);
  MPI_Type_commit(&plane_xy_full_);

  MPI_Type_vector(nx, nz*nhalolocal, ny*nz, DT_Site, &plane_xz_full_);
  MPI_Type_commit(&plane_xz_full_);

  MPI_Type_vector(1, ny*nz*nhalolocal, 1, DT_Site, &plane_yz_full_);
  MPI_Type_commit(&plane_yz_full_);


  distribution_mpi_init();
  distribution_io_info_init();
  initialised_ = 1;

  distribution_halo_set_complete();

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
  int ndist = 2;
  int nlocal[3];
  int nx, ny, nz;
  int * blocklen;
  MPI_Aint * disp_fwd;
  MPI_Aint * disp_bwd;
  MPI_Datatype * types;

  assert(ndist == 2);

  get_N_local(nlocal);
  nx = nlocal[X] + 2*nhalo_;
  ny = nlocal[Y] + 2*nhalo_;
  nz = nlocal[Z] + 2*nhalo_;

  /* X direction */

  count = ndist*CVXBLOCK + 2;

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

  count = ndist*CVYBLOCK + 2;

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

  count = ndist*CVZBLOCK + 2;

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
  const int ndist = 2;

  assert(ndist == 2);

  block[0] = 1; /* For MPI_LB */

  for (n = 0; n < ndist; n++) {
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
 *  This must handle the number of distributions (currently f, g)
 *
 *****************************************************************************/

static void distribution_set_displacements(const int ndisp,
					   MPI_Aint * disp,
					   const int nbasic,
					   const int * disp_basic) {
  int p;
  MPI_Aint disp0, disp1;

  MPI_Address(&site[0].f[0], &disp0);

  disp[0] = 0; /* For MPI_LB */

  for (p = 0; p < nbasic; p++) {
    MPI_Address(&site[0].f[disp_basic[p]], &disp1);
    disp[1 + p] = disp1 - disp0;
    MPI_Address(&site[0].g[disp_basic[p]], &disp1);
    disp[1 + nbasic + p] = disp1 - disp0;
  }

  disp[ndisp - 1] = sizeof(Site); /* For MPI_UB */

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
  free(site);

  MPI_Type_free(&DT_Site);
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

  sprintf(string, "2 x Distribution: d%dq%d", ND, NVEL);

  io_info_set_name(io_info_distribution_, string);
  io_info_set_read(io_info_distribution_, distributions_read);
  io_info_set_write(io_info_distribution_, distributions_write);
  io_info_set_bytesize(io_info_distribution_, sizeof(Site));

  io_write_metadata("config", io_info_distribution_);

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
  int    p;

  assert(index >= 0 || index < nsites_);

  rho = 0.0;

  for (p = 0; p < NVEL; p++) {
    rho += site[index].f[p];
  }

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

  double phi;
  int    p;

  assert(index >= 0 || index < nsites_);

  phi = 0.0;

  for (p = 0; p < NVEL; p++) {
    phi += site[index].g[p];
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

  int i, p;

  assert(index >= 0 || index < nsites_);

  for (i = 0; i < ND; i++) {
    rhou[i] = 0.0;
  }

  for (p = 0; p < NVEL; p++) {
    for (i = 0; i < ND; i++) {
      rhou[i] += site[index].f[p]*cv[p][i];
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

  const int tagf = 900;
  const int tagb = 901;

  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm = cart_comm();

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
    MPI_Irecv(&site[ihalo].f[0], 1, plane_yz_[BACKWARD],
	      cart_neighb(FORWARD,X), tagb, comm, &request[0]);
    ihalo = get_site_index(0, 1-nhalo_, 1-nhalo_);
    MPI_Irecv(&site[ihalo].f[0], 1, plane_yz_[FORWARD],
	      cart_neighb(BACKWARD,X), tagf, comm, &request[1]);
    ireal = get_site_index(1, 1-nhalo_, 1-nhalo_);
    MPI_Issend(&site[ireal].f[0], 1, plane_yz_[BACKWARD],
	       cart_neighb(BACKWARD,X), tagb, comm, &request[2]);
    ireal = get_site_index(N[X], 1-nhalo_, 1-nhalo_);
    MPI_Issend(&site[ireal].f[0], 1, plane_yz_[FORWARD],
	       cart_neighb(FORWARD,X), tagf, comm, &request[3]);
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
    MPI_Irecv(site[ihalo].f, 1, plane_xz_[BACKWARD],
	      cart_neighb(FORWARD,Y), tagb, comm, &request[0]);
    ihalo = get_site_index(1-nhalo_, 0, 1-nhalo_);
    MPI_Irecv(site[ihalo].f, 1, plane_xz_[FORWARD], cart_neighb(BACKWARD,Y),
	      tagf, comm, &request[1]);
    ireal = get_site_index(1-nhalo_, 1, 1-nhalo_);
    MPI_Issend(site[ireal].f, 1, plane_xz_[BACKWARD], cart_neighb(BACKWARD,Y),
	       tagb, comm, &request[2]);
    ireal = get_site_index(1-nhalo_, N[Y], 1-nhalo_);
    MPI_Issend(site[ireal].f, 1, plane_xz_[FORWARD], cart_neighb(FORWARD,Y),
	       tagf, comm, &request[3]);
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
    MPI_Irecv(site[ihalo].f, 1, plane_xy_[BACKWARD], cart_neighb(FORWARD,Z),
	      tagb, comm, &request[0]);
    ihalo = get_site_index(1-nhalo_, 1-nhalo_, 0);
    MPI_Irecv(site[ihalo].f, 1, plane_xy_[FORWARD], cart_neighb(BACKWARD,Z),
	      tagf, comm, &request[1]);
    ireal = get_site_index(1-nhalo_, 1-nhalo_, 1);
    MPI_Issend(site[ireal].f, 1, plane_xy_[BACKWARD], cart_neighb(BACKWARD,Z),
	       tagb, comm, &request[2]);
    ireal = get_site_index(1-nhalo_, 1-nhalo_, N[Z]);
    MPI_Issend(site[ireal].f, 1, plane_xy_[FORWARD], cart_neighb(FORWARD,Z),
	       tagf, comm, &request[3]);  
    MPI_Waitall(4, request, status);
  }
 
  TIMER_stop(TIMER_HALO_LATTICE);

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

  index = get_site_index(ic, jc, kc);

  n = fread(site + index, sizeof(Site), 1, fp);

  if (n != 1) fatal("fread(distribution) failed at %d %d %d\n", ic, jc,kc);

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

  index = get_site_index(ic, jc, kc);

  n = fwrite(site + index, sizeof(Site), 1, fp);

  if (n != 1) fatal("fwrite(distribution) failde at %d %d %d\n", ic, jc, kc);

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
