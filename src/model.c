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
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "lb_model_s.h"
#include "targetDP.h"
#include "io_harness.h"

const double cs2  = (1.0/3.0);
const double rcs2 = 3.0;

static int lb_mpi_init(lb_t * lb);
static int lb_set_types(int, MPI_Datatype *);
static int lb_set_blocks(lb_t * lb, int, int *, int, const int *);
static int lb_set_displacements(lb_t * lb, int, MPI_Aint *, int, const int *);
static int lb_f_read(FILE *, int index, void * self);
static int lb_f_write(FILE *, int index, void * self);

__targetConst__ int tc_cv[NVEL][3];
__targetConst__ int tc_ndist;

/****************************************************************************
 *
 *  lb_create_ndist
 *
 *  Allocate memory for lb_data_s only.
 *
 ****************************************************************************/

int lb_create_ndist(int ndist, lb_t ** plb) {

  lb_t * lb = NULL;

  assert(ndist > 0);
  assert(plb);

  lb = (lb_t *) calloc(1, sizeof(lb_t));
  if (lb == NULL) fatal("calloc(1, lb_t) failed\n");

  lb->ndist = ndist;
  lb->model = DATA_MODEL;
  *plb = lb;

  return 0;
}

/*****************************************************************************
 *
 *  lb_create
 *
 *  Default ndist = 1
 *
 *****************************************************************************/

int lb_create(lb_t ** plb) {

  assert(plb);

  return lb_create_ndist(1, plb);
}

/*****************************************************************************
 *
 *  lb_free
 *
 *  Clean up.
 *
 *****************************************************************************/

__host__ void lb_free(lb_t * lb) {

  int ndevice;
  double * tmp;

  assert(lb);

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    copyFromTarget(&tmp, &lb->target->f, sizeof(double *)); 
    targetFree(tmp);

    copyFromTarget(&tmp, &lb->target->fprime, sizeof(double *)); 
    targetFree(tmp);
    targetFree(lb->target);
  }

  if (lb->io_info) io_info_destroy(lb->io_info);
  if (lb->f) free(lb->f);
  if (lb->fprime) free(lb->fprime);

  MPI_Type_free(&lb->plane_xy_full);
  MPI_Type_free(&lb->plane_xz_full);
  MPI_Type_free(&lb->plane_yz_full);

  MPI_Type_free(&lb->plane_xy_reduced[0]);
  MPI_Type_free(&lb->plane_xy_reduced[1]);
  MPI_Type_free(&lb->plane_xz_reduced[0]);
  MPI_Type_free(&lb->plane_xz_reduced[1]);
  MPI_Type_free(&lb->plane_yz_reduced[0]);
  MPI_Type_free(&lb->plane_yz_reduced[1]);

  MPI_Type_free(&lb->site_x[0]);
  MPI_Type_free(&lb->site_x[1]);
  MPI_Type_free(&lb->site_y[0]);
  MPI_Type_free(&lb->site_y[1]);
  MPI_Type_free(&lb->site_z[0]);
  MPI_Type_free(&lb->site_z[1]);

  return;
}

/*****************************************************************************
 *
 *  lb_model_copy
 *
 *****************************************************************************/

__host__ int lb_model_copy(lb_t * lb, int flag) {

  lb_t * target;
  int ndevice;
  double * tmpf = NULL;

  assert(lb);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Make sure we alias */
    assert(lb->target == lb);
  }
  else {

    assert(lb->target);
    target = lb->target;

    copyFromTarget(&tmpf, &target->f, sizeof(double *)); 
    if (flag == cudaMemcpyHostToDevice) {
      copyToTarget(&target->ndist, &lb->ndist, sizeof(int)); 
      copyToTarget(&target->nsite, &lb->nsite, sizeof(int)); 
      copyToTarget(&target->model, &lb->model, sizeof(int)); 
      copyToTarget(tmpf, lb->f, NVEL*lb->nsite*lb->ndist*sizeof(double));
    }
    else {
      copyFromTarget(lb->f, tmpf, NVEL*lb->nsite*lb->ndist*sizeof(double));
    }
  }

  return 0;
}

/***************************************************************************
 *
 *  lb_init
 *
 *  Irrespective of the value of nhalo associated with coords.c,
 *  we only ever at the moment pass one plane worth of distribution
 *  values. This is nhalolocal.
 *
 ***************************************************************************/
 
int lb_init(lb_t * lb) {

  int nlocal[3];
  int nx, ny, nz;
  int nhalolocal = 1;
  int ndata;
  int nhalo;
  int ndevice;
  double * tmp;

  assert(lb);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nx = nlocal[X] + 2*nhalo;
  ny = nlocal[Y] + 2*nhalo;
  nz = nlocal[Z] + 2*nhalo;
  lb->nsite = nx*ny*nz;

  /* The total number of distribution data is then... */

  ndata = lb->nsite*lb->ndist*NVEL;

  lb->f = (double  *) malloc(ndata*sizeof(double));
  if (lb->f == NULL) fatal("malloc(distributions) failed\n");

  lb->fprime = (double  *) malloc(ndata*sizeof(double));
  if (lb->fprime == NULL) fatal("malloc(distributions) failed\n");


  /* Allocate target copy of structure or alias */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    lb->target = lb;
  }
  else {
    targetMalloc((void **) &lb->target, sizeof(lb_t));

    targetCalloc((void **) &tmp, ndata*sizeof(double));
    copyToTarget(&lb->target->f, &tmp, sizeof(double *));
 
    targetCalloc((void **) &tmp, ndata*sizeof(double));
    copyToTarget(&lb->target->fprime, &tmp, sizeof(double *));
  }

  /* Set up the MPI Datatypes used for full halo messages:
   *
   * in XY plane nx*ny blocks of 1 site with stride nz;
   * in XZ plane nx blocks of nz sites with stride ny*nz;
   * in YZ plane one contiguous block of ny*nz sites. */

  MPI_Type_vector(nx*ny, lb->ndist*NVEL*nhalolocal, lb->ndist*NVEL*nz,
		  MPI_DOUBLE, &lb->plane_xy_full);
  MPI_Type_commit(&lb->plane_xy_full);

  MPI_Type_vector(nx, lb->ndist*NVEL*nz*nhalolocal, lb->ndist*NVEL*ny*nz,
		  MPI_DOUBLE, &lb->plane_xz_full);
  MPI_Type_commit(&lb->plane_xz_full);

  MPI_Type_vector(1, lb->ndist*NVEL*ny*nz*nhalolocal, 1, MPI_DOUBLE,
		  &lb->plane_yz_full);
  MPI_Type_commit(&lb->plane_yz_full);

  lb_mpi_init(lb);
  lb_halo_set(lb, LB_HALO_FULL);
  lb_model_copy(lb, cudaMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  lb_nvel
 *
 *  Return number of velocities at runtime.
 *
 *****************************************************************************/

int lb_nvel(lb_t * lb, int * nvel) {

  assert(nvel);

  *nvel = NVEL;

  return 0;
}

/*****************************************************************************
 *
 *  lb_ndim
 *
 *  Return dimension of model at runtime.
 *
 *****************************************************************************/

int lb_ndim(lb_t * lb, int * ndim) {

  assert(ndim);

  *ndim = NDIM;

  return 0;
}

/*****************************************************************************
 *
 *  lb_nblock
 *
 *  Return cv block size.
 *
 *****************************************************************************/

int lb_nblock(lb_t * lb, int dim, int * nblock) {

  assert(dim == X || dim == Y || dim == Z);
  assert(nblock);

  if (dim == X) *nblock = CVXBLOCK;
  if (dim == Y) *nblock = CVYBLOCK;
  if (dim == Z) *nblock = CVZBLOCK;

  return 0;
}

/*****************************************************************************
 *
 *  lb_init_rest_f
 *
 *  Fluid uniformly at rest.
 *
 *****************************************************************************/

int lb_init_rest_f(lb_t * lb, double rho0) {

  int nlocal[3];
  int ic, jc, kc, index;

  assert(lb);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	lb_0th_moment_equilib_set(lb, index, 0, rho0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_mpi_init
 *
 *  Commit the various datatypes required for halo swaps.
 *
 *****************************************************************************/

static int lb_mpi_init(lb_t * lb) {

  int count;
  int nlocal[3];
  int nx, ny, nz;
  int * blocklen;
  int nhalo;
  MPI_Aint * disp_fwd;
  MPI_Aint * disp_bwd;
  MPI_Datatype * types;

  assert(lb);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nx = nlocal[X] + 2*nhalo;
  ny = nlocal[Y] + 2*nhalo;
  nz = nlocal[Z] + 2*nhalo;

  /* X direction */

  count = lb->ndist*CVXBLOCK + 2;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  lb_set_types(count, types);
  lb_set_blocks(lb, count, blocklen, CVXBLOCK, xblocklen_cv);
  lb_set_displacements(lb, count, disp_fwd, CVXBLOCK, xdisp_fwd_cv);
  lb_set_displacements(lb, count, disp_bwd, CVXBLOCK, xdisp_bwd_cv);

  MPI_Type_struct(count, blocklen, disp_fwd, types, &lb->site_x[FORWARD]);
  MPI_Type_struct(count, blocklen, disp_bwd, types, &lb->site_x[BACKWARD]);
  MPI_Type_commit(&lb->site_x[FORWARD]);
  MPI_Type_commit(&lb->site_x[BACKWARD]);

  MPI_Type_contiguous(ny*nz, lb->site_x[FORWARD], &lb->plane_yz_reduced[FORWARD]);
  MPI_Type_contiguous(ny*nz, lb->site_x[BACKWARD], &lb->plane_yz_reduced[BACKWARD]);
  MPI_Type_commit(&lb->plane_yz_reduced[FORWARD]);
  MPI_Type_commit(&lb->plane_yz_reduced[BACKWARD]);

  free(blocklen);
  free(disp_fwd);
  free(disp_bwd);
  free(types);

  /* Y direction */

  count = lb->ndist*CVYBLOCK + 2;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  lb_set_types(count, types);
  lb_set_blocks(lb, count, blocklen, CVYBLOCK, yblocklen_cv);
  lb_set_displacements(lb, count, disp_fwd, CVYBLOCK, ydisp_fwd_cv);
  lb_set_displacements(lb, count, disp_bwd, CVYBLOCK, ydisp_bwd_cv);

  MPI_Type_struct(count, blocklen, disp_fwd, types, &lb->site_y[FORWARD]);
  MPI_Type_struct(count, blocklen, disp_bwd, types, &lb->site_y[BACKWARD]);
  MPI_Type_commit(&lb->site_y[FORWARD]);
  MPI_Type_commit(&lb->site_y[BACKWARD]);

  MPI_Type_vector(nx, nz, ny*nz, lb->site_y[FORWARD],
		  &lb->plane_xz_reduced[FORWARD]);
  MPI_Type_vector(nx, nz, ny*nz, lb->site_y[BACKWARD],
		  &lb->plane_xz_reduced[BACKWARD]);
  MPI_Type_commit(&lb->plane_xz_reduced[FORWARD]);
  MPI_Type_commit(&lb->plane_xz_reduced[BACKWARD]);

  free(blocklen);
  free(disp_fwd);
  free(disp_bwd);
  free(types);

  /* Z direction */

  count = lb->ndist*CVZBLOCK + 2;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  lb_set_types(count, types);
  lb_set_blocks(lb, count, blocklen, CVZBLOCK, zblocklen_cv);
  lb_set_displacements(lb, count, disp_fwd, CVZBLOCK, zdisp_fwd_cv);
  lb_set_displacements(lb, count, disp_bwd, CVZBLOCK, zdisp_bwd_cv);

  MPI_Type_struct(count, blocklen, disp_fwd, types, &lb->site_z[FORWARD]);
  MPI_Type_struct(count, blocklen, disp_bwd, types, &lb->site_z[BACKWARD]);
  MPI_Type_commit(&lb->site_z[FORWARD]);
  MPI_Type_commit(&lb->site_z[BACKWARD]);

  MPI_Type_vector(nx*ny, 1, nz, lb->site_z[FORWARD], &lb->plane_xy_reduced[FORWARD]);
  MPI_Type_vector(nx*ny, 1, nz, lb->site_z[BACKWARD],
		  &lb->plane_xy_reduced[BACKWARD]);
  MPI_Type_commit(&lb->plane_xy_reduced[FORWARD]);
  MPI_Type_commit(&lb->plane_xy_reduced[BACKWARD]);

  free(blocklen);
  free(disp_fwd);
  free(disp_bwd);
  free(types);

  return 0;
}

/*****************************************************************************
 *
 *  lb_set_types
 *
 *  Set the type signature for reduced halo struct.
 *
 *****************************************************************************/

static int lb_set_types(int ntype, MPI_Datatype * type) {

  int n;

  assert(type);

  type[0] = MPI_LB;
  for (n = 1; n < ntype - 1; n++) {
    type[n] = MPI_DOUBLE;
  }
  type[ntype - 1] = MPI_UB;

  return 0;
}

/*****************************************************************************
 *
 *  lb_set_blocks
 *
 *  Set the blocklengths for the reduced halo struct.
 *  This means 'expanding' the basic blocks for the current DdQn model
 *  to the current number of distributions.
 *
 *****************************************************************************/

static int lb_set_blocks(lb_t * lb, int nblock, int * block, int nbasic,
			 const int * basic_block) {

  int n, p;

  assert(lb);

  block[0] = 1; /* For MPI_LB */

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < nbasic; p++) {
      block[1 + n*nbasic + p] = basic_block[p];
    }
  }

  block[nblock - 1] = 1; /* For MPI_UB */

  return 0;
}

/*****************************************************************************
 *
 *  lb_set_displacements
 *
 *  Set the displacements for the reduced halo struct.
 *
 *****************************************************************************/

static int lb_set_displacements(lb_t * lb, int ndisp, MPI_Aint * disp,
				int nbasic, const int * disp_basic) {
  int n, p;
  MPI_Aint disp0, disp1;

  assert(lb);

  MPI_Address(lb->f, &disp0);

  disp[0] = 0; /* For MPI_LB */

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < nbasic; p++) {
      MPI_Address(lb->f + n*NVEL + disp_basic[p], &disp1);
      disp[1 + n*nbasic + p] = disp1 - disp0;
    }
  }

  /* For MPI_UB */
  MPI_Address(lb->f + lb->ndist*NVEL, &disp1);
  disp[ndisp - 1] = disp1 - disp0;

  return 0;
}

/*****************************************************************************
 *
 *  lb_io_info_set
 *
 *****************************************************************************/

int lb_io_info_set(lb_t * lb, io_info_t * io_info) {

  char string[FILENAME_MAX];

  assert(lb);
  assert(io_info);

  lb->io_info = io_info;

  sprintf(string, "%1d x Distribution: d%dq%d", lb->ndist, NDIM, NVEL);

  io_info_set_name(lb->io_info, string);
  io_info_read_set(lb->io_info, IO_FORMAT_BINARY, lb_f_read);
  io_info_write_set(lb->io_info, IO_FORMAT_BINARY, lb_f_write);
  io_info_set_bytesize(lb->io_info, lb->ndist*NVEL*sizeof(double));
  io_info_format_set(lb->io_info, IO_FORMAT_BINARY, IO_FORMAT_BINARY);

  return 0;
}

/*****************************************************************************
 *
 *  lb_io_info
 *
 *****************************************************************************/

int lb_io_info(lb_t * lb, io_info_t ** io_info) {

  assert(lb);
  assert(io_info);

  *io_info = lb->io_info;

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo
 *
 *  Swap the distributions at the periodic/processor boundaries
 *  in each direction.
 *
 *****************************************************************************/

int lb_halo(lb_t * lb) {

  assert(lb);

  lb_halo_via_copy(lb);

  /* If MODEL order and NSIMDVL is 1 the struct
   * approach is still available. */

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_via_struct
 *
 *  Uses MPI datatypes to perform distribution halo swap, either 'full'
 *  or 'reduced', depending on what is currently selected.
 *
 *  This works only for MODEL, not MODEL_R.
 *
 *****************************************************************************/

int lb_halo_via_struct(lb_t * lb) {

  int ic, jc, kc;
  int ihalo, ireal;
  int nhalo;
  int nlocal[3];

  const int tagf = 900;
  const int tagb = 901;

  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm = cart_comm();

  assert(lb);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  /* The x-direction (YZ plane) */

  if (cart_size(X) == 1) {
    if (is_periodic(X)) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  ihalo = lb->ndist*NVEL*coords_index(0, jc, kc);
	  ireal = lb->ndist*NVEL*coords_index(nlocal[X], jc, kc);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*coords_index(nlocal[X]+1, jc, kc);
	  ireal = lb->ndist*NVEL*coords_index(1, jc, kc);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {
    ihalo = lb->ndist*NVEL*coords_index(nlocal[X] + 1, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_yz[BACKWARD],
	      cart_neighb(FORWARD,X), tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*coords_index(0, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_yz[FORWARD],
	      cart_neighb(BACKWARD,X), tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*coords_index(1, 1-nhalo, 1-nhalo);
    MPI_Issend(lb->f + ireal, 1, lb->plane_yz[BACKWARD],
	       cart_neighb(BACKWARD,X), tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*coords_index(nlocal[X], 1-nhalo, 1-nhalo);
    MPI_Issend(lb->f + ireal, 1, lb->plane_yz[FORWARD],
	       cart_neighb(FORWARD,X), tagf, comm, &request[3]);
    MPI_Waitall(4, request, status);
  }
  
  /* The y-direction (XZ plane) */

  if (cart_size(Y) == 1) {
    if (is_periodic(Y)) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  ihalo = lb->ndist*NVEL*coords_index(ic, 0, kc);
	  ireal = lb->ndist*NVEL*coords_index(ic, nlocal[Y], kc);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*coords_index(ic, nlocal[Y] + 1, kc);
	  ireal = lb->ndist*NVEL*coords_index(ic, 1, kc);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {
    ihalo = lb->ndist*NVEL*coords_index(1 - nhalo, nlocal[Y] + 1, 1 - nhalo);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_xz[BACKWARD],
	      cart_neighb(FORWARD,Y), tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*coords_index(1 - nhalo, 0, 1 - nhalo);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_xz[FORWARD], cart_neighb(BACKWARD,Y),
	      tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*coords_index(1 - nhalo, 1, 1 - nhalo);
    MPI_Issend(lb->f + ireal, 1, lb->plane_xz[BACKWARD], cart_neighb(BACKWARD,Y),
	       tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*coords_index(1 - nhalo, nlocal[Y], 1 - nhalo);
    MPI_Issend(lb->f + ireal, 1, lb->plane_xz[FORWARD], cart_neighb(FORWARD,Y),
	       tagf, comm, &request[3]);
    MPI_Waitall(4, request, status);
  }
  
  /* Finally, z-direction (XY plane) */

  if (cart_size(Z) == 1) {
    if (is_periodic(Z)) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (jc = 0; jc <= nlocal[Y] + 1; jc++) {

	  ihalo = lb->ndist*NVEL*coords_index(ic, jc, 0);
	  ireal = lb->ndist*NVEL*coords_index(ic, jc, nlocal[Z]);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*coords_index(ic, jc, nlocal[Z] + 1);
	  ireal = lb->ndist*NVEL*coords_index(ic, jc, 1);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {

    ihalo = lb->ndist*NVEL*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z] + 1);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_xy[BACKWARD], cart_neighb(FORWARD,Z),
	      tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*coords_index(1 - nhalo, 1 - nhalo, 0);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_xy[FORWARD], cart_neighb(BACKWARD,Z),
	      tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*coords_index(1 - nhalo, 1 - nhalo, 1);
    MPI_Issend(lb->f + ireal, 1, lb->plane_xy[BACKWARD], cart_neighb(BACKWARD,Z),
	       tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z]);
    MPI_Issend(lb->f + ireal, 1, lb->plane_xy[FORWARD], cart_neighb(FORWARD,Z),
	       tagf, comm, &request[3]);  
    MPI_Waitall(4, request, status);
  }
 
  return 0;
}

/*****************************************************************************
 *
 *  lb_f_read
 *
 *  Read one lattice site (index) worth of distributions.
 *  Note that read-write is always 'MODEL' order.
 *
 *****************************************************************************/

static int lb_f_read(FILE * fp, int index, void * self) {

  int iread, n, p;
  int nr = 0;
  lb_t * lb = (lb_t*) self;

  assert(fp);
  assert(lb);

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < NVEL; p++) {
      iread = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      nr += fread(lb->f + iread, sizeof(double), 1, fp);
    }
  }

  if (nr != lb->ndist*NVEL) fatal("fread(lb) failed at %d\n", index);

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_write
 *
 *  Write one lattice site (index) worth of distributions.
 *  Note that read/write is always 'MODEL' order.
 *
 *****************************************************************************/

static int lb_f_write(FILE * fp, int index, void * self) {

  int iwrite, n, p;
  int nw = 0;
  lb_t * lb = (lb_t*) self;

  assert(fp);
  assert(self);

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < NVEL; p++) {
      iwrite = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      nw += fwrite(lb->f + iwrite, sizeof(double), 1, fp);
    }
  }

  if (nw != lb->ndist*NVEL) fatal("fwrite(lb) failed at %d\n", index);

  return 0;
}

static int isReduced_=0;

/*****************************************************************************
 *
 *  lb_halo_set
 *
 *  Set the actual halo datatype.
 *
 *****************************************************************************/

int lb_halo_set(lb_t * lb, lb_halo_enum_t type) {

  assert(lb);

  if (type == LB_HALO_REDUCED) {
    lb->plane_xy[FORWARD]  = lb->plane_xy_reduced[FORWARD];
    lb->plane_xy[BACKWARD] = lb->plane_xy_reduced[BACKWARD];
    lb->plane_xz[FORWARD]  = lb->plane_xz_reduced[FORWARD];
    lb->plane_xz[BACKWARD] = lb->plane_xz_reduced[BACKWARD];
    lb->plane_yz[FORWARD]  = lb->plane_yz_reduced[FORWARD];
    lb->plane_yz[BACKWARD] = lb->plane_yz_reduced[BACKWARD];

    isReduced_=1;
  }
  else {
    /* Default to full halo. */
    lb->plane_xy[FORWARD]  = lb->plane_xy_full;
    lb->plane_xy[BACKWARD] = lb->plane_xy_full;
    lb->plane_xz[FORWARD]  = lb->plane_xz_full;
    lb->plane_xz[BACKWARD] = lb->plane_xz_full;
    lb->plane_yz[FORWARD]  = lb->plane_yz_full;
    lb->plane_yz[BACKWARD] = lb->plane_yz_full;
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_reduced
 *
 *  Return 1 if halo is reduced, 0 otherwise.
 *
 *****************************************************************************/

int lb_halo_reduced(lb_t * lb) {

  return isReduced_;

}

/*****************************************************************************
 *
 *  lb_ndist
 *
 *  Return the number of distribution functions.
 *
 *****************************************************************************/

int lb_ndist(lb_t * lb, int * ndist) {

  assert(lb);
  assert(ndist);

  *ndist = lb->ndist;

  return 0;
}

/*****************************************************************************
 *
 *  lb_ndist_set
 *
 *  Set the number of distribution functions to be used.
 *  This needs to occur between create() and init()
 *
 *****************************************************************************/

int lb_ndist_set(lb_t * lb, int n) {

  assert(lb);
  assert(n > 0);
  assert(lb->f == NULL); /* don't change after initialisation */

  lb->ndist = n;

  return 0;
}

/*****************************************************************************
 *
 *  lb_f
 *
 *  Get the distribution at site index, velocity p, distribution n.
 *
 *****************************************************************************/

int lb_f(lb_t * lb, int index, int p, int n, double * f) {

  assert(lb);
  assert(index >= 0 && index < lb->nsite);
  assert(p >= 0 && p < NVEL);
  assert(n >= 0 && n < lb->ndist);

  *f = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)];

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_set
 *
 *  Set the distribution for site index, velocity p, distribution n.
 *
 *****************************************************************************/

int lb_f_set(lb_t * lb, int index, int p, int n, double fvalue) {

  assert(lb);
  assert(index >= 0 && index < lb->nsite);
  assert(p >= 0 && p < NVEL);
  assert(n >= 0 && n < lb->ndist);

  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)] = fvalue;

  return 0;
}

/*****************************************************************************
 *
 *  lb_0th_moment
 *
 *  Return the zeroth moment of the distribution (rho for n = 0).
 *
 *****************************************************************************/

int lb_0th_moment(lb_t * lb, int index, lb_dist_enum_t nd, double * rho) {

  int p;

  assert(lb);
  assert(rho);
  assert(index >= 0 && index < lb->nsite);
  assert(nd >= 0 && nd < lb->ndist);

  *rho = 0.0;

  for (p = 0; p < NVEL; p++) {
    *rho += lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, nd, p)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_1st_moment
 *
 *  Return the first moment of the distribution p.
 *
 *****************************************************************************/

int lb_1st_moment(lb_t * lb, int index, lb_dist_enum_t nd, double g[3]) {

  int p;
  int n;

  assert(lb);
  assert(index >= 0 && index < lb->nsite);
  assert(nd >= 0 && nd < lb->ndist);

  for (n = 0; n < NDIM; n++) {
    g[n] = 0.0;
  }

  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < NDIM; n++) {
      g[n] += cv[p][n]*lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, nd, p)];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_2nd_moment
 *
 *  Return the (deviatoric) stress at index.
 *
 *****************************************************************************/

int lb_2nd_moment(lb_t * lb, int index, lb_dist_enum_t nd, double s[3][3]) {

  int p, ia, ib;

  assert(lb);
  assert(nd == LB_RHO);
  assert(index >= 0  && index < lb->nsite);

  for (ia = 0; ia < NDIM; ia++) {
    for (ib = 0; ib < NDIM; ib++) {
      s[ia][ib] = 0.0;
    }
  }

  for (p = 0; p < NVEL; p++) {
    for (ia = 0; ia < NDIM; ia++) {
      for (ib = 0; ib < NDIM; ib++) {
	s[ia][ib] += lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, nd, p)]
	  *q_[p][ia][ib];
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_0th_moment_equilib_set
 *
 *  Project the given value of rho onto the equilibrium distribution via
 *
 *    f_i = w_i rho
 *
 *****************************************************************************/

int lb_0th_moment_equilib_set(lb_t * lb, int index, int n, double rho) {

  int p;

  assert(lb);
  assert (n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)] = wv[p]*rho;
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_1st_moment_equilib_set
 *
 *  Set equilibrium f_i for a given rho, u using an equilibrium stress.
 *
 *****************************************************************************/

int lb_1st_moment_equilib_set(lb_t * lb, int index, double rho, double u[3]) {

  int ia, ib, p;
  double udotc;
  double sdotq;

  assert(lb);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    udotc = 0.0;
    sdotq = 0.0;
    for (ia = 0; ia < 3; ia++) {
      udotc += u[ia]*cv[p][ia];
      for (ib = 0; ib < 3; ib++) {
	sdotq += q_[p][ia][ib]*u[ia]*u[ib];
      }
    }

    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, LB_RHO, p)]
      = rho*wv[p]*(1.0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_index
 *
 *  Return the distribution n at index.
 *
 *****************************************************************************/

int lb_f_index(lb_t * lb, int index, int n, double f[NVEL]) {

  int p;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    f[p] = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)];
  }

  return 0;
}


/*****************************************************************************
 *
 *  lb_f_multi_index
 *
 *  Return a vector of distributions starting at index 
 *  where the vector length is fixed at NSIMDVL
 *
 *****************************************************************************/

int lb_f_multi_index(lb_t * lb, int index, int n, double fv[NVEL][NSIMDVL]) {

  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    for (iv = 0; iv < NSIMDVL; iv++) {
      fv[p][iv] = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index + iv, n, p)];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_multi_index_part
 *
 *  Return a vector of distributions starting at index 
 *  where the vector length is passed in at runtime
 *
 *****************************************************************************/

int lb_f_multi_index_part(lb_t * lb, int index, int n, double fv[NVEL][NSIMDVL],
			  int nv) {

  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    for (iv = 0; iv < nv; iv++) {
      fv[p][iv] = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index + iv, n, p)];
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  lb_index_set
 *
 *  Set distribution n and index.
 *
 *****************************************************************************/

int lb_f_index_set(lb_t * lb, int index, int n, double f[NVEL]) {

  int p;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)] = f[p];
  }

  return 0;
}


/*****************************************************************************
 *
 *  lb_f_multi_index_set
 *
 *  Set a vector of distributions starting at index 
 *  where the vector length is fixed at NSIMDVL
 *
 *****************************************************************************/

int lb_f_multi_index_set(lb_t * lb, int index, int n, double fv[NVEL][NSIMDVL]){

  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);
  assert(0);
  for (p = 0; p < NVEL; p++) {
    for (iv = 0; iv < NSIMDVL; iv++) {
      lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index + iv, n, p)] = fv[p][iv];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_multi_index_set_part
 *
 *  Set a vector of distributions starting at index 
 *  where the vector length is passed in at runtime
 *
 *****************************************************************************/

int lb_f_multi_index_set_part(lb_t * lb, int index, int n,
			      double fv[NVEL][NSIMDVL], int nv) {
  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);
  assert(0);
  for (p = 0; p < NVEL; p++) {
    for (iv = 0; iv < nv; iv++) {
      lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index + iv, n, p)] = fv[p][iv];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_order
 *
 *  Model data ordering
 *
 *****************************************************************************/

int lb_order(lb_t * lb) {

  assert(lb);

  return lb->model;
}

/*****************************************************************************
 *
 *  lb_halo_via_copy
 *
 *  A version of the halo swap which uses a flat buffer to copy the
 *  relevant data rathar than MPI data types. It is therefore a 'full'
 *  halo swapping NVEL distributions at each site.
 *
 *  It works for both MODEL and MODEL_R, but the loop order favours
 *  MODEL_R.
 *
 *****************************************************************************/

int lb_halo_via_copy(lb_t * lb) {

  int ic, jc, kc;
  int n, p;
  int pforw, pback;
  int index, indexhalo, indexreal;
  int nsend, count;
  int nlocal[3];

  const int tagf = 900;
  const int tagb = 901;

  double * sendforw;
  double * sendback;
  double * recvforw;
  double * recvback;

  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm = cart_comm();

  assert(lb);

  coords_nlocal(nlocal);

  /* The x-direction (YZ plane) */

  nsend = NVEL*lb->ndist*nlocal[Y]*nlocal[Z];
  sendforw = (double *) malloc(nsend*sizeof(double));
  sendback = (double *) malloc(nsend*sizeof(double));
  recvforw = (double *) malloc(nsend*sizeof(double));
  recvback = (double *) malloc(nsend*sizeof(double));
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvforw == NULL) fatal("malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(nlocal[X], jc, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendforw[count] = lb->f[indexreal];

	  index = coords_index(1, jc, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = lb->f[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);

  if (cart_size(X) == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cart_neighb(FORWARD, X);
    pback = cart_neighb(BACKWARD, X);

    MPI_Irecv(recvforw, nsend, MPI_DOUBLE, pforw, tagb, comm, request);
    MPI_Irecv(recvback, nsend, MPI_DOUBLE, pback, tagf, comm, request + 1);

    MPI_Issend(sendback, nsend, MPI_DOUBLE, pback, tagb, comm, request + 2);
    MPI_Issend(sendforw, nsend, MPI_DOUBLE, pforw, tagf, comm, request + 3);

    /* Wait for receives */
    MPI_Waitall(2, request, status);
  }

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(0, jc, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvback[count];

	  index = coords_index(nlocal[X] + 1, jc, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvforw[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);

  free(recvback);
  free(recvforw);

  if (cart_size(X) > 1) {
    /* Wait for sends */
    MPI_Waitall(2, request + 2, status);
  }

  free(sendback);
  free(sendforw);
  

  /* The y-direction (XZ plane) */

  nsend = NVEL*lb->ndist*(nlocal[X] + 2)*nlocal[Z];
  sendforw = (double *) malloc(nsend*sizeof(double));
  sendback = (double *) malloc(nsend*sizeof(double));
  recvforw = (double *) malloc(nsend*sizeof(double));
  recvback = (double *) malloc(nsend*sizeof(double));
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvforw == NULL) fatal("malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(ic, nlocal[Y], kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendforw[count] = lb->f[indexreal];

	  index = coords_index(ic, 1, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = lb->f[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);


  if (cart_size(Y) == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cart_neighb(FORWARD, Y);
    pback = cart_neighb(BACKWARD, Y);

    MPI_Irecv(recvforw, nsend, MPI_DOUBLE, pforw, tagb, comm, request);
    MPI_Irecv(recvback, nsend, MPI_DOUBLE, pback, tagf, comm, request + 1);

    MPI_Issend(sendback, nsend, MPI_DOUBLE, pback, tagb, comm, request + 2);
    MPI_Issend(sendforw, nsend, MPI_DOUBLE, pforw, tagf, comm, request + 3);

    /* Wait of receives */
    MPI_Waitall(2, request, status);
  }

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(ic, 0, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvback[count];

	  index = coords_index(ic, nlocal[Y] + 1, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvforw[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);
  free(recvback);
  free(recvforw);

  if (cart_size(Y) > 1) {
    /* Wait for sends */
    MPI_Waitall(2, request + 2, status);
  }

  free(sendback);
  free(sendforw);

  /* Finally, z-direction (XY plane) */

  nsend = NVEL*lb->ndist*(nlocal[X] + 2)*(nlocal[Y] + 2);
  sendforw = (double *) malloc(nsend*sizeof(double));
  sendback = (double *) malloc(nsend*sizeof(double));
  recvforw = (double *) malloc(nsend*sizeof(double));
  recvback = (double *) malloc(nsend*sizeof(double));
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvforw == NULL) fatal("malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (jc = 0; jc <= nlocal[Y] + 1; jc++) {

	  index = coords_index(ic, jc, nlocal[Z]);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendforw[count] = lb->f[indexreal];

	  index = coords_index(ic, jc, 1);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = lb->f[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);

  if (cart_size(Z) == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cart_neighb(FORWARD, Z);
    pback = cart_neighb(BACKWARD, Z);

    MPI_Irecv(recvforw, nsend, MPI_DOUBLE, pforw, tagb, comm, request);
    MPI_Irecv(recvback, nsend, MPI_DOUBLE, pback, tagf, comm, request + 1);

    MPI_Issend(sendback, nsend, MPI_DOUBLE, pback, tagb, comm, request + 2);
    MPI_Issend(sendforw, nsend, MPI_DOUBLE, pforw, tagf, comm, request + 3);

    /* Wait for receives */
    MPI_Waitall(2, request, status);
  }

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (jc = 0; jc <= nlocal[Y] + 1; jc++) {

	  index = coords_index(ic, jc, 0);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvback[count];

	  index = coords_index(ic, jc, nlocal[Z] + 1);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvforw[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);
  free(recvback);
  free(recvforw);

  if (cart_size(Z) > 1) {
    /* Wait for sends */
    MPI_Waitall(2, request + 2, status);
  }

  free(sendback);
  free(sendforw);
 
  return 0;
}
