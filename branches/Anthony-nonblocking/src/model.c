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
 *  (c) 2010-2014 The University of Edinburgh
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
#include "control.h"
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
  lb->model = LB_DATA_MODEL;
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

void lb_free(lb_t * lb) {

  assert(lb);

  if (lb->io_info) io_info_destroy(lb->io_info);
  if (lb->f) free(lb->f);
  if (lb->fprime) free(lb->fprime);
  //if (lb->t_f) targetFree(lb->t_f);
  //if (lb->t_fprime) targetFree(lb->t_fprime);


  if (lb->tcopy) {

    //free data space on target 
    double* tmpptr;
    lb_t* t_obj = lb->tcopy;
    copyFromTarget(&tmpptr,&(t_obj->f),sizeof(double*)); 
    targetFree(tmpptr);

    copyFromTarget(&tmpptr,&(t_obj->fprime),sizeof(double*)); 
    targetFree(tmpptr);
    
    //free target copy of structure
    targetFree(lb->tcopy);
  }

  targetFinalize();

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


  int Nall[3]; 
  Nall[X]=nx;  Nall[Y]=ny;  Nall[Z]=nz;
  targetInit(Nall, lb->ndist*NVEL, nhalo);

  /* allocate target copy of structure */
  targetMalloc((void**) &(lb->tcopy),sizeof(lb_t));

  /* allocate data space on target */
  double* tmpptr;
  lb_t* t_obj = lb->tcopy;
  targetCalloc((void**) &tmpptr,ndata*sizeof(double));
  copyToTarget(&(t_obj->f),&tmpptr,sizeof(double*)); 

  //  copyToTarget(&(t_obj->io_info),&(lb->io_info),sizeof(io_info_t)); 
  copyToTarget(&(t_obj->ndist),&(lb->ndist),sizeof(int)); 
  copyToTarget(&(t_obj->nsite),&(lb->nsite),sizeof(int)); 
  copyToTarget(&(t_obj->model),&(lb->model),sizeof(int)); 

  copyToTarget(&(t_obj->plane_xy_full),&(lb->plane_xy_full),sizeof(MPI_Datatype)); 
  copyToTarget(&(t_obj->plane_xz_full),&(lb->plane_xz_full),sizeof(MPI_Datatype)); 
  copyToTarget(&(t_obj->plane_yz_full),&(lb->plane_yz_full),sizeof(MPI_Datatype)); 

  copyToTarget(&(t_obj->plane_xy_reduced),&(lb->plane_xy_reduced),2*sizeof(MPI_Datatype)); 
  copyToTarget(&(t_obj->plane_xz_reduced),&(lb->plane_xz_reduced),2*sizeof(MPI_Datatype)); 
  copyToTarget(&(t_obj->plane_yz_reduced),&(lb->plane_yz_reduced),2*sizeof(MPI_Datatype)); 

  copyToTarget(&(t_obj->plane_xy),&(lb->plane_xy),2*sizeof(MPI_Datatype)); 
  copyToTarget(&(t_obj->plane_xz),&(lb->plane_xz),2*sizeof(MPI_Datatype)); 
  copyToTarget(&(t_obj->plane_yz),&(lb->plane_yz),2*sizeof(MPI_Datatype)); 

  copyToTarget(&(t_obj->site_x),&(lb->site_x),2*sizeof(MPI_Datatype)); 
  copyToTarget(&(t_obj->site_y),&(lb->site_y),2*sizeof(MPI_Datatype)); 
  copyToTarget(&(t_obj->site_z),&(lb->site_z),2*sizeof(MPI_Datatype)); 

  lb->t_f= tmpptr; //DEPRECATED direct access to target data.


  /* allocate another space on target for staging data */

  targetCalloc((void**) &tmpptr,ndata*sizeof(double));
  copyToTarget(&(t_obj->fprime),&tmpptr,sizeof(double*)); 
  
  lb->t_fprime= tmpptr; //DEPRECATED direct access to target data.





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

  if (lb_order(lb) == MODEL) {
    lb_halo_via_copy_nonblocking_start(lb);//lb_halo_via_copy(lb);//lb_halo_via_struct(lb);
    lb_halo_via_copy_nonblocking_end(lb);
  }
  else {
    /* MODEL_R only has ... */
    lb_halo_via_copy(lb);//lb_halo_via_copy_nonblocking(lb);//
  }

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

  double* fptr; /*pointer to the distribution*/
  if (get_step()) /* we are in the timestep, so use target copy */
    fptr=lb->tcopy->f;
  else
    fptr=lb->f;  /* we are not in a timestep, use host copy */
  
  /* The x-direction (YZ plane) */

  if (cart_size(X) == 1) {
    if (is_periodic(X)) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  ihalo = lb->ndist*NVEL*coords_index(0, jc, kc);
	  ireal = lb->ndist*NVEL*coords_index(nlocal[X], jc, kc);
	  memcpy(fptr + ihalo, fptr + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*coords_index(nlocal[X]+1, jc, kc);
	  ireal = lb->ndist*NVEL*coords_index(1, jc, kc);
	  memcpy(fptr + ihalo, fptr + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {
    ihalo = lb->ndist*NVEL*coords_index(nlocal[X] + 1, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(fptr + ihalo, 1, lb->plane_yz[BACKWARD],
	      cart_neighb(FORWARD,X), tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*coords_index(0, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(fptr + ihalo, 1, lb->plane_yz[FORWARD],
	      cart_neighb(BACKWARD,X), tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*coords_index(1, 1-nhalo, 1-nhalo);
    MPI_Issend(fptr + ireal, 1, lb->plane_yz[BACKWARD],
	       cart_neighb(BACKWARD,X), tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*coords_index(nlocal[X], 1-nhalo, 1-nhalo);
    MPI_Issend(fptr + ireal, 1, lb->plane_yz[FORWARD],
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
	  memcpy(fptr + ihalo, fptr + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*coords_index(ic, nlocal[Y] + 1, kc);
	  ireal = lb->ndist*NVEL*coords_index(ic, 1, kc);
	  memcpy(fptr + ihalo, fptr + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {
    ihalo = lb->ndist*NVEL*coords_index(1 - nhalo, nlocal[Y] + 1, 1 - nhalo);
    MPI_Irecv(fptr + ihalo, 1, lb->plane_xz[BACKWARD],
	      cart_neighb(FORWARD,Y), tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*coords_index(1 - nhalo, 0, 1 - nhalo);
    MPI_Irecv(fptr + ihalo, 1, lb->plane_xz[FORWARD], cart_neighb(BACKWARD,Y),
	      tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*coords_index(1 - nhalo, 1, 1 - nhalo);
    MPI_Issend(fptr + ireal, 1, lb->plane_xz[BACKWARD], cart_neighb(BACKWARD,Y),
	       tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*coords_index(1 - nhalo, nlocal[Y], 1 - nhalo);
    MPI_Issend(fptr + ireal, 1, lb->plane_xz[FORWARD], cart_neighb(FORWARD,Y),
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
	  memcpy(fptr + ihalo, fptr + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*coords_index(ic, jc, nlocal[Z] + 1);
	  ireal = lb->ndist*NVEL*coords_index(ic, jc, 1);
	  memcpy(fptr + ihalo, fptr + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {

    ihalo = lb->ndist*NVEL*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z] + 1);
    MPI_Irecv(fptr + ihalo, 1, lb->plane_xy[BACKWARD], cart_neighb(FORWARD,Z),
	      tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*coords_index(1 - nhalo, 1 - nhalo, 0);
    MPI_Irecv(fptr + ihalo, 1, lb->plane_xy[FORWARD], cart_neighb(BACKWARD,Z),
	      tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*coords_index(1 - nhalo, 1 - nhalo, 1);
    MPI_Issend(fptr + ireal, 1, lb->plane_xy[BACKWARD], cart_neighb(BACKWARD,Z),
	       tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z]);
    MPI_Issend(fptr + ireal, 1, lb->plane_xy[FORWARD], cart_neighb(FORWARD,Z),
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

  /* *f = lb->f[lb->ndist*NVEL*index + n*NVEL + p];*/
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

  /* lb->f[lb->ndist*NVEL*index + n*NVEL + p] = fvalue;*/
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
    /* *rho += lb->f[lb->ndist*NVEL*index + nd*NVEL + p];*/
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
      /* g[n] += cv[p][n]*lb->f[lb->ndist*NVEL*index + nd*NVEL + p];*/
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
	/* s[ia][ib] += lb->f[lb->ndist*NVEL*index + p]*q_[p][ia][ib];*/
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
    /* lb->f[lb->ndist*NVEL*index + n*NVEL + p] = wv[p]*rho;*/
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
      /* lb->f[lb->ndist*NVEL*index + p]*/
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
    /* f[p] = lb->f[lb->ndist*NVEL*index + n*NVEL + p];*/
  }

  return 0;
}


/*****************************************************************************
 *
 *  lb_f_multi_index
 *
 *  Return a vector of distributions starting at index 
 *  where the vector length is fixed at SIMDVL
 *
 *****************************************************************************/

int lb_f_multi_index(lb_t * lb, int index, int n, double fv[NVEL][SIMDVL]) {

  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    for (iv = 0; iv < SIMDVL; iv++) {
      /* fv[p][iv] = lb->f[lb->ndist*NVEL*(index + iv) + n*NVEL + p];*/
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

int lb_f_multi_index_part(lb_t * lb, int index, int n, double fv[NVEL][SIMDVL],
			  int nv) {

  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    for (iv = 0; iv < nv; iv++) {
      /* fv[p][iv] = lb->f[lb->ndist*NVEL*(index + iv) + n*NVEL + p];*/
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
    /* lb->f[lb->ndist*NVEL*index + n*NVEL + p] = f[p];*/
    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)] = f[p];
  }

  return 0;
}


/*****************************************************************************
 *
 *  lb_f_multi_index_set
 *
 *  Set a vector of distributions starting at index 
 *  where the vector length is fixed at SIMDVL
 *
 *****************************************************************************/

int lb_f_multi_index_set(lb_t * lb, int index, int n, double fv[NVEL][SIMDVL]){

  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    for (iv = 0; iv < SIMDVL; iv++) {
      /* lb->f[lb->ndist*NVEL*(index + iv) + n*NVEL + p] = fv[p][iv];*/
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
			      double fv[NVEL][SIMDVL], int nv) {
  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < NVEL; p++) {
    for (iv = 0; iv < nv; iv++) {
      /* lb->f[lb->ndist*NVEL*(index + iv) + n*NVEL + p] = fv[p][iv];*/
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

  double* fptr; /*pointer to the distribution*/
  if (get_step()) /* we are in the timestep, so use target copy */
    fptr=lb->tcopy->f;
  else
    fptr=lb->f;  /* we are not in a timestep, use host copy */


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
	  sendforw[count] = fptr[indexreal];

	  index = coords_index(1, jc, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = fptr[indexreal];
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
	  fptr[indexhalo] = recvback[count];

	  index = coords_index(nlocal[X] + 1, jc, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = recvforw[count];
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
	  sendforw[count] = fptr[indexreal];

	  index = coords_index(ic, 1, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = fptr[indexreal];
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
	  fptr[indexhalo] = recvback[count];

	  index = coords_index(ic, nlocal[Y] + 1, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = recvforw[count];
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
	  sendforw[count] = fptr[indexreal];

	  index = coords_index(ic, jc, 1);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = fptr[indexreal];
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
	  fptr[indexhalo] = recvback[count];

	  index = coords_index(ic, jc, nlocal[Z] + 1);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = recvforw[count];
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


/*****************************************************************************
 *
 *  lb_halo_via_copy_nonblocking
 *
 *  A version of the halo swap which uses a flat buffer to copy the
 *  relevant data rathar than MPI data types. This version is NON-
 *  BLOCKING and sends 26 messages instead of 6: 6 planes, 12 edges
 *  and 8 corners.
 *
 *  It works for both MODEL and MODEL_R, but the loop order favours
 *  MODEL_R.
 *
 *****************************************************************************/

int lb_halo_via_copy_nonblocking_start(lb_t * lb) {

  /* Send messages to 6 neighbouring plains 
                      12 neighbouring edges
                      8 neighbouring corners */
  halo_planes(lb);
  halo_edges(lb);
  halo_corners(lb);

  return 0;
}

int lb_halo_via_copy_nonblocking_end(lb_t * lb) {

  int n;
  int recvcount;
  int sendcount;

  for (n=0; n<26; n++){
    MPI_Waitany(26, lb->hl.recvreq, &recvcount, lb->hl.status);
  }

  /* Copy data from MPI buffers */
  unpack_halo_buffers(lb);

  for (n=0; n<26; n++){
    MPI_Waitany(26, lb->hl.sendreq, &sendcount, lb->hl.status);
  }

  return 0;
}


/*****************************************************************************
 *
 *  halo_planes
 *
 *  Sends 6 MPI messages (the planes) for the Non-Blocking version.
 *
 *****************************************************************************/

void halo_planes(lb_t * lb) {

  int ic, jc, kc;
  int n, p;
  int index, indexhalo, indexreal;
  int count;
  int nlocal[3];

  const int tagf = 900;
  const int tagb = 901;

  /* The ranks of neighbouring planes */
  int pforwX, pbackX, pforwY, pbackY, pforwZ, pbackZ;

  MPI_Comm comm = cart_comm();

  assert(lb);

  coords_nlocal(nlocal);

  double* fptr; /*pointer to the distribution*/
  if (get_step()) /* we are in the timestep, so use target copy */
    fptr=lb->tcopy->f;
  else
    fptr=lb->f;  /* we are not in a timestep, use host copy */

  /* Allocate size of sendplanes and number of elements send with each plane */
  int nsendXZ, nsendYZ, nsendXY;
  nsendYZ = NVEL*lb->ndist*nlocal[Y]*nlocal[Z];
  nsendXZ = NVEL*lb->ndist*nlocal[X]*nlocal[Z];
  nsendXY = NVEL*lb->ndist*nlocal[X]*nlocal[Y];

  /* Allocate message sizes for plane send/receives */
  lb->hl .sendforwYZ = (double *) malloc(nsendYZ*sizeof(double));
  lb->hl .sendbackYZ = (double *) malloc(nsendYZ*sizeof(double));
  lb->hl .recvforwYZ = (double *) malloc(nsendYZ*sizeof(double));
  lb->hl .recvbackYZ = (double *) malloc(nsendYZ*sizeof(double));
  lb->hl .sendforwXZ = (double *) malloc(nsendXZ*sizeof(double));
  lb->hl .sendbackXZ = (double *) malloc(nsendXZ*sizeof(double));
  lb->hl .recvforwXZ = (double *) malloc(nsendXZ*sizeof(double));
  lb->hl .recvbackXZ = (double *) malloc(nsendXZ*sizeof(double));
  lb->hl .sendforwXY = (double *) malloc(nsendXY*sizeof(double));
  lb->hl .sendbackXY = (double *) malloc(nsendXY*sizeof(double));
  lb->hl .recvforwXY = (double *) malloc(nsendXY*sizeof(double));
  lb->hl .recvbackXY = (double *) malloc(nsendXY*sizeof(double));

  /* Receive planes in the x-direction */
  /* PPM, NMM, P=Positive, M=Middle, N=Negative for the XYZ directions respectively */
  pforwX = nonblocking_cart_neighb(PMM);
  pbackX = nonblocking_cart_neighb(NMM);

  MPI_Irecv(&lb->hl.recvforwYZ[0], nsendYZ, MPI_DOUBLE, pforwX, tagb, comm, &lb->hl.recvreq[0]);
  MPI_Irecv(&lb->hl .recvbackYZ[0], nsendYZ, MPI_DOUBLE, pbackX, tagf, comm, &lb->hl.recvreq[1]);

  /* Receive planes in the y-direction */
  pforwY = nonblocking_cart_neighb(MPM);
  pbackY = nonblocking_cart_neighb(MNM);

  MPI_Irecv(&lb->hl .recvforwXZ[0], nsendXZ, MPI_DOUBLE, pforwY, tagb, comm, &lb->hl.recvreq[2]);
  MPI_Irecv(&lb->hl .recvbackXZ[0], nsendXZ, MPI_DOUBLE, pbackY, tagf, comm, &lb->hl.recvreq[3]);

  /* Receive planes in the z-direction */
  pforwZ = nonblocking_cart_neighb(MMP);
  pbackZ = nonblocking_cart_neighb(MMN);

  MPI_Irecv(&lb->hl .recvforwXY[0], nsendXY, MPI_DOUBLE, pforwZ, tagb, comm, &lb->hl.recvreq[4]);
  MPI_Irecv(&lb->hl .recvbackXY[0], nsendXY, MPI_DOUBLE, pbackZ, tagf, comm, &lb->hl.recvreq[5]);

  /* Send in the x-direction (YZ plane) */
  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(nlocal[X], jc, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendforwYZ[count] = fptr[indexreal];

	  index = coords_index(1, jc, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendbackYZ[count] = fptr[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsendYZ);

  MPI_Issend(&lb->hl .sendbackYZ [0], nsendYZ, MPI_DOUBLE, pbackX, tagb, comm, &lb->hl.sendreq[0]);
  MPI_Issend(&lb->hl .sendforwYZ [0], nsendYZ, MPI_DOUBLE, pforwX, tagf, comm, &lb->hl.sendreq[1]);


  /* Send in the y-direction (XZ plane) */
  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 1; ic <= nlocal[X]; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(ic, nlocal[Y], kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendforwXZ[count] = fptr[indexreal];

	  index = coords_index(ic, 1, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendbackXZ[count] = fptr[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsendXZ);

  MPI_Issend(&lb->hl .sendbackXZ[0], nsendXZ, MPI_DOUBLE, pbackY, tagb, comm, &lb->hl.sendreq[2]);
  MPI_Issend(&lb->hl .sendforwXZ[0], nsendXZ, MPI_DOUBLE, pforwY, tagf, comm, &lb->hl.sendreq[3]);


  /* Finally, Send in the z-direction (XY plane) */
  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 1; ic <= nlocal[X]; ic++) {
	for (jc = 1; jc <= nlocal[Y]; jc++) {

	  index = coords_index(ic, jc, nlocal[Z]);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendforwXY[count] = fptr[indexreal];	

	  index = coords_index(ic, jc, 1);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendbackXY[count] = fptr[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsendXY);

  MPI_Issend(&lb->hl .sendbackXY[0], nsendXY, MPI_DOUBLE, pbackZ, tagb, comm, &lb->hl.sendreq[4]);
  MPI_Issend(&lb->hl .sendforwXY[0], nsendXY, MPI_DOUBLE, pforwZ, tagf, comm, &lb->hl.sendreq[5]);

  return;

}

/*****************************************************************************
 *
 *  halo_edges
 *
 *  Sends 12 MPI messages (the edges) for the Non-Blocking version.
 *
 *****************************************************************************/


void halo_edges(lb_t * lb) {

  int ic, jc, kc;
  int n, p;
  int index, indexhalo, indexreal;
  int count;
  int nlocal[3];

  const int tagnn = 903;
  const int tagnp = 904;
  const int tagpn = 905;
  const int tagpp = 906;

  /* Ranks of neighbouring edges.
     Xpp, X is the direction, parallel to which edges are
     sent pp refers to the YZ directions respectively*/
  int Xpp, Xpn, Xnp, Xnn, Ypp, Ypn, Ynp, Ynn, Zpp, Zpn, Znp, Znn;

  MPI_Comm comm = cart_comm();

  assert(lb);

  coords_nlocal(nlocal);

  double* fptr; /*pointer to the distribution*/
  if (get_step()) /* we are in the timestep, so use target copy */
    fptr=lb->tcopy->f;
  else
    fptr=lb->f;  /* we are not in a timestep, use host copy */

  int nsendX, nsendY, nsendZ;
  nsendX = NVEL*lb->ndist*nlocal[X];
  nsendY = NVEL*lb->ndist*nlocal[Y]; 
  nsendZ = NVEL*lb->ndist*nlocal[Z];

  /* Allocate message sizes for edges send/receives */
  lb->hl .sendXnn = (double *) malloc(nsendX*sizeof(double));
  lb->hl .recvXnn = (double *) malloc(nsendX*sizeof(double));
  lb->hl .sendXnp = (double *) malloc(nsendX*sizeof(double));
  lb->hl .recvXnp = (double *) malloc(nsendX*sizeof(double));
  lb->hl .sendXpn = (double *) malloc(nsendX*sizeof(double));
  lb->hl .recvXpn = (double *) malloc(nsendX*sizeof(double));
  lb->hl .sendXpp = (double *) malloc(nsendX*sizeof(double));
  lb->hl .recvXpp = (double *) malloc(nsendX*sizeof(double));
  lb->hl .sendYnn = (double *) malloc(nsendY*sizeof(double));
  lb->hl .recvYnn = (double *) malloc(nsendY*sizeof(double));
  lb->hl .sendYnp = (double *) malloc(nsendY*sizeof(double));
  lb->hl .recvYnp = (double *) malloc(nsendY*sizeof(double));

  lb->hl .sendYpn = (double *) malloc(nsendY*sizeof(double));
  lb->hl .recvYpn = (double *) malloc(nsendY*sizeof(double));
  lb->hl .sendYpp = (double *) malloc(nsendY*sizeof(double));
  lb->hl .recvYpp = (double *) malloc(nsendY*sizeof(double));
  lb->hl .sendZnn = (double *) malloc(nsendZ*sizeof(double));
  lb->hl .recvZnn = (double *) malloc(nsendZ*sizeof(double));
  lb->hl .sendZnp = (double *) malloc(nsendZ*sizeof(double));
  lb->hl .recvZnp = (double *) malloc(nsendZ*sizeof(double));
  lb->hl .sendZpn = (double *) malloc(nsendZ*sizeof(double));
  lb->hl .recvZpn = (double *) malloc(nsendZ*sizeof(double));
  lb->hl .sendZpp = (double *) malloc(nsendZ*sizeof(double));
  lb->hl .recvZpp = (double *) malloc(nsendZ*sizeof(double));

  /* Receive edges parallel to x-direction*/
  Xnn = nonblocking_cart_neighb(MNN);
  Xnp = nonblocking_cart_neighb(MNP);
  Xpn = nonblocking_cart_neighb(MPN);
  Xpp = nonblocking_cart_neighb(MPP);

  MPI_Irecv(&lb->hl.recvXnn[0], nsendX, MPI_DOUBLE, Xnn, tagpp, comm, &lb->hl.recvreq[6]);
  MPI_Irecv(&lb->hl.recvXnp[0], nsendX, MPI_DOUBLE, Xnp, tagpn, comm, &lb->hl.recvreq[7]);
  MPI_Irecv(&lb->hl.recvXpn[0], nsendX, MPI_DOUBLE, Xpn, tagnp, comm, &lb->hl.recvreq[8]);
  MPI_Irecv(&lb->hl.recvXpp[0], nsendX, MPI_DOUBLE, Xpp, tagnn, comm, &lb->hl.recvreq[9]);

  /* Receive edges parallel to y-direction*/
  Ynn = nonblocking_cart_neighb(NMN);
  Ynp = nonblocking_cart_neighb(NMP);
  Ypn = nonblocking_cart_neighb(PMN);
  Ypp = nonblocking_cart_neighb(PMP);

  MPI_Irecv(&lb->hl.recvYnn[0], nsendY, MPI_DOUBLE, Ynn, tagpp, comm, &lb->hl.recvreq[10]);
  MPI_Irecv(&lb->hl.recvYnp[0], nsendY, MPI_DOUBLE, Ynp, tagpn, comm, &lb->hl.recvreq[11]);
  MPI_Irecv(&lb->hl.recvYpn[0], nsendY, MPI_DOUBLE, Ypn, tagnp, comm, &lb->hl.recvreq[12]);
  MPI_Irecv(&lb->hl.recvYpp[0], nsendY, MPI_DOUBLE, Ypp, tagnn, comm, &lb->hl.recvreq[13]);

  /* Receive edges parallel to z-direction*/
  Znn = nonblocking_cart_neighb(NNM);
  Znp = nonblocking_cart_neighb(NPM);
  Zpn = nonblocking_cart_neighb(PNM);
  Zpp = nonblocking_cart_neighb(PPM);

  MPI_Irecv(&lb->hl.recvZnn[0], nsendZ, MPI_DOUBLE, Znn, tagpp, comm, &lb->hl.recvreq[14]);
  MPI_Irecv(&lb->hl.recvZnp[0], nsendZ, MPI_DOUBLE, Znp, tagpn, comm, &lb->hl.recvreq[15]);
  MPI_Irecv(&lb->hl.recvZpn[0], nsendZ, MPI_DOUBLE, Zpn, tagnp, comm, &lb->hl.recvreq[16]);
  MPI_Irecv(&lb->hl.recvZpp[0], nsendZ, MPI_DOUBLE, Zpp, tagnn, comm, &lb->hl.recvreq[17]);

  /* Send edges parallel to x-direction */
  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 1; ic <= nlocal[X]; ic++) {

	  index = coords_index(ic, 1, 1);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendXnn[count] = fptr[indexreal];

	  index = coords_index(ic, 1, nlocal[Z]);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendXnp[count] = fptr[indexreal];

	  index = coords_index(ic, nlocal[Y], 1);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendXpn[count] = fptr[indexreal];

	  index = coords_index(ic, nlocal[Y], nlocal[Z]);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendXpp[count] = fptr[indexreal];
	  ++count;
      }
    }
  }
  assert(count == nsendX);

  MPI_Issend(&lb->hl .sendXpp[0], nsendX, MPI_DOUBLE, Xpp, tagpp, comm, &lb->hl.sendreq[6]);
  MPI_Issend(&lb->hl .sendXpn[0], nsendX, MPI_DOUBLE, Xpn, tagpn, comm, &lb->hl.sendreq[7]);
  MPI_Issend(&lb->hl .sendXnp[0], nsendX, MPI_DOUBLE, Xnp, tagnp, comm, &lb->hl.sendreq[8]);
  MPI_Issend(&lb->hl .sendXnn[0], nsendX, MPI_DOUBLE, Xnn, tagnn, comm, &lb->hl.sendreq[9]);

  /* Send edges parallel to y-direction */
  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	  index = coords_index(1, jc, 1);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendYnn[count] = fptr[indexreal];

	  index = coords_index(1, jc, nlocal[Z]);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendYnp[count] = fptr[indexreal];

	  index = coords_index(nlocal[X], jc, 1);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendYpn[count] = fptr[indexreal];

	  index = coords_index(nlocal[X], jc, nlocal[Z]);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendYpp[count] = fptr[indexreal];
	  ++count;
      }
    }
  }
  assert(count == nsendY);

  MPI_Issend(&lb->hl .sendYpp[0], nsendY, MPI_DOUBLE, Ypp, tagpp, comm, &lb->hl.sendreq[10]);
  MPI_Issend(&lb->hl .sendYpn[0], nsendY, MPI_DOUBLE, Ypn, tagpn, comm, &lb->hl.sendreq[11]);
  MPI_Issend(&lb->hl .sendYnp[0], nsendY, MPI_DOUBLE, Ynp, tagnp, comm, &lb->hl.sendreq[12]);
  MPI_Issend(&lb->hl .sendYnn[0], nsendY, MPI_DOUBLE, Ynn, tagnn, comm, &lb->hl.sendreq[13]);

  /* Send edges parallel to z-direction */
  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(1, 1, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendZnn[count] = fptr[indexreal];

	  index = coords_index(1, nlocal[Y], kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendZnp[count] = fptr[indexreal];

	  index = coords_index(nlocal[X], 1, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendZpn[count] = fptr[indexreal];

	  index = coords_index(nlocal[X], nlocal[Y], kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->hl .sendZpp[count] = fptr[indexreal];
	  ++count;
      }
    }
  }
  assert(count == nsendZ);

  MPI_Issend(&lb->hl .sendZpp[0] , nsendZ, MPI_DOUBLE, Zpp, tagpp, comm, &lb->hl.sendreq[14]);
  MPI_Issend(&lb->hl .sendZpn[0], nsendZ, MPI_DOUBLE, Zpn, tagpn, comm, &lb->hl.sendreq[15]);
  MPI_Issend(&lb->hl .sendZnp[0], nsendZ, MPI_DOUBLE, Znp, tagnp, comm, &lb->hl.sendreq[16]);
  MPI_Issend(&lb->hl .sendZnn[0], nsendZ, MPI_DOUBLE, Znn, tagnn, comm, &lb->hl.sendreq[17]);

  return;
}

/*****************************************************************************
 *
 *  halo_corners
 *
 *  Sends 8 MPI messages (the corners) for the Non-Blocking version.
 *
 *****************************************************************************/

void halo_corners(lb_t * lb) {

  int ic, jc, kc;
  int n, p;
  int count;
  int index, indexhalo, indexreal;
  int nlocal[3];

  const int tagnnn = 907;
  const int tagnnp = 908;
  const int tagnpn = 909;
  const int tagnpp = 910;
  const int tagpnn = 911;
  const int tagpnp = 912;
  const int tagppn = 913;
  const int tagppp = 914;

  /* Ranks of neighbouring corners XYZ direction*/
  int ppp, ppn, pnp, pnn, npp, npn, nnp, nnn;

  MPI_Comm comm = cart_comm();

  assert(lb);

  coords_nlocal(nlocal);

  double* fptr; 
  if (get_step()) 
    fptr=lb->tcopy->f;
  else
    fptr=lb->f; 

  int nsend;
  nsend = NVEL*lb->ndist*1;

  /* Allocate message sizes for plane send/receives */
  lb->hl .sendnnn = (double *) malloc(nsend*sizeof(double));
  lb->hl .sendnnp = (double *) malloc(nsend*sizeof(double));
  lb->hl .sendnpn = (double *) malloc(nsend*sizeof(double));
  lb->hl .sendnpp = (double *) malloc(nsend*sizeof(double));
  lb->hl .sendpnn = (double *) malloc(nsend*sizeof(double));
  lb->hl .sendpnp = (double *) malloc(nsend*sizeof(double));
  lb->hl .sendppn = (double *) malloc(nsend*sizeof(double));
  lb->hl .sendppp = (double *) malloc(nsend*sizeof(double));

  lb->hl .recvnnn = (double *) malloc(nsend*sizeof(double));
  lb->hl .recvnnp = (double *) malloc(nsend*sizeof(double));
  lb->hl .recvnpn = (double *) malloc(nsend*sizeof(double));
  lb->hl .recvnpp = (double *) malloc(nsend*sizeof(double));
  lb->hl .recvpnn = (double *) malloc(nsend*sizeof(double));
  lb->hl .recvpnp = (double *) malloc(nsend*sizeof(double));
  lb->hl .recvppn = (double *) malloc(nsend*sizeof(double));
  lb->hl .recvppp = (double *) malloc(nsend*sizeof(double));

  nnn = nonblocking_cart_neighb(NNN);
  nnp = nonblocking_cart_neighb(NNP);
  npn = nonblocking_cart_neighb(NPN);
  npp = nonblocking_cart_neighb(NPP);
  pnn = nonblocking_cart_neighb(PNN);
  pnp = nonblocking_cart_neighb(PNP);
  ppn = nonblocking_cart_neighb(PPN);
  ppp = nonblocking_cart_neighb(PPP);

  MPI_Irecv(&lb->hl.recvnnn[0], nsend, MPI_DOUBLE, nnn, tagppp, comm, &lb->hl.recvreq[18]);
  MPI_Irecv(&lb->hl.recvnnp[0], nsend, MPI_DOUBLE, nnp, tagppn, comm, &lb->hl.recvreq[19]);
  MPI_Irecv(&lb->hl.recvnpn[0], nsend, MPI_DOUBLE, npn, tagpnp, comm, &lb->hl.recvreq[20]);
  MPI_Irecv(&lb->hl.recvnpp[0], nsend, MPI_DOUBLE, npp, tagpnn, comm, &lb->hl.recvreq[21]);
  MPI_Irecv(&lb->hl.recvpnn[0], nsend, MPI_DOUBLE, pnn, tagnpp, comm, &lb->hl.recvreq[22]);
  MPI_Irecv(&lb->hl.recvpnp[0], nsend, MPI_DOUBLE, pnp, tagnpn, comm, &lb->hl.recvreq[23]);
  MPI_Irecv(&lb->hl.recvppn[0], nsend, MPI_DOUBLE, ppn, tagnnp, comm, &lb->hl.recvreq[24]);
  MPI_Irecv(&lb->hl.recvppp[0], nsend, MPI_DOUBLE, ppp, tagnnn, comm, &lb->hl.recvreq[25]);

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {

      index = coords_index(1, 1, 1);
      indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      lb->hl .sendnnn[count] = fptr[indexreal];

      index = coords_index(1, 1, nlocal[Z]);
      indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      lb->hl .sendnnp[count] = fptr[indexreal];

      index = coords_index(1, nlocal[Y], 1);
      indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      lb->hl .sendnpn[count] = fptr[indexreal];

      index = coords_index(1, nlocal[Y], nlocal[Z]);
      indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      lb->hl .sendnpp[count] = fptr[indexreal];

      index = coords_index(nlocal[X], 1, 1);
      indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      lb->hl .sendpnn[count] = fptr[indexreal];

      index = coords_index(nlocal[X], 1, nlocal[Z]);
      indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      lb->hl .sendpnp[count] = fptr[indexreal];

      index = coords_index(nlocal[X], nlocal[Y], 1);
      indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      lb->hl .sendppn[count] = fptr[indexreal];

      index = coords_index(nlocal[X], nlocal[Y], nlocal[Z]);
      indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      lb->hl .sendppp[count] = fptr[indexreal];
      count++;
    }
  }

  MPI_Issend(&lb->hl .sendppp[0], nsend, MPI_DOUBLE, ppp, tagppp, comm, &lb->hl.sendreq[18]);
  MPI_Issend(&lb->hl .sendppn[0], nsend, MPI_DOUBLE, ppn, tagppn, comm, &lb->hl.sendreq[19]);
  MPI_Issend(&lb->hl .sendpnp[0], nsend, MPI_DOUBLE, pnp, tagpnp, comm, &lb->hl.sendreq[20]);
  MPI_Issend(&lb->hl .sendpnn[0], nsend, MPI_DOUBLE, pnn, tagpnn, comm, &lb->hl.sendreq[21]);
  MPI_Issend(&lb->hl .sendnpp[0], nsend, MPI_DOUBLE, npp, tagnpp, comm, &lb->hl.sendreq[22]);
  MPI_Issend(&lb->hl .sendnpn[0], nsend, MPI_DOUBLE, npn, tagnpn, comm, &lb->hl.sendreq[23]);
  MPI_Issend(&lb->hl .sendnnp[0], nsend, MPI_DOUBLE, nnp, tagnnp, comm, &lb->hl.sendreq[24]);
  MPI_Issend(&lb->hl .sendnnn[0], nsend, MPI_DOUBLE, nnn, tagnnn, comm, &lb->hl.sendreq[25]);

  return;
}


/*****************************************************************************
 *
 *  unpack_halo_buffers
 *
 *  Unpacks buffers from MPI messages into Halo values for next step of simulation
 *
 *****************************************************************************/

void unpack_halo_buffers(lb_t * lb) {

  int ic, jc, kc;
  int n, p;
  int index, indexhalo;
  int count;
  int nlocal[3];

  coords_nlocal(nlocal);

  double* fptr; /*pointer to the distribution*/
  if (get_step()) /* we are in the timestep, so use target copy */
    fptr=lb->tcopy->f;
  else
    fptr=lb->f;  /* we are not in a timestep, use host copy */

  int nsendXZ, nsendYZ, nsendXY;
  nsendYZ = NVEL*lb->ndist*nlocal[Y]*nlocal[Z];
  nsendXZ = NVEL*lb->ndist*nlocal[X]*nlocal[Z];
  nsendXY = NVEL*lb->ndist*nlocal[X]*nlocal[Y];

  int nsendX, nsendY, nsendZ;
  nsendX = NVEL*lb->ndist*nlocal[X];
  nsendY = NVEL*lb->ndist*nlocal[Y]; 
  nsendZ = NVEL*lb->ndist*nlocal[Z];

  int nsend;
  nsend = NVEL*lb->ndist*1;

  /* Unpack Planes */

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(0, jc, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvbackYZ[count];

	  index = coords_index(nlocal[X] + 1, jc, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvforwYZ[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsendYZ);

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 1; ic <= nlocal[X]; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(ic, 0, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvbackXZ[count];

	  index = coords_index(ic, nlocal[Y] + 1, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvforwXZ[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsendXZ);

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 1; ic <= nlocal[X]; ic++) {
	for (jc = 1; jc <= nlocal[Y]; jc++) {

	  index = coords_index(ic, jc, 0);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvbackXY[count];

	  index = coords_index(ic, jc, nlocal[Z] + 1);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvforwXY[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsendXY);

  /* Free memory for planes buffers */
  free(lb->hl .sendforwYZ);
  free(lb->hl .recvforwYZ);
  free(lb->hl .sendbackYZ);
  free(lb->hl .recvbackYZ);
  free(lb->hl .sendforwXZ);
  free(lb->hl .recvforwXZ);
  free(lb->hl .sendbackXZ);
  free(lb->hl .recvbackXZ);
  free(lb->hl .sendforwXY);
  free(lb->hl .recvforwXY);
  free(lb->hl .sendbackXY);
  free(lb->hl .recvbackXY);

  /* Unpack Edges */

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 1; ic <= nlocal[X]; ic++) {

	  index = coords_index(ic, nlocal[Y] + 1, nlocal[Z] + 1);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvXpp[count];

	  index = coords_index(ic, nlocal[Y] + 1, 0);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvXpn[count];

	  index = coords_index(ic, 0, nlocal[Z] + 1);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvXnp[count];

	  index = coords_index(ic, 0, 0);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvXnn[count];
	  ++count;
      }
    }
  }
  assert(count == nsendX);

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	  index = coords_index(nlocal[X] + 1, jc, nlocal[Z] + 1);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvYpp[count];

	  index = coords_index(nlocal[X] + 1, jc, 0);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvYpn[count];

	  index = coords_index(0, jc, nlocal[Z] + 1);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvYnp[count];

	  index = coords_index(0, jc, 0);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvYnn[count];
	  ++count;
      }
    }
  }
  assert(count == nsendY);

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(nlocal[X] + 1, nlocal[Y] + 1, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvZpp[count];

	  index = coords_index(nlocal[X] + 1, 0, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvZpn[count];

	  index = coords_index(0, nlocal[Y] + 1, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvZnp[count];

	  index = coords_index(0, 0, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  fptr[indexhalo] = lb->hl .recvZnn[count];
	  ++count;
      }
    }
  }
  assert(count == nsendZ);

  /* Free memory for planes buffers */
  free(lb->hl .sendXnn);
  free(lb->hl .recvXnn);
  free(lb->hl .sendXnp);
  free(lb->hl .recvXnp);
  free(lb->hl .sendXpn);
  free(lb->hl .recvXpn);
  free(lb->hl .sendXpp);
  free(lb->hl .recvXpp);
  free(lb->hl .sendYnn);
  free(lb->hl .recvYnn);
  free(lb->hl .sendYnp);
  free(lb->hl .recvYnp);

  free(lb->hl .sendYpn);
  free(lb->hl .recvYpn);
  free(lb->hl .sendYpp);
  free(lb->hl .recvYpp);
  free(lb->hl .sendZnn);
  free(lb->hl .recvZnn);
  free(lb->hl .sendZnp);
  free(lb->hl .recvZnp);
  free(lb->hl .sendZpn);
  free(lb->hl .recvZpn);
  free(lb->hl .sendZpp);
  free(lb->hl .recvZpp);

  /* Unpack corners buffers */

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {

      index = coords_index(nlocal[X] + 1, nlocal[Y] + 1, nlocal[Z] + 1);
      indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      fptr[indexhalo] = lb->hl .recvppp[count];

      index = coords_index(nlocal[X] + 1, nlocal[Y] + 1, 0);
      indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      fptr[indexhalo] = lb->hl .recvppn[count];

      index = coords_index(nlocal[X] + 1, 0, nlocal[Z] + 1);
      indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      fptr[indexhalo] = lb->hl .recvpnp[count];

      index = coords_index(nlocal[X] + 1, 0, 0);
      indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      fptr[indexhalo] = lb->hl .recvpnn[count];

      index = coords_index(0, nlocal[Y] + 1, nlocal[Z] + 1);
      indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      fptr[indexhalo] = lb->hl .recvnpp[count];

      index = coords_index(0, nlocal[Y] + 1, 0);
      indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      fptr[indexhalo] = lb->hl .recvnpn[count];

      index = coords_index(0, 0, nlocal[Z] + 1);
      indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      fptr[indexhalo] = lb->hl .recvnnp[count];

      index = coords_index(0, 0, 0);
      indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
      fptr[indexhalo] = lb->hl .recvnnn[count];
      count++;
    }
  }

  /* Free memory for corner buffers */
  free(lb->hl .sendnnn);
  free(lb->hl .sendnnp);
  free(lb->hl .sendnpn);
  free(lb->hl .sendnpp);
  free(lb->hl .sendpnn);
  free(lb->hl .sendpnp);
  free(lb->hl .sendppn);
  free(lb->hl .sendppp);

  free(lb->hl .recvnnn);
  free(lb->hl .recvnnp);
  free(lb->hl .recvnpn);
  free(lb->hl .recvnpp);
  free(lb->hl .recvpnn);
  free(lb->hl .recvpnp);
  free(lb->hl .recvppn);
  free(lb->hl .recvppp);

  return;
}


