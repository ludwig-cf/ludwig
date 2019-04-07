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
 *  (c) 2010-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Erlend Davidson provided the original reduced halo swap.
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
#include "io_harness.h"

const double cs2  = (1.0/3.0);
const double rcs2 = 3.0;

static int lb_mpi_init(lb_t * lb);
static int lb_set_types(int, MPI_Datatype *);
static int lb_set_blocks(lb_t * lb, int, int *, int, const int *);
static int lb_set_displacements(lb_t * lb, int, MPI_Aint *, int, const int *);
static int lb_f_read(FILE *, int index, void * self);
static int lb_f_read_ascii(FILE *, int index, void * self);
static int lb_f_write(FILE *, int index, void * self);
static int lb_f_write_ascii(FILE *, int index, void * self);
static int lb_rho_write(FILE *, int index, void * self);
static int lb_rho_write_ascii(FILE *, int index, void * self);
static int lb_model_param_init(lb_t * lb);

static __constant__ lb_collide_param_t static_param;

/****************************************************************************
 *
 *  lb_create_ndist
 *
 *  Allocate memory for lb_data_s only.
 *
 ****************************************************************************/

__host__ int lb_create_ndist(pe_t * pe, cs_t * cs, int ndist, lb_t ** plb) {

  lb_t * lb = NULL;

  assert(pe);
  assert(cs);
  assert(ndist > 0);
  assert(plb);

  lb = (lb_t *) calloc(1, sizeof(lb_t));
  assert(lb);
  if (lb == NULL) pe_fatal(pe, "calloc(1, lb_t) failed\n");

  lb->param = (lb_collide_param_t *) calloc(1, sizeof(lb_collide_param_t));
  if (lb->param == NULL) pe_fatal(pe, "calloc(1, lb_collide_param_t) failed\n");

  lb->pe = pe;
  lb->cs = cs;
  lb->ndist = ndist;
  lb->model = DATA_MODEL;
  lb->nrelax = LB_RELAXATION_M10;

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

__host__ int lb_create(pe_t * pe, cs_t * cs, lb_t ** plb) {

  assert(pe);
  assert(cs);
  assert(plb);

  return lb_create_ndist(pe, cs, 1, plb);
}

/*****************************************************************************
 *
 *  lb_free
 *
 *  Clean up.
 *
 *****************************************************************************/

__host__ int lb_free(lb_t * lb) {

  int ndevice;
  double * tmp;

  assert(lb);

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpMemcpy(&tmp, &lb->target->f, sizeof(double *), tdpMemcpyDeviceToHost); 
    tdpFree(tmp);

    tdpMemcpy(&tmp, &lb->target->fprime, sizeof(double *),
	      tdpMemcpyDeviceToHost); 
    tdpFree(tmp);
    tdpFree(lb->target);
  }

  if (lb->halo) halo_swap_free(lb->halo);
  if (lb->io_info) io_info_free(lb->io_info);
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

  free(lb->param);
  free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  lb_memcpy
 *
 *****************************************************************************/

__host__ int lb_memcpy(lb_t * lb, tdpMemcpyKind flag) {

  int ndevice;
  double * tmpf = NULL;

  assert(lb);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Make sure we alias */
    assert(lb->target == lb);
  }
  else {

    assert(lb->target);

    tdpMemcpy(&tmpf, &lb->target->f, sizeof(double *), tdpMemcpyDeviceToHost);

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpMemcpy(&lb->target->ndist, &lb->ndist, sizeof(int), flag); 
      tdpMemcpy(&lb->target->nsite, &lb->nsite, sizeof(int), flag); 
      tdpMemcpy(&lb->target->model, &lb->model, sizeof(int), flag);
      tdpMemcpy(tmpf, lb->f, NVEL*lb->nsite*lb->ndist*sizeof(double), flag);
      break;
    case tdpMemcpyDeviceToHost:
      tdpMemcpy(lb->f, tmpf, NVEL*lb->nsite*lb->ndist*sizeof(double), flag);
      break;
    default:
      pe_fatal(lb->pe, "Bad flag in lb_memcpy\n");
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
 
__host__ int lb_init(lb_t * lb) {

  int nlocal[3];
  int nx, ny, nz;
  int nhalolocal = 1;
  int ndata;
  int nhalo;
  int ndevice;
  double * tmp;

  assert(lb);

  cs_nhalo(lb->cs, &nhalo);
  cs_nlocal(lb->cs, nlocal);

  nx = nlocal[X] + 2*nhalo;
  ny = nlocal[Y] + 2*nhalo;
  nz = nlocal[Z] + 2*nhalo;
  lb->nsite = nx*ny*nz;

  /* The total number of distribution data is then... */

  ndata = lb->nsite*lb->ndist*NVEL;
#ifndef OLD_DATA
  lb->f = (double  *) malloc(ndata*sizeof(double));
  if (lb->f == NULL) pe_fatal(lb->pe, "malloc(distributions) failed\n");

  lb->fprime = (double  *) malloc(ndata*sizeof(double));
  if (lb->fprime == NULL) pe_fatal(lb->pe, "malloc(distributions) failed\n");
#else
  lb->f = (double  *) mem_aligned_malloc(MEM_PAGESIZE, ndata*sizeof(double));
  if (lb->f == NULL) pe_fatal(lb->pe, "malloc(distributions) failed\n");

  lb->fprime = (double *) mem_aligned_malloc(MEM_PAGESIZE, ndata*sizeof(double));
  if (lb->fprime == NULL) pe_fatal(lb->pe, "malloc(distributions) failed\n");
#endif

  /* Allocate target copy of structure or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    lb->target = lb;
  }
  else {
    lb_collide_param_t * ptmp  = NULL;

    tdpMalloc((void **) &lb->target, sizeof(lb_t));
    tdpMemset(lb->target, 0, sizeof(lb_t));

    tdpMalloc((void **) &tmp, ndata*sizeof(double));
    tdpMemset(tmp, 0, ndata*sizeof(double));
    tdpMemcpy(&lb->target->f, &tmp, sizeof(double *), tdpMemcpyHostToDevice);
 
    tdpMalloc((void **) &tmp, ndata*sizeof(double));
    tdpMemset(tmp, 0, ndata*sizeof(double));
    tdpMemcpy(&lb->target->fprime, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);

    tdpGetSymbolAddress((void **) &ptmp, tdpSymbol(static_param));
    tdpMemcpy(&lb->target->param, &ptmp, sizeof(lb_collide_param_t *),
	      tdpMemcpyHostToDevice);
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
  lb_model_param_init(lb);
  lb_halo_set(lb, LB_HALO_FULL);
  lb_memcpy(lb, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  lb_collide_param_commit
 *
 *  TODO: responsibility for initialisation of various parameters
 *        is rather diffuse; needs checking.
 *
 *****************************************************************************/

__host__ int lb_collide_param_commit(lb_t * lb) {

  assert(lb);

  tdpMemcpyToSymbol(tdpSymbol(static_param), lb->param,
		    sizeof(lb_collide_param_t), 0, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  lb_model_param_init
 *
 *****************************************************************************/

static int lb_model_param_init(lb_t * lb) {

  int ia, ib, p;

  assert(lb);
  assert(lb->param);

  lb->param->nsite = lb->nsite;
  lb->param->ndist = lb->ndist;

  for (p = 0; p < NVEL; p++) {
    lb->param->wv[p] = wv[p];
    for (ia = 0; ia < 3; ia++) {
      lb->param->cv[p][ia] = cv[p][ia];
    }
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	lb->param->q[p][ia][ib] = q_[p][ia][ib];
      }
    }
  }

  for (ia = 0; ia < NVEL; ia++) {
    for (ib = 0; ib < NVEL; ib++) {
      lb->param->ma[ia][ib] = ma_[ia][ib];
      lb->param->mi[ia][ib] = mi_[ia][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_nvel
 *
 *  Return number of velocities at runtime.
 *
 *****************************************************************************/

__host__ int lb_nvel(lb_t * lb, int * nvel) {

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

__host__ int lb_ndim(lb_t * lb, int * ndim) {

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

__host__ int lb_nblock(lb_t * lb, int dim, int * nblock) {

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

__host__ int lb_init_rest_f(lb_t * lb, double rho0) {

  int nlocal[3];
  int ic, jc, kc, index;

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(lb->cs, ic, jc, kc);
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
  MPI_Aint extent;
  MPI_Aint * disp_fwd;
  MPI_Aint * disp_bwd;
  MPI_Datatype * types;
  MPI_Datatype tmpf, tmpb;

  assert(lb);

  cs_nhalo(lb->cs, &nhalo);
  cs_nlocal(lb->cs, nlocal);

  nx = nlocal[X] + 2*nhalo;
  ny = nlocal[Y] + 2*nhalo;
  nz = nlocal[Z] + 2*nhalo;

  /* extent of single site (AOS) */
  extent = NVEL*lb->ndist*sizeof(double);

  /* X direction */

  count = lb->ndist*CVXBLOCK;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  lb_set_types(count, types);
  lb_set_blocks(lb, count, blocklen, CVXBLOCK, xblocklen_cv);
  lb_set_displacements(lb, count, disp_fwd, CVXBLOCK, xdisp_fwd_cv);
  lb_set_displacements(lb, count, disp_bwd, CVXBLOCK, xdisp_bwd_cv);

  MPI_Type_create_struct(count, blocklen, disp_fwd, types, &tmpf);
  MPI_Type_create_struct(count, blocklen, disp_bwd, types, &tmpb);
  MPI_Type_commit(&tmpf);
  MPI_Type_commit(&tmpb);
  MPI_Type_create_resized(tmpf, 0, extent, &lb->site_x[CS_FORW]);
  MPI_Type_create_resized(tmpb, 0, extent, &lb->site_x[CS_BACK]);
  MPI_Type_free(&tmpf);
  MPI_Type_free(&tmpb);

  MPI_Type_commit(&lb->site_x[CS_FORW]);
  MPI_Type_commit(&lb->site_x[CS_BACK]);

  MPI_Type_contiguous(ny*nz, lb->site_x[FORWARD], &lb->plane_yz_reduced[FORWARD]);
  MPI_Type_contiguous(ny*nz, lb->site_x[BACKWARD], &lb->plane_yz_reduced[BACKWARD]);
  MPI_Type_commit(&lb->plane_yz_reduced[FORWARD]);
  MPI_Type_commit(&lb->plane_yz_reduced[BACKWARD]);

  free(blocklen);
  free(disp_fwd);
  free(disp_bwd);
  free(types);

  /* Y direction */

  count = lb->ndist*CVYBLOCK;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  lb_set_types(count, types);
  lb_set_blocks(lb, count, blocklen, CVYBLOCK, yblocklen_cv);
  lb_set_displacements(lb, count, disp_fwd, CVYBLOCK, ydisp_fwd_cv);
  lb_set_displacements(lb, count, disp_bwd, CVYBLOCK, ydisp_bwd_cv);

  MPI_Type_create_struct(count, blocklen, disp_fwd, types, &tmpf);
  MPI_Type_create_struct(count, blocklen, disp_bwd, types, &tmpb);
  MPI_Type_commit(&tmpf);
  MPI_Type_commit(&tmpb);
  MPI_Type_create_resized(tmpf, 0, extent, &lb->site_y[CS_FORW]);
  MPI_Type_create_resized(tmpb, 0, extent, &lb->site_y[CS_BACK]);
  MPI_Type_free(&tmpf);
  MPI_Type_free(&tmpb);

  MPI_Type_commit(&lb->site_y[CS_FORW]);
  MPI_Type_commit(&lb->site_y[CS_BACK]);

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

  count = lb->ndist*CVZBLOCK;

  blocklen = (int *) malloc(count*sizeof(int));
  disp_fwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  disp_bwd = (MPI_Aint *) malloc(count*sizeof(MPI_Aint));
  types    = (MPI_Datatype *) malloc(count*sizeof(MPI_Datatype));

  lb_set_types(count, types);
  lb_set_blocks(lb, count, blocklen, CVZBLOCK, zblocklen_cv);
  lb_set_displacements(lb, count, disp_fwd, CVZBLOCK, zdisp_fwd_cv);
  lb_set_displacements(lb, count, disp_bwd, CVZBLOCK, zdisp_bwd_cv);

  MPI_Type_create_struct(count, blocklen, disp_fwd, types, &tmpf);
  MPI_Type_create_struct(count, blocklen, disp_bwd, types, &tmpb);
  MPI_Type_commit(&tmpf);
  MPI_Type_commit(&tmpb);
  MPI_Type_create_resized(tmpf, 0, extent, &lb->site_z[CS_FORW]);
  MPI_Type_create_resized(tmpb, 0, extent, &lb->site_z[CS_BACK]);
  MPI_Type_free(&tmpf);
  MPI_Type_free(&tmpb);

  MPI_Type_commit(&lb->site_z[CS_FORW]);
  MPI_Type_commit(&lb->site_z[CS_BACK]);

  MPI_Type_vector(nx*ny, 1, nz, lb->site_z[FORWARD], &lb->plane_xy_reduced[FORWARD]);
  MPI_Type_vector(nx*ny, 1, nz, lb->site_z[BACKWARD],
		  &lb->plane_xy_reduced[BACKWARD]);
  MPI_Type_commit(&lb->plane_xy_reduced[FORWARD]);
  MPI_Type_commit(&lb->plane_xy_reduced[BACKWARD]);

  free(blocklen);
  free(disp_fwd);
  free(disp_bwd);
  free(types);

  halo_swap_create_r2(lb->pe, lb->cs, 1, lb->nsite, lb->ndist, NVEL,&lb->halo);
  halo_swap_handlers_set(lb->halo, halo_swap_pack_rank1, halo_swap_unpack_rank1);

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

  for (n = 0; n < ntype; n++) {
    type[n] = MPI_DOUBLE;
  }

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

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < nbasic; p++) {
      block[n*nbasic + p] = basic_block[p];
    }
  }

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

  /* Lower end */
  MPI_Get_address(lb->f, &disp0);

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < nbasic; p++) {
      MPI_Get_address(lb->f + n*NVEL + disp_basic[p], &disp1);
      disp[n*nbasic + p] = disp1 - disp0;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_io_info_set
 *
 *****************************************************************************/

__host__ int lb_io_info_set(lb_t * lb, io_info_t * io_info, int form_in, int form_out) {

  char string[FILENAME_MAX];

  assert(lb);
  assert(io_info);

  lb->io_info = io_info;

  sprintf(string, "%1d x Distribution: d%dq%d", lb->ndist, NDIM, NVEL);

  io_info_set_name(lb->io_info, string);
  io_info_read_set(lb->io_info, IO_FORMAT_BINARY, lb_f_read);
  io_info_write_set(lb->io_info, IO_FORMAT_BINARY, lb_f_write);
  io_info_set_bytesize(lb->io_info, IO_FORMAT_BINARY, lb->ndist*NVEL*sizeof(double));
  io_info_read_set(lb->io_info, IO_FORMAT_ASCII, lb_f_read_ascii);
  io_info_write_set(lb->io_info, IO_FORMAT_ASCII, lb_f_write_ascii);
  io_info_format_set(lb->io_info, form_in, form_out);

  return 0;
}

/*****************************************************************************
 *
 *  lb_io_rho_set
 *
 *  There is no input for rho, as it is never required.
 *
 *****************************************************************************/

__host__ int lb_io_rho_set(lb_t * lb, io_info_t * io_rho, int form_in,
                           int form_out) {

  char string[FILENAME_MAX];

  assert(lb);
  assert(io_rho);

  lb->io_rho = io_rho;

  sprintf(string, "Fluid density (rho)");

  io_info_set_name(lb->io_rho, string);
  io_info_write_set(lb->io_rho, IO_FORMAT_BINARY, lb_rho_write);
  io_info_set_bytesize(lb->io_rho, IO_FORMAT_BINARY, sizeof(double));
  io_info_write_set(lb->io_rho, IO_FORMAT_ASCII, lb_rho_write_ascii);
  io_info_format_set(lb->io_rho, form_in, form_out);

  return 0;
}

/*****************************************************************************
 *
 *  lb_io_info
 *
 *****************************************************************************/

__host__ int lb_io_info(lb_t * lb, io_info_t ** io_info) {

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
 *  in each direction. Default target swap.
 *
 *****************************************************************************/

__host__ int lb_halo(lb_t * lb) {

  assert(lb);

  lb_halo_swap(lb, LB_HALO_TARGET);

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_swap
 *
 *  Specify the type of swap wanted.
 *
 *****************************************************************************/

__host__ int lb_halo_swap(lb_t * lb, lb_halo_enum_t flag) {

  double * data;
  const char * msg = "Attempting halo via struct with NSIMDVL > 1 or (AO)SOA";

  assert(lb);

  switch (flag) {
  case LB_HALO_HOST:
    lb_halo_via_copy(lb);
    break;
  case LB_HALO_TARGET:
    tdpMemcpy(&data, &lb->target->f, sizeof(double *), tdpMemcpyDeviceToHost);
    halo_swap_packed(lb->halo, data);
    break;
  case LB_HALO_FULL:
    if (NSIMDVL > 1) pe_fatal(lb->pe, "%s\n", msg);
    if (DATA_MODEL != DATA_MODEL_AOS) pe_fatal(lb->pe, "%s\n", msg);
    lb_halo_set(lb, LB_HALO_FULL);
    lb_halo_via_struct(lb);
    break;
  case LB_HALO_REDUCED:
    if (NSIMDVL > 1) pe_fatal(lb->pe, "%s\n", msg);
    if (DATA_MODEL != DATA_MODEL_AOS) pe_fatal(lb->pe, "%s\n", msg);
    lb_halo_set(lb, LB_HALO_REDUCED);
    lb_halo_via_struct(lb);
    break;
  default:
    /* Should not be here... */
    assert(0);
  }

  /* In the limited case  MODEL order and NSIMDVL is 1 the struct
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

__host__ int lb_halo_via_struct(lb_t * lb) {

  int ic, jc, kc;
  int ihalo, ireal;
  int nhalo;
  int pforw, pback;
  int nlocal[3];
  int mpi_cartsz[3];
  int periodic[3];

  const int tagf = 900;
  const int tagb = 901;

  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm;

  assert(lb);

  cs_nhalo(lb->cs, &nhalo);
  cs_nlocal(lb->cs, nlocal);
  cs_cart_comm(lb->cs, &comm);
  cs_periodic(lb->cs, periodic);
  cs_cartsz(lb->cs, mpi_cartsz);

  /* The x-direction (YZ plane) */

  if (mpi_cartsz[X] == 1) {
    if (periodic[X]) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  ihalo = lb->ndist*NVEL*cs_index(lb->cs, 0, jc, kc);
	  ireal = lb->ndist*NVEL*cs_index(lb->cs, nlocal[X], jc, kc);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*cs_index(lb->cs, nlocal[X]+1, jc, kc);
	  ireal = lb->ndist*NVEL*cs_index(lb->cs, 1, jc, kc);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {
    pforw = cs_cart_neighb(lb->cs, CS_FORW, X);
    pback = cs_cart_neighb(lb->cs, CS_BACK, X);
    ihalo = lb->ndist*NVEL*cs_index(lb->cs, nlocal[X] + 1, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_yz[BACKWARD],
	      pforw, tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*cs_index(lb->cs, 0, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_yz[FORWARD],
	      pback, tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*cs_index(lb->cs, 1, 1-nhalo, 1-nhalo);
    MPI_Issend(lb->f + ireal, 1, lb->plane_yz[BACKWARD],
	       pback, tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*cs_index(lb->cs, nlocal[X], 1-nhalo, 1-nhalo);
    MPI_Issend(lb->f + ireal, 1, lb->plane_yz[FORWARD],
	       pforw, tagf, comm, &request[3]);
    MPI_Waitall(4, request, status);
  }
  
  /* The y-direction (XZ plane) */

  if (mpi_cartsz[Y] == 1) {
    if (periodic[Y]) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  ihalo = lb->ndist*NVEL*cs_index(lb->cs, ic, 0, kc);
	  ireal = lb->ndist*NVEL*cs_index(lb->cs, ic, nlocal[Y], kc);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*cs_index(lb->cs, ic, nlocal[Y] + 1, kc);
	  ireal = lb->ndist*NVEL*cs_index(lb->cs, ic, 1, kc);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {
    pforw = cs_cart_neighb(lb->cs, CS_FORW, Y);
    pback = cs_cart_neighb(lb->cs, CS_BACK, Y);
    ihalo = lb->ndist*NVEL*cs_index(lb->cs, 1 - nhalo, nlocal[Y] + 1, 1 - nhalo);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_xz[BACKWARD],
	      pforw, tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*cs_index(lb->cs, 1 - nhalo, 0, 1 - nhalo);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_xz[FORWARD], pback,
	      tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*cs_index(lb->cs, 1 - nhalo, 1, 1 - nhalo);
    MPI_Issend(lb->f + ireal, 1, lb->plane_xz[BACKWARD], pback,
	       tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*cs_index(lb->cs, 1 - nhalo, nlocal[Y], 1 - nhalo);
    MPI_Issend(lb->f + ireal, 1, lb->plane_xz[FORWARD], pforw,
	       tagf, comm, &request[3]);
    MPI_Waitall(4, request, status);
  }
  
  /* Finally, z-direction (XY plane) */

  if (mpi_cartsz[Z] == 1) {
    if (periodic[Z]) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (jc = 0; jc <= nlocal[Y] + 1; jc++) {

	  ihalo = lb->ndist*NVEL*cs_index(lb->cs, ic, jc, 0);
	  ireal = lb->ndist*NVEL*cs_index(lb->cs, ic, jc, nlocal[Z]);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));

	  ihalo = lb->ndist*NVEL*cs_index(lb->cs, ic, jc, nlocal[Z] + 1);
	  ireal = lb->ndist*NVEL*cs_index(lb->cs, ic, jc, 1);
	  memcpy(lb->f + ihalo, lb->f + ireal, lb->ndist*NVEL*sizeof(double));
	}
      }
    }
  }
  else {
    pforw = cs_cart_neighb(lb->cs, CS_FORW, Z);
    pback = cs_cart_neighb(lb->cs, CS_BACK, Z);
    ihalo = lb->ndist*NVEL*cs_index(lb->cs, 1 - nhalo, 1 - nhalo, nlocal[Z] + 1);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_xy[BACKWARD], pforw,
	      tagb, comm, &request[0]);
    ihalo = lb->ndist*NVEL*cs_index(lb->cs, 1 - nhalo, 1 - nhalo, 0);
    MPI_Irecv(lb->f + ihalo, 1, lb->plane_xy[FORWARD], pback,
	      tagf, comm, &request[1]);
    ireal = lb->ndist*NVEL*cs_index(lb->cs, 1 - nhalo, 1 - nhalo, 1);
    MPI_Issend(lb->f + ireal, 1, lb->plane_xy[BACKWARD], pback,
	       tagb, comm, &request[2]);
    ireal = lb->ndist*NVEL*cs_index(lb->cs, 1 - nhalo, 1 - nhalo, nlocal[Z]);
    MPI_Issend(lb->f + ireal, 1, lb->plane_xy[FORWARD], pforw,
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

  if (nr != lb->ndist*NVEL) pe_fatal(lb->pe, "fread(lb) failed at %d\n", index);

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_read_ascii
 *
 *  Read one lattice site (index) worth of distributions.
 *  Note that read-write is always 'MODEL' order.
 *
 *****************************************************************************/

static int lb_f_read_ascii(FILE * fp, int index, void * self) {

  int n, p;
  int nr;
  pe_t * pe = NULL;
  lb_t * lb = (lb_t *) self;

  assert(fp);
  assert(lb);

  pe = lb->pe;

  /* skip output index */
  nr = fscanf(fp, "%*d %*d %*d");
  assert(nr == 1);

  nr = 0;
  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < NVEL; p++) {
      nr += fscanf(fp, "%le",
		   &lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)]);
    }
  }

  if (nr != lb->ndist*NVEL) pe_fatal(pe, "fread(lb) failed at %d\n", index);

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

  if (nw != lb->ndist*NVEL) pe_fatal(lb->pe, "fwrite(lb) failed at %d\n", index);

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_write_ascii
 *
 *  Write one lattice site (index) worth of distributions.
 *  Note that read/write is always 'MODEL' order.
 *
 *****************************************************************************/

static int lb_f_write_ascii(FILE * fp, int index, void * self) {

  int n, p;
  int nw = 0;
  lb_t * lb = (lb_t*) self;

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  int coords[3], noffset[3];

  assert(fp);
  assert(self);

  pe = lb->pe;
  cs = lb->cs;

  cs_index_to_ijk(cs, index, coords);
  cs_nlocal_offset(cs, noffset);

  /* write output index */
  fprintf(fp, "%d %d %d ", noffset[X]+coords[X], noffset[Y]+coords[Y], noffset[Z]+coords[Z]);

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < NVEL; p++) {
      fprintf(fp, "%le ", lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)]);
      nw++;
    }
  }
  fprintf(fp, "\n");

  if (nw != lb->ndist*NVEL) pe_fatal(pe, "fwrite(lb) failed at %d\n", index);

  return 0;
}

/******************************************************************************
 *
 *  lb_rho_write
 *
 *  Write density data to file (binary)
 *
 *****************************************************************************/

static int lb_rho_write(FILE * fp, int index, void * self) {

  size_t iw;
  double rho;
  lb_t * lb = (lb_t *) self;

  assert(fp);
  assert(lb);

  lb_0th_moment(lb, index, LB_RHO, &rho);
  iw = fwrite(&rho, sizeof(double), 1, fp);

  if (iw != 1) pe_fatal(lb->pe, "lb_rho-write failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  lb_rho_write_ascii
 *
 *****************************************************************************/

static int lb_rho_write_ascii(FILE * fp, int index, void * self) {

  int nwrite;
  double rho;
  lb_t * lb = (lb_t *) self;

  assert(fp);
  assert(lb);

  lb_0th_moment(lb, index, LB_RHO, &rho);
  nwrite = fprintf(fp, "%22.15e\n", rho);
  if (nwrite != 23) pe_fatal(lb->pe, "lb_rho_write_ascii failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_set
 *
 *  Set the actual halo datatype.
 *
 *****************************************************************************/

__host__ int lb_halo_set(lb_t * lb, lb_halo_enum_t type) {

  assert(lb);

  if (type == LB_HALO_REDUCED) {
    lb->plane_xy[FORWARD]  = lb->plane_xy_reduced[FORWARD];
    lb->plane_xy[BACKWARD] = lb->plane_xy_reduced[BACKWARD];
    lb->plane_xz[FORWARD]  = lb->plane_xz_reduced[FORWARD];
    lb->plane_xz[BACKWARD] = lb->plane_xz_reduced[BACKWARD];
    lb->plane_yz[FORWARD]  = lb->plane_yz_reduced[FORWARD];
    lb->plane_yz[BACKWARD] = lb->plane_yz_reduced[BACKWARD];
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
 *  lb_ndist
 *
 *  Return the number of distribution functions.
 *
 *****************************************************************************/

__host__ __device__ int lb_ndist(lb_t * lb, int * ndist) {

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

__host__ int lb_ndist_set(lb_t * lb, int n) {

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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
int lb_0th_moment(lb_t * lb, int index, lb_dist_enum_t nd, double * rho) {

  int p;

  assert(lb);
  assert(rho);
  assert(index >= 0 && index < lb->nsite);
  assert(nd < lb->ndist);

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

__host__
int lb_1st_moment(lb_t * lb, int index, lb_dist_enum_t nd, double g[3]) {

  int p;
  int n;

  assert(lb);
  assert(index >= 0 && index < lb->nsite);
  assert(nd < lb->ndist);

  /* Loop to 3 here to cover initialisation in D2Q9 (appears in momentum) */
  for (n = 0; n < 3; n++) {
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

__host__
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

__host__
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

__host__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
int lb_f_multi_index_part(lb_t * lb, int index, int n, double fv[NVEL][NSIMDVL],
			  int nv) {

  int p, iv;

  assert(lb);
  assert(n >= 0 && n < lb->ndist);
  assert(index >= 0 && index < lb->nsite);
  assert(0); /* Not used */

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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ int lb_order(lb_t * lb) {

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

__host__ int lb_halo_via_copy(lb_t * lb) {

  int ic, jc, kc;
  int n, p;
  int pforw, pback;
  int index, indexhalo, indexreal;
  int nsend, count;
  int nlocal[3];
  int mpi_cartsz[3];

  const int tagf = 900;
  const int tagb = 901;

  double * sendforw;
  double * sendback;
  double * recvforw;
  double * recvback;

  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm;

  assert(lb);

  cs_nlocal(lb->cs, nlocal);
  cs_cart_comm(lb->cs, &comm);
  cs_cartsz(lb->cs, mpi_cartsz);

  /* The x-direction (YZ plane) */

  nsend = NVEL*lb->ndist*nlocal[Y]*nlocal[Z];
  sendforw = (double *) malloc(nsend*sizeof(double));
  sendback = (double *) malloc(nsend*sizeof(double));
  recvforw = (double *) malloc(nsend*sizeof(double));
  recvback = (double *) malloc(nsend*sizeof(double));
  assert(sendback && sendforw);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(lb->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(lb->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(lb->pe, "malloc(recvforw) failed\n");
  if (recvforw == NULL) pe_fatal(lb->pe, "malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = cs_index(lb->cs, nlocal[X], jc, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendforw[count] = lb->f[indexreal];

	  index = cs_index(lb->cs, 1, jc, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = lb->f[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);

  if (mpi_cartsz[X] == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cs_cart_neighb(lb->cs, CS_FORW, X);
    pback = cs_cart_neighb(lb->cs, CS_BACK, X);

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

	  index = cs_index(lb->cs, 0, jc, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvback[count];

	  index = cs_index(lb->cs, nlocal[X] + 1, jc, kc);
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

  if (mpi_cartsz[X] > 1) {
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
  if (sendforw == NULL) pe_fatal(lb->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(lb->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(lb->pe, "malloc(recvforw) failed\n");
  if (recvforw == NULL) pe_fatal(lb->pe, "malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = cs_index(lb->cs, ic, nlocal[Y], kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendforw[count] = lb->f[indexreal];

	  index = cs_index(lb->cs, ic, 1, kc);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = lb->f[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);


  if (mpi_cartsz[Y] == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cs_cart_neighb(lb->cs, CS_FORW, Y);
    pback = cs_cart_neighb(lb->cs, CS_BACK, Y);

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

	  index = cs_index(lb->cs, ic, 0, kc);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvback[count];

	  index = cs_index(lb->cs, ic, nlocal[Y] + 1, kc);
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

  if (mpi_cartsz[Y] > 1) {
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
  if (sendforw == NULL) pe_fatal(lb->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(lb->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(lb->pe, "malloc(recvforw) failed\n");
  if (recvforw == NULL) pe_fatal(lb->pe, "malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < lb->ndist; n++) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (jc = 0; jc <= nlocal[Y] + 1; jc++) {

	  index = cs_index(lb->cs, ic, jc, nlocal[Z]);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendforw[count] = lb->f[indexreal];

	  index = cs_index(lb->cs, ic, jc, 1);
	  indexreal = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  sendback[count] = lb->f[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);

  if (mpi_cartsz[Z] == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cs_cart_neighb(lb->cs, CS_FORW, Z);
    pback = cs_cart_neighb(lb->cs, CS_BACK, Z);

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

	  index = cs_index(lb->cs, ic, jc, 0);
	  indexhalo = LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p);
	  lb->f[indexhalo] = recvback[count];

	  index = cs_index(lb->cs, ic, jc, nlocal[Z] + 1);
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

  if (mpi_cartsz[Z] > 1) {
    /* Wait for sends */
    MPI_Waitall(2, request + 2, status);
  }

  free(sendback);
  free(sendforw);
 
  return 0;
}
