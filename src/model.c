/*****************************************************************************
 *
 *  model.c
 *
 *  This encapsulates data/operations related to distributions.
 *  However, the implementation of the distribution is exposed
 *  for performance-critical operations ehich, for space
 *  considerations, are not included in this file.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Erlend Davidson provided the original reduced halo swap
 *  (now retired, the code that is).
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "lb_data.h"
#include "io_harness.h"

#include "timer.h"
#include "util.h"

static int lb_mpi_init(lb_t * lb);
static int lb_f_read(FILE *, int index, void * self);
static int lb_f_read_ascii(FILE *, int index, void * self);
static int lb_f_write(FILE *, int index, void * self);
static int lb_f_write_ascii(FILE *, int index, void * self);
static int lb_model_param_init(lb_t * lb);
static int lb_init(lb_t * lb);
static int lb_data_touch(lb_t * lb);

int lb_halo_dequeue_recv(lb_t * lb, const lb_halo_t * h, int irreq);
int lb_halo_enqueue_send(const lb_t * lb, lb_halo_t * h, int irreq);

static __constant__ lb_collide_param_t static_param;

/*****************************************************************************
 *
 *  lb_data_create
 *
 *****************************************************************************/

int lb_data_create(pe_t * pe, cs_t * cs, const lb_data_options_t * options,
		   lb_t ** lb) {

  lb_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(lb);

  obj = (lb_t *) calloc(1, sizeof(lb_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(1, lb_t) failed\n");

  /* Check options. An internal error if we haven't sanitised the options... */

  if (lb_data_options_valid(options) == 0) {
    pe_fatal(pe, "Internal error: lb_data_options not valid.\n");
  }

  obj->pe = pe;
  obj->cs = cs;
  obj->ndim = options->ndim;
  obj->nvel = options->nvel;
  obj->ndist = options->ndist;
  obj->nrelax = options->nrelax;
  obj->haloscheme = options->halo;

  /* Note there is some duplication of options/parameters */
  /* ... which should really be rationalised. */
  obj->opts = *options;

  lb_model_create(obj->nvel, &obj->model);

  /* Storage */

  {
    /* Allocate storage following cs specification */
    int nhalo = 1;
    int nlocal[3] = {0};
    cs_nhalo(cs, &nhalo);
    cs_nlocal(cs, nlocal);

    {
      int nx = nlocal[X] + 2*nhalo;
      int ny = nlocal[Y] + 2*nhalo;
      int nz = nlocal[Z] + 2*nhalo;
      obj->nsite = nx*ny*nz;
    }
    {
      size_t sz = sizeof(double)*obj->nsite*obj->ndist*obj->nvel;
      assert(sz > 0); /* Should not overflow in size_t I hope! */
      obj->f      = (double *) mem_aligned_malloc(MEM_PAGESIZE, sz);
      obj->fprime = (double *) mem_aligned_malloc(MEM_PAGESIZE, sz);
      assert(obj->f);
      assert(obj->fprime);
      if (obj->f      == NULL) pe_fatal(pe, "malloc(lb->f) failed\n");
      if (obj->fprime == NULL) pe_fatal(pe, "malloc(lb->fprime) failed\n");
      if (options->usefirsttouch) {
	lb_data_touch(obj);
	pe_info(pe, "Host data:        first touch\n");
      }
      else {
	memset(obj->f, 0, sz);
	memset(obj->fprime, 0, sz);
      }
    }
  }

  /* Collision parameters. This is fixed-size struct could not allocate...*/ 
  obj->param = (lb_collide_param_t *) calloc(1, sizeof(lb_collide_param_t));
  assert(obj->param);
  if (obj->param == NULL) {
    pe_fatal(pe, "calloc(1, lb_collide_param_t) failed\n");
  }

  lb_halo_create(obj, &obj->h, obj->haloscheme);
  lb_init(obj);

  /* i/o metadata */
  {
    io_element_t ascii = {
      .datatype = MPI_CHAR,
      .datasize = sizeof(char),
      .count    = obj->nvel*(1 + LB_RECORD_LENGTH_ASCII*obj->ndist),
      .endian   = io_endianness()
    };

    io_element_t binary = {
      .datatype = MPI_DOUBLE,
      .datasize = sizeof(double),
      .count    = obj->nvel*obj->ndist,
      .endian   = io_endianness()
    };

    /* Record element information */
    obj->ascii = ascii;
    obj->binary = binary;

    /* Establish metadata */
    int ifail = 0;
    io_element_t element = {0};

    if (options->iodata.input.iorformat == IO_RECORD_ASCII)  element = ascii;
    if (options->iodata.input.iorformat == IO_RECORD_BINARY) element = binary;
    ifail = io_metadata_initialise(cs, &options->iodata.input, &element,
				   &obj->input);
    if (ifail != 0) pe_fatal(pe, "lb_data: bad i/o input decomposition\n");


    if (options->iodata.output.iorformat == IO_RECORD_ASCII)  element = ascii;
    if (options->iodata.output.iorformat == IO_RECORD_BINARY) element = binary;
    ifail = io_metadata_initialise(cs, &options->iodata.output, &element,
				   &obj->output);
    if (ifail != 0) pe_fatal(pe, "lb_data: bad i/o output decomposition\n"); 
  }

  *lb = obj;

  return 0;
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

  io_metadata_finalise(&lb->input);
  io_metadata_finalise(&lb->output);

  if (lb->halo) halo_swap_free(lb->halo);
  if (lb->io_info) io_info_free(lb->io_info);
  if (lb->f) free(lb->f);
  if (lb->fprime) free(lb->fprime);

  lb_halo_free(lb, &lb->h);
  lb_model_free(&lb->model);

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

    size_t nsz = (size_t) lb->model.nvel*lb->nsite*lb->ndist*sizeof(double);

    assert(lb->target);

    tdpMemcpy(&tmpf, &lb->target->f, sizeof(double *), tdpMemcpyDeviceToHost);

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpMemcpy(&lb->target->ndim,  &lb->ndim,  sizeof(int), flag);
      tdpMemcpy(&lb->target->nvel,  &lb->nvel,  sizeof(int), flag);
      tdpMemcpy(&lb->target->ndist, &lb->ndist, sizeof(int), flag);
      tdpMemcpy(&lb->target->nsite, &lb->nsite, sizeof(int), flag);
      tdpMemcpy(tmpf, lb->f, nsz, flag);
      break;
    case tdpMemcpyDeviceToHost:
      tdpMemcpy(lb->f, tmpf, nsz, flag);
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
 
static int lb_init(lb_t * lb) {

  int nlocal[3];
  int nx, ny, nz;
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

  ndata = lb->nsite*lb->ndist*lb->model.nvel;

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

  lb_mpi_init(lb);
  lb_model_param_init(lb);
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
  assert(lb->model.nvel <= 27); /* Currently largest model supported */

  lb->param->nsite = lb->nsite;
  lb->param->ndist = lb->ndist;
  lb->param->nvel  = lb->model.nvel;

  for (p = 0; p < lb->model.nvel; p++) {
    lb->param->wv[p] = lb->model.wv[p];
    for (ia = 0; ia < 3; ia++) {
      lb->param->cv[p][ia] = lb->model.cv[p][ia];
    }
  }

  for (ia = 0; ia < lb->model.nvel; ia++) {
    lb->param->rna[ia] = 1.0/lb->model.na[ia];
  }

  for (ia = 0; ia < lb->model.nvel; ia++) {
    for (ib = 0; ib < lb->model.nvel; ib++) {
      double maba = lb->model.ma[ib][ia];
      lb->param->ma[ia][ib] = lb->model.ma[ia][ib];
      lb->param->mi[ia][ib] = lb->model.wv[ia]*lb->model.na[ib]*maba;
    }
  }

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
	double u0[3] = {0};

	index = cs_index(lb->cs, ic, jc, kc);
	lb_1st_moment_equilib_set(lb, index, rho0, u0);
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

  assert(lb);

  halo_swap_create_r2(lb->pe, lb->cs, 1, lb->nsite, lb->ndist, lb->nvel,
		      &lb->halo);
  halo_swap_handlers_set(lb->halo, halo_swap_pack_rank1, halo_swap_unpack_rank1);

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

  sprintf(string, "%1d x Distribution: d%dq%d", lb->ndist, lb->ndim, lb->nvel);

  io_info_set_name(lb->io_info, string);
  io_info_read_set(lb->io_info, IO_FORMAT_BINARY, lb_f_read);
  io_info_write_set(lb->io_info, IO_FORMAT_BINARY, lb_f_write);
  io_info_set_bytesize(lb->io_info, IO_FORMAT_BINARY,
		       sizeof(double)*lb->ndist*lb->nvel);
  io_info_read_set(lb->io_info, IO_FORMAT_ASCII, lb_f_read_ascii);
  io_info_write_set(lb->io_info, IO_FORMAT_ASCII, lb_f_write_ascii);
  io_info_format_set(lb->io_info, form_in, form_out);

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
 *  lb_data_touch
 *
 *  Kernel driver to initialise data.
 *  A conscious choice to use limits which are the standard domain.
 *
 *****************************************************************************/

__host__ void lb_data_touch_kernel(cs_limits_t lim, lb_t * lb) {

  int nx = 1 + lim.imax - lim.imin;
  int ny = 1 + lim.jmax - lim.jmin;
  int nz = 1 + lim.kmax - lim.kmin;

  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  #pragma omp for nowait
  for (int ik = 0; ik < nx*ny*nz; ik++) {
    int ic = lim.imin + (ik       )/strx;
    int jc = lim.jmin + (ik % strx)/stry;
    int kc = lim.kmin + (ik % stry)/strz;
    int index = cs_index(lb->cs, ic, jc, kc);
    for (int p = 0; p < lb->nvel; p++) {
      for (int n = 0; n < lb->ndist; n++) {
	int lindex = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p);
	lb->f[lindex] = 0.0;
	lb->fprime[lindex] = 0.0;
      }
    }
  }

  return;
}

__host__ int lb_data_touch(lb_t * lb) {

  int nlocal[3] = {0};

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  {
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};

    #pragma omp parallel
    {
      lb_data_touch_kernel(lim, lb);
    }
  }

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

  lb_halo_swap(lb, lb->haloscheme);

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

  assert(lb);

  switch (flag) {
  case LB_HALO_TARGET:
    tdpMemcpy(&data, &lb->target->f, sizeof(double *), tdpMemcpyDeviceToHost);
    halo_swap_packed(lb->halo, data);
    break;
  case LB_HALO_OPENMP_FULL:
    lb_halo_post(lb, &lb->h);
    lb_halo_wait(lb, &lb->h);
    break;
  case LB_HALO_OPENMP_REDUCED:
    lb_halo_post(lb, &lb->h);
    lb_halo_wait(lb, &lb->h);
    break;
  default:
    lb_halo_post(lb, &lb->h);
    lb_halo_wait(lb, &lb->h);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_read
 *
 *  Read one lattice site (index) worth of distributions.
 *  Note that read-write is always the same order irrespective
 *  of memory ordering (host/device).
 *
 *****************************************************************************/

static int lb_f_read(FILE * fp, int index, void * self) {

  int iread, n, p;
  int nr = 0;
  lb_t * lb = (lb_t*) self;

  assert(fp);
  assert(lb);

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < lb->model.nvel; p++) {
      iread = LB_ADDR(lb->nsite, lb->ndist, lb->model.nvel, index, n, p);
      nr += fread(lb->f + iread, sizeof(double), 1, fp);
    }
  }

  if (nr != lb->ndist*lb->model.nvel) {
    pe_fatal(lb->pe, "fread(lb) failed at %d\n", index);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_read_ascii
 *
 *  Read one lattice site (index) worth of distributions.
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
    for (p = 0; p < lb->model.nvel; p++) {
      int ijkp = LB_ADDR(lb->nsite, lb->ndist, lb->model.nvel, index, n, p);
      nr += fscanf(fp, "%le", &lb->f[ijkp]);
    }
  }

  if (nr != lb->ndist*lb->model.nvel) {
    pe_fatal(pe, "fread(lb) failed at %d\n", index);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_write
 *
 *  Write one lattice site (index) worth of distributions.
 *
 *****************************************************************************/

static int lb_f_write(FILE * fp, int index, void * self) {

  int iwrite, n, p;
  int nw = 0;
  lb_t * lb = (lb_t*) self;

  assert(fp);
  assert(self);

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < lb->model.nvel; p++) {
      iwrite = LB_ADDR(lb->nsite, lb->ndist, lb->model.nvel, index, n, p);
      nw += fwrite(lb->f + iwrite, sizeof(double), 1, fp);
    }
  }

  if (nw != lb->ndist*lb->model.nvel) {
    pe_fatal(lb->pe, "fwrite(lb) failed at %d\n", index);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_write_ascii
 *
 *  Write one lattice site (index) worth of distributions.
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
  {
    int ic = noffset[X] + coords[X];
    int jc = noffset[Y] + coords[Y];
    int kc = noffset[Z] + coords[Z];
    fprintf(fp, "%d %d %d ", ic, jc, kc);
  }

  for (n = 0; n < lb->ndist; n++) {
    for (p = 0; p < lb->model.nvel; p++) {
      int ijkp = LB_ADDR(lb->nsite, lb->ndist, lb->model.nvel, index, n, p);
      fprintf(fp, "%le ", lb->f[ijkp]);
      nw++;
    }
  }
  fprintf(fp, "\n");

  if (nw != lb->ndist*lb->model.nvel) {
    pe_fatal(pe, "fwrite(lb) failed at %d\n", index);
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
 *  lb_f
 *
 *  Get the distribution at site index, velocity p, distribution n.
 *
 *****************************************************************************/

__host__ __device__
int lb_f(lb_t * lb, int index, int p, int n, double * f) {

  assert(lb);
  assert(index >= 0 && index < lb->nsite);
  assert(p >= 0 && p < lb->nvel);
  assert(n >= 0 && n < lb->ndist);

  *f = lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p)];

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
  assert(p >= 0 && p < lb->nvel);
  assert(n >= 0 && n < lb->ndist);

  lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p)] = fvalue;

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

  assert(lb);
  assert(rho);
  assert(index >= 0 && index < lb->nsite);
  assert(nd < lb->ndist);

  *rho = 0.0;

  for (int p = 0; p < lb->nvel; p++) {
    *rho += lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, nd, p)];
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

  for (p = 0; p < lb->model.nvel; p++) {
    for (n = 0; n < lb->model.ndim; n++) {
      g[n] += lb->model.cv[p][n]
	*lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, nd, p)];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_2nd_moment
 *
 *  Return the (deviatoric) stress at index. [Test coverage?]
 *
 *****************************************************************************/

__host__
int lb_2nd_moment(lb_t * lb, int index, lb_dist_enum_t nd, double s[3][3]) {

  int p, ia, ib;

  assert(lb);
  assert(nd == LB_RHO);
  assert(index >= 0  && index < lb->nsite);

  for (ia = 0; ia < lb->model.ndim; ia++) {
    for (ib = 0; ib < lb->model.ndim; ib++) {
      s[ia][ib] = 0.0;
    }
  }

  for (p = 0; p < lb->model.nvel; p++) {
    for (ia = 0; ia < lb->model.ndim; ia++) {
      for (ib = 0; ib < lb->model.ndim; ib++) {
	double f = 0.0;
	double cs2 = lb->model.cs2;
	double dab = (ia == ib);
	f = lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, nd, p)];
	s[ia][ib] += f*(lb->model.cv[p][ia]*lb->model.cv[p][ib] - cs2*dab);
      }
    }
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

  assert(lb);
  assert(index >= 0 && index < lb->nsite);

  for (p = 0; p < lb->model.nvel; p++) {
    double cs2 = lb->model.cs2;
    double rcs2 = 1.0/cs2;
    double udotc = 0.0;
    double sdotq = 0.0;
    for (ia = 0; ia < 3; ia++) {
      udotc += u[ia]*lb->model.cv[p][ia];
      for (ib = 0; ib < 3; ib++) {
	double dab = (ia == ib);
	sdotq += (lb->model.cv[p][ia]*lb->model.cv[p][ib] - cs2*dab)*u[ia]*u[ib];
      }
    }

    lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, LB_RHO, p)]
      = rho*lb->model.wv[p]*(1.0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq);
  }

  return 0;
}

/* We will not exceed 27 directions! Direction index 0, in keeping
 * with the LB model definition, is (0,0,0) - so no communication. */

/*****************************************************************************
 *
 *  lb_halo_size
 *
 *  Utility to compute a number of sites from cs_limits_t.
 *
 *****************************************************************************/

int lb_halo_size(cs_limits_t lim) {

  int szx = 1 + lim.imax - lim.imin;
  int szy = 1 + lim.jmax - lim.jmin;
  int szz = 1 + lim.kmax - lim.kmin;

  return szx*szy*szz;
}

/*****************************************************************************
 *
 *  lb_halo_enqueue_send
 *
 *  Pack the send buffer. The ireq determines the direction of the
 *  communication.
 *
 *****************************************************************************/

int lb_halo_enqueue_send(const lb_t * lb, lb_halo_t * h, int ireq) {

  assert(0 <= ireq && ireq < h->map.nvel);

  if (h->count[ireq] > 0) {

    int8_t mx = h->map.cv[ireq][X];
    int8_t my = h->map.cv[ireq][Y];
    int8_t mz = h->map.cv[ireq][Z];
    int8_t mm = mx*mx + my*my + mz*mz;

    int nx = 1 + h->slim[ireq].imax - h->slim[ireq].imin;
    int ny = 1 + h->slim[ireq].jmax - h->slim[ireq].jmin;
    int nz = 1 + h->slim[ireq].kmax - h->slim[ireq].kmin;

    int strz = 1;
    int stry = strz*nz;
    int strx = stry*ny;

    assert(mm == 1 || mm == 2 || mm == 3);

    #pragma omp for nowait
    for (int ih = 0; ih < nx*ny*nz; ih++) {
      int ic = h->slim[ireq].imin + ih/strx;
      int jc = h->slim[ireq].jmin + (ih % strx)/stry;
      int kc = h->slim[ireq].kmin + (ih % stry)/strz;
      int ib = 0; /* Buffer index */

      for (int n = 0; n < lb->ndist; n++) {
	for (int p = 0; p < lb->nvel; p++) {
	  /* Recall, if full, we need p = 0 */
	  int8_t px = lb->model.cv[p][X];
	  int8_t py = lb->model.cv[p][Y];
	  int8_t pz = lb->model.cv[p][Z];
	  int dot = mx*px + my*py + mz*pz;
	  if (h->full || dot == mm) {
	    int index = cs_index(lb->cs, ic, jc, kc);
	    int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p);
	    h->send[ireq][ih*h->count[ireq] + ib] = lb->f[laddr];
	    ib++;
	  }
	}
      }
      assert(ib == h->count[ireq]);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_dequeue_recv
 *
 *  Unpack the recv buffer into place in the distributions.
 *
 *****************************************************************************/

int lb_halo_dequeue_recv(lb_t * lb, const lb_halo_t * h, int ireq) {

  assert(lb);
  assert(h);
  assert(0 <= ireq && ireq < h->map.nvel);

  if (h->count[ireq] > 0) {

    /* The communication direction is reversed cf. the send... */
    int8_t mx = h->map.cv[h->map.nvel-ireq][X];
    int8_t my = h->map.cv[h->map.nvel-ireq][Y];
    int8_t mz = h->map.cv[h->map.nvel-ireq][Z];
    int8_t mm = mx*mx + my*my + mz*mz;

    int nx = 1 + h->rlim[ireq].imax - h->rlim[ireq].imin;
    int ny = 1 + h->rlim[ireq].jmax - h->rlim[ireq].jmin;
    int nz = 1 + h->rlim[ireq].kmax - h->rlim[ireq].kmin;

    int strz = 1;
    int stry = strz*nz;
    int strx = stry*ny;

    double * recv = h->recv[ireq];

    {
      int i = 1 + mx;
      int j = 1 + my;
      int k = 1 + mz;
      /* If Cartesian neighbour is self, just copy out of send buffer. */
      if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) recv = h->send[ireq]; 
    }

    assert(mm == 1 || mm == 2 || mm == 3);

    #pragma omp for nowait
    for (int ih = 0; ih < nx*ny*nz; ih++) {
      int ic = h->rlim[ireq].imin + ih/strx;
      int jc = h->rlim[ireq].jmin + (ih % strx)/stry;
      int kc = h->rlim[ireq].kmin + (ih % stry)/strz;
      int ib = 0; /* Buffer index */

      for (int n = 0; n < lb->ndist; n++) {
	for (int p = 0; p < lb->nvel; p++) {
	  /* For reduced swap, we must have -cv[p] here... */
	  int8_t px = lb->model.cv[lb->nvel-p][X];
	  int8_t py = lb->model.cv[lb->nvel-p][Y];
	  int8_t pz = lb->model.cv[lb->nvel-p][Z];
	  int dot = mx*px + my*py + mz*pz;

	  if (h->full || dot == mm) {
	    int index = cs_index(lb->cs, ic, jc, kc);
	    int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p);
	    lb->f[laddr] = recv[ih*h->count[ireq] + ib];
	    ib++;
	  }
	}
      }
      assert(ib == h->count[ireq]);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_create
 *
 *  Currently: generate all send and receive requests.
 *
 *****************************************************************************/

int lb_halo_create(const lb_t * lb, lb_halo_t * h, lb_halo_enum_t scheme) {

  assert(lb);
  assert(h);

  *h = (lb_halo_t) {0};

  /* Communication model */
  if (lb->model.ndim == 2) lb_model_create( 9, &h->map);
  if (lb->model.ndim == 3) lb_model_create(27, &h->map);

  assert(lb->model.ndim == 2 || lb->model.ndim == 3);
  assert(h->map.ndim == lb->model.ndim);

  cs_nlocal(lb->cs, h->nlocal);
  cs_cart_comm(lb->cs, &h->comm);
  h->tagbase = 211216;

  /* Default to full swap unless reduced is requested. */

  h->full = 1;
  if (scheme == LB_HALO_OPENMP_REDUCED) h->full = 0;

  /* Determine look-up table of ranks of neighbouring processes */
  {
    int dims[3] = {0};
    int periods[3] = {0};
    int coords[3] = {0};

    MPI_Cart_get(h->comm, 3, dims, periods, coords);

    for (int p = 0; p < h->map.nvel; p++) {
      int nbr[3] = {0};
      int out[3] = {0};  /* Out-of-range is erroneous for non-perioidic dims */
      int i = 1 + h->map.cv[p][X];
      int j = 1 + h->map.cv[p][Y];
      int k = 1 + h->map.cv[p][Z];

      nbr[X] = coords[X] + h->map.cv[p][X];
      nbr[Y] = coords[Y] + h->map.cv[p][Y];
      nbr[Z] = coords[Z] + h->map.cv[p][Z];
      out[X] = (!periods[X] && (nbr[X] < 0 || nbr[X] > dims[X] - 1));
      out[Y] = (!periods[Y] && (nbr[Y] < 0 || nbr[Y] > dims[Y] - 1));
      out[Z] = (!periods[Z] && (nbr[Z] < 0 || nbr[Z] > dims[Z] - 1));

      if (out[X] || out[Y] || out[Z]) {
	h->nbrrank[i][j][k] = MPI_PROC_NULL;
      }
      else {
	MPI_Cart_rank(h->comm, nbr, &h->nbrrank[i][j][k]);
      }
    }
    /* I must be in the middle */
    assert(h->nbrrank[1][1][1] == cs_cart_rank(lb->cs));
  }


  /* Limits of the halo regions in each communication direction */

  for (int p = 1; p < h->map.nvel; p++) {

    /* Limits for send and recv regions*/
    int8_t cx = h->map.cv[p][X];
    int8_t cy = h->map.cv[p][Y];
    int8_t cz = h->map.cv[p][Z];

    cs_limits_t send = {1, h->nlocal[X], 1, h->nlocal[Y], 1, h->nlocal[Z]};
    cs_limits_t recv = {1, h->nlocal[X], 1, h->nlocal[Y], 1, h->nlocal[Z]};

    if (cx == -1) send.imax = 1;
    if (cx == +1) send.imin = send.imax;
    if (cy == -1) send.jmax = 1;
    if (cy == +1) send.jmin = send.jmax;
    if (cz == -1) send.kmax = 1;
    if (cz == +1) send.kmin = send.kmax;

    /* velocity is reversed... */
    if (cx == +1) recv.imax = recv.imin = 0;
    if (cx == -1) recv.imin = recv.imax = recv.imax + 1;
    if (cy == +1) recv.jmax = recv.jmin = 0;
    if (cy == -1) recv.jmin = recv.jmax = recv.jmax + 1;
    if (cz == +1) recv.kmax = recv.kmin = 0;
    if (cz == -1) recv.kmin = recv.kmax = recv.kmax + 1;

    h->slim[p] = send;
    h->rlim[p] = recv;
  }

  /* Message count (velocities) for each communication direction */

  for (int p = 1; p < h->map.nvel; p++) {

    int count = 0;

    if (h->full) {
      count = lb->model.nvel;
    }
    else {
      int8_t mx = h->map.cv[p][X];
      int8_t my = h->map.cv[p][Y];
      int8_t mz = h->map.cv[p][Z];
      int8_t mm = mx*mx + my*my + mz*mz;

      /* Consider each model velocity in turn */
      for (int q = 1; q < lb->model.nvel; q++) {
	int8_t qx = lb->model.cv[q][X];
	int8_t qy = lb->model.cv[q][Y];
	int8_t qz = lb->model.cv[q][Z];
	int8_t dot = mx*qx + my*qy + mz*qz;

	if (mm == 3 && dot == mm) count +=1;   /* This is a corner */
	if (mm == 2 && dot == mm) count +=1;   /* This is an edge */
	if (mm == 1 && dot == mm) count +=1;   /* This is a side */
      }
    }

    count = lb->ndist*count;
    h->count[p] = count;
    /* Allocate send buffer for send region */
    if (count > 0) {
      int scount = count*lb_halo_size(h->slim[p]);
      h->send[p] = (double *) calloc(scount, sizeof(double));
      assert(h->send[p]);
    }
    /* Allocate recv buffer */
    if (count > 0) {
      int rcount = count*lb_halo_size(h->rlim[p]);
      h->recv[p] = (double *) calloc(rcount, sizeof(double));
      assert(h->recv[p]);
    }
  }

  /* Ensure all requests are NULL in case a particular one is not required */

  for (int ireq = 0; ireq < 2*27; ireq++) {
    h->request[ireq] = MPI_REQUEST_NULL;
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_post
 *
 *****************************************************************************/

int lb_halo_post(const lb_t * lb, lb_halo_t * h) {

  assert(lb);
  assert(h);

  /* Imbalance timer */
  if (lb->opts.reportimbalance) {
    TIMER_start(TIMER_LB_HALO_IMBAL);
    MPI_Barrier(h->comm);
    TIMER_stop(TIMER_LB_HALO_IMBAL);
  }

  /* Post recvs (from opposite direction cf send) */

  TIMER_start(TIMER_LB_HALO_IRECV);

  for (int ireq = 0; ireq < h->map.nvel; ireq++) {

    h->request[ireq] = MPI_REQUEST_NULL;

    if (h->count[ireq] > 0) {
      int i = 1 + h->map.cv[h->map.nvel-ireq][X];
      int j = 1 + h->map.cv[h->map.nvel-ireq][Y];
      int k = 1 + h->map.cv[h->map.nvel-ireq][Z];
      int mcount = h->count[ireq]*lb_halo_size(h->rlim[ireq]);

      if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) mcount = 0;
      
      MPI_Irecv(h->recv[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
		h->tagbase + ireq, h->comm, h->request + ireq);
    }
  }

  TIMER_stop(TIMER_LB_HALO_IRECV);

  /* Load send buffers */
  /* Enqueue sends (second half of request array) */

  TIMER_start(TIMER_LB_HALO_PACK);

  #pragma omp parallel
  {
    for (int ireq = 0; ireq < h->map.nvel; ireq++) {
      lb_halo_enqueue_send(lb, h, ireq);
    }
  }

  TIMER_stop(TIMER_LB_HALO_PACK);

  TIMER_start(TIMER_LB_HALO_ISEND);

  for (int ireq = 0; ireq < h->map.nvel; ireq++) {

    h->request[27+ireq] = MPI_REQUEST_NULL;

    if (h->count[ireq] > 0) {
      int i = 1 + h->map.cv[ireq][X];
      int j = 1 + h->map.cv[ireq][Y];
      int k = 1 + h->map.cv[ireq][Z];
      int mcount = h->count[ireq]*lb_halo_size(h->slim[ireq]);

      /* Short circuit messages to self. */
      if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) mcount = 0;

      MPI_Isend(h->send[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
		h->tagbase + ireq, h->comm, h->request + 27 + ireq);
    }
  }

  TIMER_stop(TIMER_LB_HALO_ISEND);

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_wait
 *
 *****************************************************************************/

int lb_halo_wait(lb_t * lb, lb_halo_t * h) {

  assert(lb);
  assert(h);

  TIMER_start(TIMER_LB_HALO_WAIT);

  MPI_Waitall(2*h->map.nvel, h->request, MPI_STATUSES_IGNORE);

  TIMER_stop(TIMER_LB_HALO_WAIT);

  TIMER_start(TIMER_LB_HALO_UNPACK);

  #pragma omp parallel
  {
    for (int ireq = 0; ireq < h->map.nvel; ireq++) {
      lb_halo_dequeue_recv(lb, h, ireq);
    }
  }

  TIMER_stop(TIMER_LB_HALO_UNPACK);

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_free
 *
 *  Complete all the send and receive requests.
 *
 *****************************************************************************/

int lb_halo_free(lb_t * lb, lb_halo_t * h) {

  assert(lb);
  assert(h);

  for (int ireq = 0; ireq < 27; ireq++) {
    free(h->send[ireq]);
    free(h->recv[ireq]);
  }

  lb_model_free(&h->map);

  return 0;
}

/*****************************************************************************
 *
 *  lb_write_buf
 *
 *  Write output buffer independent of in-memory order.
 *
 *****************************************************************************/

int lb_write_buf(const lb_t * lb, int index, char * buf) {

  double data[NVELMAX] = {0};

  assert(lb);
  assert(buf);

  for (int n = 0; n < lb->ndist; n++) {
    size_t sz = lb->model.nvel*sizeof(double);
    for (int p = 0; p < lb->model.nvel; p++) {
      int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->model.nvel, index, n, p);
      data[p] = lb->f[laddr];
    }
    memcpy(buf + n*sz, data, sz);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_read_buf
 *
 *****************************************************************************/

int lb_read_buf(lb_t * lb, int index, const char * buf) {

  double data[NVELMAX] = {0};

  assert(lb);
  assert(buf);

  for (int n = 0; n < lb->ndist; n++) {
    size_t sz = lb->model.nvel*sizeof(double);
    memcpy(data, buf + n*sz, sz);
    for (int p = 0; p < lb->model.nvel; p++) {
      int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->model.nvel, index, n, p);
      lb->f[laddr] = data[p];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_write_buf_ascii
 *
 *  For ascii, we are going to put ndist distributions on a single line...
 *  This is merely cosmetic, and for appearances.
 *
 *****************************************************************************/

int lb_write_buf_ascii(const lb_t * lb, int index, char * buf) {

  const int nbyte = LB_RECORD_LENGTH_ASCII;  /* bytes per " %22.15s" datum */
  int ifail = 0;

  assert(lb);
  assert(buf);
  assert((lb->ndist*nbyte + 1)*sizeof(char) < BUFSIZ);

  for (int p = 0; p < lb->model.nvel; p++) {
    char tmp[BUFSIZ] = {0};
    int poffset = p*(lb->ndist*nbyte + 1); /* +1 for each newline */
    for (int n = 0; n < lb->ndist; n++) {
      int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->model.nvel, index, n, p);
      int np = snprintf(tmp, nbyte + 1, " %22.15e", lb->f[laddr]);
      if (np != nbyte) ifail = 1;
      memcpy(buf + poffset + n*nbyte, tmp, nbyte*sizeof(char));
    }
    /* Add newline */
    if (1 != snprintf(tmp, 2, "\n")) ifail = 2;
    memcpy(buf + poffset + lb->ndist*nbyte, tmp, sizeof(char));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  lb_read_buf_ascii
 *
 *****************************************************************************/

int lb_read_buf_ascii(lb_t * lb, int index, const char * buf) {

  const int nbyte = LB_RECORD_LENGTH_ASCII;  /* bytes per " %22.15s" datum */
  int ifail = 0;

  assert(lb);
  assert(buf);

  for (int p = 0; p < lb->model.nvel; p++) {
    int poffset = p*(lb->ndist*nbyte + 1); /* +1 for each newline */
    for (int n = 0; n < lb->ndist; n++) {
      int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->model.nvel, index, n, p);
      char tmp[BUFSIZ] = {0};              /* Make sure we have a \0 */
      memcpy(tmp, buf + poffset + n*nbyte, nbyte*sizeof(char));
      int nr = sscanf(tmp, "%le", lb->f + laddr);
      if (nr != 1) ifail = 1;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  lb_io_aggr_pack
 *
 *****************************************************************************/

__host__ int lb_io_aggr_pack(const lb_t * lb, io_aggregator_t * aggr) {

  assert(lb);
  assert(aggr);
  assert(aggr->buf);

  #pragma omp parallel
  {
    int iasc = lb->opts.iodata.output.iorformat == IO_RECORD_ASCII;
    int ibin = lb->opts.iodata.output.iorformat == IO_RECORD_BINARY;
    assert(iasc ^ ibin); /* One or other */

    #pragma omp for
    for (int ib = 0; ib < cs_limits_size(aggr->lim); ib++) {
      int ic = cs_limits_ic(aggr->lim, ib);
      int jc = cs_limits_jc(aggr->lim, ib);
      int kc = cs_limits_kc(aggr->lim, ib);

      /* Write data (ic,jc,kc) */
      int index = cs_index(lb->cs, ic, jc, kc);
      int offset = ib*aggr->szelement;
      if (iasc) lb_write_buf_ascii(lb, index, aggr->buf + offset);
      if (ibin) lb_write_buf(lb, index, aggr->buf + offset);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_io_aggr_buf_unpack
 *
 *****************************************************************************/

__host__ int lb_io_aggr_unpack(lb_t * lb, const io_aggregator_t * aggr) {

  assert(lb);
  assert(aggr);
  assert(aggr->buf);

  #pragma omp parallel
  {
    int iasc = lb->opts.iodata.input.iorformat == IO_RECORD_ASCII;
    int ibin = lb->opts.iodata.input.iorformat == IO_RECORD_BINARY;
    assert(iasc ^ ibin);

    #pragma omp for
    for (int ib = 0; ib < cs_limits_size(aggr->lim); ib++) {
      int ic = cs_limits_ic(aggr->lim, ib);
      int jc = cs_limits_jc(aggr->lim, ib);
      int kc = cs_limits_kc(aggr->lim, ib);

      /* Read data at (ic,jc,kc) */
      int index = cs_index(lb->cs, ic, jc, kc);
      int offset = ib*aggr->szelement;
      if (iasc) lb_read_buf_ascii(lb, index, aggr->buf + offset);
      if (ibin) lb_read_buf(lb, index, aggr->buf + offset);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_io_write
 *
 *****************************************************************************/

int lb_io_write(lb_t * lb, int timestep, io_event_t * event) {

  assert(lb);
  assert(event);

  const io_metadata_t * meta = &lb->output;

  if (meta->iswriten == 0) {
    /* No comments at the moment */
    cJSON * comments = NULL;
    int ifail = io_metadata_write(meta, "dist", comments);
    if (ifail == 0) lb->output.iswriten = 1;
  }

  if (meta->options.mode != IO_MODE_MPIIO) {
    /* Old-style */
    char filename[BUFSIZ] = {0};
    sprintf(filename, "dist-%8.8d", timestep);
    io_write_data(lb->io_info, filename, lb);
  }

  if (meta->options.mode == IO_MODE_MPIIO) {

    io_impl_t * io = NULL;
    char filename[BUFSIZ] = {0};

    io_subfile_name(&meta->subfile, "dist", timestep, filename, BUFSIZ);
    io_impl_create(meta, &io);
    assert(io);

    io_event_record(event, IO_EVENT_AGGR);
    lb_memcpy(lb, tdpMemcpyDeviceToHost);
    lb_io_aggr_pack(lb, io->aggr);

    io_event_record(event, IO_EVENT_WRITE);
    io->impl->write(io, filename);

    if (meta->options.report) {
      pe_info(lb->pe, "MPIIO wrote to %s\n", filename);
    }

    io->impl->free(&io);
    io_event_report(event, meta, "dist");
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_io_read
 *
 *****************************************************************************/

int lb_io_read(lb_t * lb, int timestep, io_event_t * event) {

  assert(lb);
  assert(event);

  const io_metadata_t * meta = &lb->input;

  if (meta->options.mode != IO_MODE_MPIIO) {
    char filename[BUFSIZ] = {0};
    sprintf(filename, "dist-%8.8d", timestep);
    io_read_data(lb->io_info, filename, lb);
  }

  if (meta->options.mode == IO_MODE_MPIIO) {
    io_impl_t * io = NULL;
    char filename[BUFSIZ] = {0};

    io_subfile_name(&meta->subfile, "dist", timestep, filename, BUFSIZ);
    io_impl_create(meta, &io);
    assert(io);

    io->impl->read(io, filename);
    lb_io_aggr_unpack(lb, io->aggr);
    io->impl->free(&io);
  }

  return 0;
}
