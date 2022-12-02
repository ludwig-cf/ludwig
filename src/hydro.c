/*****************************************************************************
 *
 *  hydro.c
 *
 *  Hydrodynamic quantities: velocity, body force on fluid.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h> 

#include "kernel.h"
#include "coords_field.h"
#include "util.h"
#include "hydro.h"

#include "timer.h"

static int hydro_lees_edwards_parallel(hydro_t * obj);
static int hydro_u_write(FILE * fp, int index, void * self);
static int hydro_u_write_ascii(FILE * fp, int index, void * self);
static int hydro_u_read(FILE * fp, int index, void * self);
static int hydro_u_read_ascii(FILE * fp, int index, void * self);

static __global__
void hydro_field_set(hydro_t * hydro, double * field, double, double, double);

__global__ void hydro_accumulate_kernel(kernel_ctxt_t * ktx, hydro_t * hydro,
                                        double fnet[3]);
__global__ void hydro_correct_kernel(kernel_ctxt_t * ktx, hydro_t * hydro,
                                     double fnet[3]);
__global__ void hydro_accumulate_kernel_v(kernel_ctxt_t * ktx, hydro_t * hydro,
                                          double fnet[3]);
__global__ void hydro_correct_kernel_v(kernel_ctxt_t * ktx, hydro_t * hydro,
				       double fnet[3]);
__global__ void hydro_rho0_kernel(int nsite, double rho0, double * rho);


/*****************************************************************************
 *
 *  hydro_create
 *
 *  We typically require a halo region for the velocity which is only
 *  one lattice site in width, i.e., nhcomm = 1. This is independent
 *  of the width of the halo region specified for coords object.
 *
 *****************************************************************************/

__host__ int hydro_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			  const hydro_options_t * opts,
			  hydro_t ** pobj) {
  int ndevice;
  double * tmp;
  hydro_t * obj = (hydro_t *) NULL;

  assert(pe);
  assert(cs);
  assert(opts);
  assert(pobj);

  obj = (hydro_t *) calloc(1, sizeof(hydro_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(hydro) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->le = le;
  obj->nhcomm = opts->nhcomm;

  cs_nsites(cs, &obj->nsite);
  if (le) lees_edw_nsites(le, &obj->nsite);

  /* Fields */
  {
    /* rho: scalar field with no halo swap (halo width zero). */
    field_options_t options = field_options_ndata_nhalo(1, 0);
    /* haloscheme */
    /* halo verbose */
    /* usefirsttouch */
    /* iodata */
    field_create(pe, cs, le, "rho", &options, &obj->rho);

    /* eta: scalar viscosity with no halo swap */
    field_create(pe, cs, le, "eta", &options, &obj->eta);
  }

  /* Original */

  obj->u = (double *) mem_aligned_calloc(MEM_PAGESIZE, NHDIM*obj->nsite,
					 sizeof(double));
  if (obj->u == NULL) pe_fatal(pe, "calloc(hydro->u) failed\n");

  obj->f = (double *) mem_aligned_calloc(MEM_PAGESIZE, NHDIM*obj->nsite,
					 sizeof(double));
  if (obj->f == NULL) pe_fatal(pe, "calloc(hydro->f) failed\n");

  halo_swap_create_r1(pe, cs, opts->nhcomm, obj->nsite, NHDIM, &obj->halo);
  assert(obj->halo);

  halo_swap_handlers_set(obj->halo, halo_swap_pack_rank1, halo_swap_unpack_rank1);

  /* Allocate target copy of structure (or alias) */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {

    tdpAssert(tdpMalloc((void **) &obj->target, sizeof(hydro_t)));
    tdpAssert(tdpMemset(obj->target, 0, sizeof(hydro_t)));

    tdpAssert(tdpMalloc((void **) &tmp, NHDIM*obj->nsite*sizeof(double)));
    tdpAssert(tdpMemset(tmp, 0, NHDIM*obj->nsite*sizeof(double)));
    tdpAssert(tdpMemcpy(&obj->target->u, &tmp, sizeof(double *),
			tdpMemcpyHostToDevice)); 

    tdpAssert(tdpMalloc((void **) &tmp, NHDIM*obj->nsite*sizeof(double)));
    tdpAssert(tdpMemset(tmp, 0, NHDIM*obj->nsite*sizeof(double)));
    tdpAssert(tdpMemcpy(&obj->target->f, &tmp, sizeof(double *),
			tdpMemcpyHostToDevice)); 

    tdpAssert(tdpMemcpy(&obj->target->nsite, &obj->nsite, sizeof(int),
			tdpMemcpyHostToDevice));

    /* Fields: target pointers should be copied ... */
    tdpAssert(tdpMemcpy(&obj->target->rho, &obj->rho->target,
			sizeof(field_t *),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&obj->target->eta, &obj->eta->target,
			sizeof(field_t *),
			tdpMemcpyHostToDevice));
  }

  hydro_halo_create(obj, &obj->h);
  obj->opts = *opts;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_free
 *
 *****************************************************************************/

__host__ int hydro_free(hydro_t * obj) {

  int ndevice;
  double * tmp;

  assert(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpAssert(tdpMemcpy(&tmp, &obj->target->u, sizeof(double *),
			tdpMemcpyDeviceToHost)); 
    tdpAssert(tdpFree(tmp));
    tdpAssert(tdpMemcpy(&tmp, &obj->target->f, sizeof(double *),
			tdpMemcpyDeviceToHost)); 
    tdpAssert(tdpFree(tmp));
    tdpAssert(tdpFree(obj->target));
  }

  halo_swap_free(obj->halo);
  hydro_halo_free(&obj->h);
  if (obj->info) io_info_free(obj->info);

  field_free(obj->eta);
  field_free(obj->rho);

  free(obj->f);
  free(obj->u);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_memcpy
 *
 *****************************************************************************/

__host__ int hydro_memcpy(hydro_t * obj, tdpMemcpyKind flag) {

  int ndevice;
  double * tmpu;
  double * tmpf;

  assert(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {
    tdpAssert(tdpMemcpy(&tmpf, &obj->target->f, sizeof(double *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(&tmpu, &obj->target->u, sizeof(double *),
			tdpMemcpyDeviceToHost));

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpAssert(tdpMemcpy(tmpu, obj->u, NHDIM*obj->nsite*sizeof(double), flag));
      tdpAssert(tdpMemcpy(tmpf, obj->f, NHDIM*obj->nsite*sizeof(double), flag));
      tdpAssert(tdpMemcpy(&obj->target->nsite, &obj->nsite, sizeof(int), flag));
      break;
    case tdpMemcpyDeviceToHost:
      tdpAssert(tdpMemcpy(obj->f, tmpf, NHDIM*obj->nsite*sizeof(double), flag));
      tdpAssert(tdpMemcpy(obj->u, tmpu, NHDIM*obj->nsite*sizeof(double), flag));
      break;
    default:
      pe_fatal(obj->pe, "Bad flag in hydro_memcpy\n");
    }
    /* Fields */
    field_memcpy(obj->rho, flag);
    field_memcpy(obj->eta, flag);
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_halo
 *
 *****************************************************************************/

__host__ int hydro_u_halo(hydro_t * obj) {

  assert(obj);

  hydro_halo_swap(obj, obj->opts.haloscheme);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_halo_swap
 *
 *  There is no halo swap in the density at the moment, as it is never
 *  required.
 *
 *****************************************************************************/

__host__ int hydro_halo_swap(hydro_t * obj, hydro_halo_enum_t flag) {

  double * data;

  assert(obj);

  switch (flag) {
  case HYDRO_U_HALO_HOST:
    halo_swap_host_rank1(obj->halo, obj->u, MPI_DOUBLE);
    break;
  case HYDRO_U_HALO_TARGET:
    tdpAssert(tdpMemcpy(&data, &obj->target->u, sizeof(double *),
			tdpMemcpyDeviceToHost));
    halo_swap_packed(obj->halo, data);
    break;
  case HYDRO_U_HALO_OPENMP:
    hydro_halo_post(obj);
    hydro_halo_wait(obj);
    break;
  default:
    assert(0);
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_init_io_info
 *
 *  There is no read for the velocity; this should come from the
 *  distribution.
 *
 *****************************************************************************/

__host__ int hydro_init_io_info(hydro_t * obj, int grid[3], int form_in,
				int form_out) {

  io_info_args_t args = io_info_args_default();

  assert(obj);
  assert(grid);
  assert(obj->info == NULL);

  args.grid[X] = grid[X];
  args.grid[Y] = grid[Y];
  args.grid[Z] = grid[Z];

  io_info_create(obj->pe, obj->cs, &args, &obj->info);
  if (obj->info == NULL) pe_fatal(obj->pe, "io_info_create(hydro) failed\n");

  io_info_set_name(obj->info, "Velocity field");
  io_info_write_set(obj->info, IO_FORMAT_BINARY, hydro_u_write);
  io_info_write_set(obj->info, IO_FORMAT_ASCII, hydro_u_write_ascii);
  io_info_read_set(obj->info, IO_FORMAT_BINARY, hydro_u_read);
  io_info_read_set(obj->info, IO_FORMAT_ASCII, hydro_u_read_ascii);

  /* ASCII output size (see write_ascii) is 69 bytes */
  io_info_set_bytesize(obj->info, IO_FORMAT_BINARY, NHDIM*sizeof(double));
  io_info_set_bytesize(obj->info, IO_FORMAT_ASCII, 69);

  io_info_format_set(obj->info, form_in, form_out);
  io_info_metadata_filestub_set(obj->info, "vel");

  return 0;
}

/*****************************************************************************
 *
 *  hydro_io_info
 *
 *****************************************************************************/

__host__ int hydro_io_info(hydro_t * obj, io_info_t ** info) {

  assert(obj);
  assert(obj->info); /* Should have been initialised */

  *info = obj->info;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local_set
 *
 *****************************************************************************/

__host__ __device__
int hydro_f_local_set(hydro_t * obj, int index, const double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->f[addr_rank1(obj->nsite, NHDIM, index, ia)] = force[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local
 *
 *****************************************************************************/

__host__ __device__
int hydro_f_local(hydro_t * obj, int index, double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    force[ia] = obj->f[addr_rank1(obj->nsite, NHDIM, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local_add
 *
 *  Accumulate (repeat, accumulate) the fluid force at site index.
 *
 *****************************************************************************/

__host__ __device__
int hydro_f_local_add(hydro_t * obj, int index, const double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->f[addr_rank1(obj->nsite, NHDIM, index, ia)] += force[ia]; 
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_rho_set
 *
 *****************************************************************************/

__host__ __device__ int hydro_rho_set(hydro_t * hydro, int index, double rho) {

  assert(hydro);

  hydro->rho->data[addr_rank0(hydro->nsite, index)] = rho;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_rho
 *
 *****************************************************************************/

__host__ __device__ int hydro_rho(hydro_t * hydro, int index, double * rho) {

  assert(hydro);
  assert(rho);

  *rho = hydro->rho->data[addr_rank0(hydro->nsite, index)];

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_set
 *
 *****************************************************************************/

__host__ __device__
int hydro_u_set(hydro_t * obj, int index, const double u[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->u[addr_rank1(obj->nsite, NHDIM, index, ia)] = u[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u
 *
 *****************************************************************************/

__host__ __device__
int hydro_u(hydro_t * obj, int index, double u[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    u[ia] = obj->u[addr_rank1(obj->nsite, NHDIM, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_zero
 *
 *****************************************************************************/

__host__ int hydro_u_zero(hydro_t * obj, const double uzero[NHDIM]) {

  dim3 nblk, ntpb;
  double * u = NULL;

  assert(obj);

  tdpAssert(tdpMemcpy(&u, &obj->target->u, sizeof(double *),
		      tdpMemcpyDeviceToHost));

  kernel_launch_param(obj->nsite, &nblk, &ntpb);
  tdpLaunchKernel(hydro_field_set, nblk, ntpb, 0, 0,
		  obj->target, u, uzero[X], uzero[Y], uzero[Z]);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  return 0;
}


/*****************************************************************************
 *
 *  hydro_f_zero
 *
 *****************************************************************************/

__host__ int hydro_f_zero(hydro_t * obj, const double fzero[NHDIM]) {

  dim3 nblk, ntpb;
  double * f;

  assert(obj);
  assert(obj->target);

  tdpAssert(tdpMemcpy(&f, &obj->target->f, sizeof(double *),
		      tdpMemcpyDeviceToHost));

  kernel_launch_param(obj->nsite, &nblk, &ntpb);
  tdpLaunchKernel(hydro_field_set, nblk, ntpb, 0, 0,
		  obj->target, f, fzero[X], fzero[Y], fzero[Z]);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  return 0;
}

/*****************************************************************************
 *
 *  hydro_rho0
 *
 *  Set rho uniformly everywhere; rho0 shouldn't be zero!
 *
 *****************************************************************************/

__host__ int hydro_rho0(hydro_t * obj, double rho0) {

  dim3 nblk, ntpb;
  double * rho = NULL;

  assert(obj);
  assert(obj->target);

  tdpAssert(tdpMemcpy(&rho, &obj->rho->target->data, sizeof(double *),
		      tdpMemcpyDeviceToHost));
  kernel_launch_param(obj->nsite, &nblk, &ntpb);
  tdpLaunchKernel(hydro_rho0_kernel, nblk, ntpb, 0, 0, obj->nsite, rho0, rho);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  return 0;
}

/******************************************************************************
 *
 *  hydro_rho0_kernel
 *
 *****************************************************************************/

__global__ void hydro_rho0_kernel(int nsite, double rho0, double * rho) {

  int kindex = 0;

  assert(rho);

  for_simt_parallel(kindex, nsite, 1) {
    rho[addr_rank0(nsite, kindex)] = rho0;
  }

  return;
}


/*****************************************************************************
 *
 *  hydro_field_set
 *
 *****************************************************************************/

static __global__
void hydro_field_set(hydro_t * hydro, double * field, double zx, double zy,
		     double zz) {

  int kindex;

  assert(hydro);
  assert(field);

  for_simt_parallel(kindex, hydro->nsite, 1) {
    field[addr_rank1(hydro->nsite, NHDIM, kindex, X)] = zx;
    field[addr_rank1(hydro->nsite, NHDIM, kindex, Y)] = zy;
    field[addr_rank1(hydro->nsite, NHDIM, kindex, Z)] = zz;
  }

  return;
}

/*****************************************************************************
 *
 *  hydro_lees_edwards
 *
 *  Compute the 'look-across-the-boundary' values of the velocity field,
 *  and update the velocity buffer region accordingly.
 *
 *  The communication might be improved:
 *  - only one buffer either side of the planes needs to be set?
 *  - only one communication per y sub domain if more than one buffer?
 *
 *****************************************************************************/

__host__ int hydro_lees_edwards(hydro_t * obj) {

  int nhalo;
  int nlocal[3]; /* Local system size */
  int nxbuffer;  /* Buffer planes */
  int ib;        /* Index in buffer region */
  int ib0;       /* buffer region offset */
  int ic;        /* Index corresponding x location in real system */

  int mpi_cartsz[3];
  int jc, kc, ia, index0, index1, index2;

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  int jdy;       /* Integral part of displacement */
  int j1, j2;    /* j values in real system to interpolate between */

  double ltot[3];
  double ule[3]; /* +/- velocity jump at plane */

  assert(obj);

  if (obj->le == NULL) return 0;

  cs_ltot(obj->cs, ltot);
  cs_cartsz(obj->cs, mpi_cartsz);

  /* All on host at the moment, so copy here and copy back at end */

  {
    int nplane = lees_edw_nplane_total(obj->le);
    if (nplane > 0) hydro_memcpy(obj, tdpMemcpyDeviceToHost);
  }

  if (mpi_cartsz[Y] > 1) {
    hydro_lees_edwards_parallel(obj);
  }
  else {

    cs_nhalo(obj->cs, &nhalo);
    cs_nlocal(obj->cs, nlocal);
    lees_edw_nxbuffer(obj->le, &nxbuffer);

    ib0 = nlocal[X] + nhalo + 1;

    for (ib = 0; ib < nxbuffer; ib++) {

      ic = lees_edw_ibuff_to_real(obj->le, ib);
      lees_edw_buffer_du(obj->le, ib, ule);

      lees_edw_buffer_dy(obj->le, ib, 1.0, &dy);
      dy = fmod(dy, ltot[Y]);
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

	/* Actually required here is j1 = jc - jdy - 1, but there's
	 * horrible modular arithmetic for the periodic boundaries
	 * to ensure 1 <= j1,j2 <= nlocal[Y] */

	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];

	/* If nhcomm < nhalo, we could use nhcomm here in the kc loop.
	 * (As j1 and j2 are always in the domain proper, jc can use nhalo.) */

	/* Note +/- nhcomm */
	for (kc = 1 - obj->nhcomm; kc <= nlocal[Z] + obj->nhcomm; kc++) {
	  index0 = lees_edw_index(obj->le, ib0 + ib, jc, kc);
	  index1 = lees_edw_index(obj->le, ic, j1, kc);
	  index2 = lees_edw_index(obj->le, ic, j2, kc);
	  for (ia = 0; ia < 3; ia++) {
	    obj->u[addr_rank1(obj->nsite, NHDIM, index0, ia)] = ule[ia] +
	      obj->u[addr_rank1(obj->nsite, NHDIM, index1, ia)]*fr +
	      obj->u[addr_rank1(obj->nsite, NHDIM, index2, ia)]*(1.0 - fr);
	  }
	}
      }
    }
  }

  {
    int nplane = lees_edw_nplane_total(obj->le);
    if (nplane > 0) hydro_memcpy(obj, tdpMemcpyHostToDevice);
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_lees_edwards_parallel
 *
 *  The Lees Edwards transformation for the velocity field in parallel.
 *  This is a linear interpolation.
 *
 *  Note that we communicate with up to 3 processors in each direction;
 *  this avoids having to update the halos completely.
 *
 *****************************************************************************/

static int hydro_lees_edwards_parallel(hydro_t * obj) {

  int ntotal[3];
  int nlocal[3];           /* Local system size */
  int noffset[3];          /* Local starting offset */
  int nxbuffer;            /* Number of buffer planes */
  int ib;                  /* Index in buffer region */
  int ib0;                 /* buffer region offset */
  int ic;                  /* Index corresponding x location in real system */
  int jc, kc, j1, j2;
  int n1, n2, n3;
  double dy;               /* Displacement for current ic->ib pair */
  double fr;               /* Fractional displacement */
  int jdy;                 /* Integral part of displacement */
  int index, ia;
  int nhalo;
  double ule[3];
  double ltot[3];

  int nsend;
  int nrecv;
  int      nrank_s[3];     /* send ranks */
  int      nrank_r[3];     /* recv ranks */
  const int tag0 = 1256;
  const int tag1 = 1257;
  const int tag2 = 1258;

  double * sbuf = NULL;   /* Send buffer */
  double * rbuf = NULL;   /* Interpolation buffer */

  MPI_Comm    le_comm;
  MPI_Request request[6];
  MPI_Status  status[3];

  assert(obj);

  cs_ltot(obj->cs, ltot);
  cs_nhalo(obj->cs, &nhalo);
  cs_ntotal(obj->cs, ntotal);
  cs_nlocal(obj->cs, nlocal);
  cs_nlocal_offset(obj->cs, noffset);
  ib0 = nlocal[X] + nhalo + 1;

  lees_edw_comm(obj->le, &le_comm);
  lees_edw_nxbuffer(obj->le, &nxbuffer);

  /* Allocate the temporary buffer */

  nsend = NHDIM*nlocal[Y]*(nlocal[Z] + 2*nhalo);
  nrecv = NHDIM*(nlocal[Y] + 2*nhalo + 1)*(nlocal[Z] + 2*nhalo);

  sbuf = (double *) calloc(nsend, sizeof(double));
  rbuf = (double *) calloc(nrecv, sizeof(double));
 
  if (sbuf == NULL) pe_fatal(obj->pe, "hydro: malloc(le sbuf) failed\n");
  if (rbuf == NULL) pe_fatal(obj->pe, "hydro: malloc(le rbuf) failed\n");


  /* One round of communication for each buffer plane */

  for (ib = 0; ib < nxbuffer; ib++) {

    ic = lees_edw_ibuff_to_real(obj->le, ib);
    lees_edw_buffer_du(obj->le, ib, ule);

    /* Work out the displacement-dependent quantities */

    lees_edw_buffer_dy(obj->le, ib, 1.0, &dy);
    dy = fmod(dy, ltot[Y]);
    jdy = floor(dy);
    fr  = dy - jdy;

    /* First j1 required is j1 = jc - jdy - 1 with jc = 1 - nhalo.
     * Modular arithmetic ensures 1 <= j1 <= ntotal[Y]. */

    jc = noffset[Y] + 1 - nhalo;
    j1 = 1 + (jc - jdy - 2 + 2*ntotal[Y]) % ntotal[Y];

    lees_edw_jstart_to_mpi_ranks(obj->le, j1, nrank_s, nrank_r);

    /* Local quantities: given a local starting index j2, we receive
     * n1 + n2 sites into the buffer, and send n1 sites starting with
     * j2, and the remaining n2 sites from starting position nhalo. */

    j2 = 1 + (j1 - 1) % nlocal[Y];

    n1 = (nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = imin(nlocal[Y], j2 + 2*nhalo)*(nlocal[Z] + 2*nhalo);
    n3 = imax(0, j2 - nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

    assert((n1+n2+n3) == (nlocal[Y] + 2*nhalo + 1)*(nlocal[Z] + 2*nhalo));

    /* Post receives, sends and wait for receives. */

    MPI_Irecv(rbuf, NHDIM*n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);
    MPI_Irecv(rbuf + NHDIM*n1, NHDIM*n2, MPI_DOUBLE, nrank_r[1], tag1,
	      le_comm, request + 1);
    MPI_Irecv(rbuf + NHDIM*(n1 + n2), NHDIM*n3, MPI_DOUBLE, nrank_r[2], tag2,
	      le_comm, request + 2);

    /* Load send buffer */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	index = lees_edw_index(obj->le, ic, jc, kc);
	for (ia = 0; ia < NHDIM; ia++) {
	  j1 = (jc - 1)*NHDIM*(nlocal[Z] + 2*nhalo) + NHDIM*(kc + nhalo - 1) + ia;
	  assert(j1 >= 0 && j1 < nsend);
	  sbuf[j1] = obj->u[addr_rank1(obj->nsite, NHDIM, index, ia)];
	}
      }
    }

    j1 = (j2 - 1)*NHDIM*(nlocal[Z] + 2*nhalo);
    MPI_Issend(sbuf + j1, NHDIM*n1, MPI_DOUBLE, nrank_s[0], tag0,
	       le_comm, request + 3);
    MPI_Issend(sbuf     , NHDIM*n2, MPI_DOUBLE, nrank_s[1], tag1,
	       le_comm, request + 4);
    MPI_Issend(sbuf     , NHDIM*n3, MPI_DOUBLE, nrank_s[2], tag2,
	       le_comm, request + 5);

    MPI_Waitall(3, request, status);

    /* Perform the actual interpolation from temporary buffer to
     * buffer region. */

    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

      j1 = (jc + nhalo - 1    )*(nlocal[Z] + 2*nhalo);
      j2 = (jc + nhalo - 1 + 1)*(nlocal[Z] + 2*nhalo);

      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	index = lees_edw_index(obj->le, ib0 + ib, jc, kc);
	for (ia = 0; ia < NHDIM; ia++) {
	  obj->u[addr_rank1(obj->nsite, NHDIM, index, ia)] = ule[ia]
	    + fr*rbuf[NHDIM*(j1 + kc + nhalo - 1) + ia]
	    + (1.0 - fr)*rbuf[NHDIM*(j2 + kc + nhalo - 1) + ia];
	}
      }
    }

    MPI_Waitall(3, request + 3, status);
  }

  free(sbuf);
  free(rbuf);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_write
 *
 *****************************************************************************/

static int hydro_u_write(FILE * fp, int index, void * arg) {

  int n;
  double u[3];
  hydro_t * obj = (hydro_t*) arg;

  assert(fp);
  assert(obj);

  hydro_u(obj, index, u);
  n = fwrite(u, sizeof(double), NHDIM, fp);
  if (n != NHDIM) pe_fatal(obj->pe, "fwrite(hydro->u) failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_write_ascii
 *
 *****************************************************************************/

static int hydro_u_write_ascii(FILE * fp, int index, void * arg) {

  int n;
  double u[3];
  hydro_t * obj = (hydro_t *) arg;

  assert(fp);
  assert(obj);

  hydro_u(obj, index, u);

  n = fprintf(fp, "%22.15e %22.15e %22.15e\n", u[X], u[Y], u[Z]);

  /* Expect total of 69 characters ... */
  if (n != 69) pe_fatal(obj->pe, "fprintf(hydro->u) failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_read
 *
 *****************************************************************************/

int hydro_u_read(FILE * fp, int index, void * self) {

  int n;
  double u[3];
  hydro_t * obj = (hydro_t *) self;

  assert(fp);
  assert(obj);

  n = fread(u, sizeof(double), NHDIM, fp);
  if (n != NHDIM) pe_fatal(obj->pe, "fread(hydro->u) failed\n");

  hydro_u_set(obj, index, u);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_read_ascii
 *
 *****************************************************************************/

static int hydro_u_read_ascii(FILE * fp, int index, void * self) {

  int n;
  double u[3];
  hydro_t * obj = (hydro_t *) self;

  assert(fp);
  assert(obj);

  n = fscanf(fp, "%le %le %le", &u[X], &u[Y], &u[Z]);
  if (n != NHDIM) pe_fatal(obj->pe, "fread(hydro->u) failed\n");

  hydro_u_set(obj, index, u);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_gradient_tensor
 *
 *  Return the velocity gradient tensor w_ab = d_b u_a at
 *  the site (ic, jc, kc).
 *
 *  The differencing is 2nd order centred.
 *
 *  This must take account of the Lees Edwards planes in  the x-direction.
 *
 *****************************************************************************/

__host__ int hydro_u_gradient_tensor(hydro_t * obj, int ic, int jc, int kc,
				     double w[3][3]) {

  int im1, ip1;
  double tr;

  assert(obj);

  im1 = lees_edw_ic_to_buff(obj->le, ic, -1);
  im1 = lees_edw_index(obj->le, im1, jc, kc);
  ip1 = lees_edw_ic_to_buff(obj->le, ic, +1);
  ip1 = lees_edw_index(obj->le, ip1, jc, kc);

  w[X][X] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, X)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, X)]);
  w[Y][X] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, Y)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, Y)]);
  w[Z][X] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, Z)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, Z)]);

  im1 = lees_edw_index(obj->le, ic, jc - 1, kc);
  ip1 = lees_edw_index(obj->le, ic, jc + 1, kc);

  w[X][Y] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, X)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, X)]);
  w[Y][Y] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, Y)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, Y)]);
  w[Z][Y] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, Z)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, Z)]);

  im1 = lees_edw_index(obj->le, ic, jc, kc - 1);
  ip1 = lees_edw_index(obj->le, ic, jc, kc + 1);

  w[X][Z] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, X)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, X)]);
  w[Y][Z] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, Y)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, Y)]);
  w[Z][Z] = 0.5*(obj->u[addr_rank1(obj->nsite, NHDIM, ip1, Z)] -
		 obj->u[addr_rank1(obj->nsite, NHDIM, im1, Z)]);

  /* Enforce tracelessness */

  tr = (1.0/3.0)*(w[X][X] + w[Y][Y] + w[Z][Z]);
  w[X][X] -= tr;
  w[Y][Y] -= tr;
  w[Z][Z] -= tr;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_correct_momentum
 *
 *  Driver to work out correction to momentum budget arising from
 *  non-conserving body force.
 *
 *****************************************************************************/

__host__ int hydro_correct_momentum(hydro_t * hydro) {

  int nlocal[3];
  double rv;
  double ltot[3];
  MPI_Comm comm;

  /* Net force */
  double fnet[3] = {0.0, 0.0, 0.0};
  double * fnetd = NULL;

  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(hydro);

  cs_nlocal(hydro->cs, nlocal);
  cs_cart_comm(hydro->cs, &comm);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(hydro->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpAssert(tdpMalloc((void **) &fnetd, 3*sizeof(double)));
  tdpAssert(tdpMemcpy(fnetd, fnet, 3*sizeof(double), tdpMemcpyHostToDevice));

  /* Accumulate net force */

  tdpLaunchKernel(hydro_accumulate_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, hydro->target, fnetd);

  tdpAssert(tdpPeekAtLastError());

  cs_ltot(hydro->cs, ltot);
  rv = 1.0/(ltot[X]*ltot[Y]*ltot[Z]);

  tdpAssert(tdpDeviceSynchronize());
  tdpAssert(tdpMemcpy(fnet, fnetd, 3*sizeof(double), tdpMemcpyDeviceToHost));

  /* Compute global correction */

  MPI_Allreduce(MPI_IN_PLACE, fnet, 3, MPI_DOUBLE, MPI_SUM, comm);

  fnet[X] = -fnet[X]*rv;
  fnet[Y] = -fnet[Y]*rv;
  fnet[Z] = -fnet[Z]*rv;

  /* Apply correction and finish */

  tdpMemcpy(fnetd, fnet, 3*sizeof(double), tdpMemcpyHostToDevice);

  tdpLaunchKernel(hydro_correct_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, hydro->target, fnetd);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());
  tdpFree(fnetd);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_accumulate_kernel
 *
 *  Work out the net total body force in the system.
 *
 *****************************************************************************/

__global__ void hydro_accumulate_kernel(kernel_ctxt_t * ktx, hydro_t * hydro,
					double fnet[3]) {

  int kindex;
  int kiterations;
  int tid;

  __shared__ double fx[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fy[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fz[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(hydro);

  tid = threadIdx.x;
  fx[TARGET_PAD*tid] = 0.0;
  fy[TARGET_PAD*tid] = 0.0;
  fz[TARGET_PAD*tid] = 0.0;

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc, index;
    double f[3];

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);
    hydro_f_local(hydro, index, f); 

    fx[TARGET_PAD*tid] += f[X];
    fy[TARGET_PAD*tid] += f[Y];
    fz[TARGET_PAD*tid] += f[Z];
  }

  __syncthreads();

  /* Reduction */

  if (tid == 0) {
    double fxb = 0.0;
    double fyb = 0.0;
    double fzb = 0.0;
    for (int it = 0; it < blockDim.x; it++) {
      fxb += fx[TARGET_PAD*it];
      fyb += fy[TARGET_PAD*it];
      fzb += fz[TARGET_PAD*it];
    }
    tdpAtomicAddDouble(fnet + X, fxb);
    tdpAtomicAddDouble(fnet + Y, fyb);
    tdpAtomicAddDouble(fnet + Z, fzb);
  }

  return;
}

/*****************************************************************************
 *
 *  hydro_accumulate_kernel_v
 *
 *  vectorised version of the above.
 *
 *****************************************************************************/

__global__ void hydro_accumulate_kernel_v(kernel_ctxt_t * ktx, hydro_t * hydro,
					  double fnet[3]) {

  int kindex;
  int kiterations;
  int tid;

  __shared__ double fx[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fy[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fz[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(hydro);

  tid = threadIdx.x;
  fx[TARGET_PAD*tid] = 0.0;
  fy[TARGET_PAD*tid] = 0.0;
  fz[TARGET_PAD*tid] = 0.0;

  kiterations = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiterations, NSIMDVL) {

    int index;
    int ia, iv;
    double f[3];

    index = kernel_baseindex(ktx, kindex);

    for (ia = 0; ia < 3; ia++) {
      double ftmp = 0.0;
      for_simd_v_reduction(iv, NSIMDVL, +: ftmp) {
        ftmp += hydro->f[addr_rank1(hydro->nsite, NHDIM, index+iv, ia)]; 
      }
      f[ia] = ftmp;
    }

    fx[TARGET_PAD*tid] += f[X];
    fy[TARGET_PAD*tid] += f[Y];
    fz[TARGET_PAD*tid] += f[Z];
  }

  __syncthreads();

  /* Reduction */

  if (tid == 0) {
    double fxb = 0.0;
    double fyb = 0.0;
    double fzb = 0.0;
    for (int it = 0; it < blockDim.x; it++) {
      fxb += fx[TARGET_PAD*it];
      fyb += fy[TARGET_PAD*it];
      fzb += fz[TARGET_PAD*it];
    }
    tdpAtomicAddDouble(fnet + X, fxb);
    tdpAtomicAddDouble(fnet + Y, fyb);
    tdpAtomicAddDouble(fnet + Z, fzb);
  }

  return;
}

/*****************************************************************************
 *
 *  hydro_correct_kernel
 *
 *  Add the net force correction to body force at each site.
 *
 *****************************************************************************/

__global__ void hydro_correct_kernel(kernel_ctxt_t * ktx, hydro_t * hydro,
				     double fnet[3]) {

  int kindex;
  int kiterations;
  int ic, jc, kc, index;

  assert(ktx);
  assert(hydro);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

    hydro_f_local_add(hydro, index, fnet);
  }

  return;
}

/*****************************************************************************
 *
 *  hydro_correct_kernel_v
 *
 *  Vectorised version of the above.
 *
 *****************************************************************************/

__global__ void hydro_correct_kernel_v(kernel_ctxt_t * ktx, hydro_t * hydro,
				       double fnet[3]) {

  int kindex;
  int kiterations;
  int index;
  int ia;
  int iv;

  assert(ktx);
  assert(hydro);

  kiterations = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiterations, NSIMDVL) {

    index = kernel_baseindex(ktx, kindex);

    for (ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	hydro->f[addr_rank1(hydro->nsite, NHDIM, index+iv, ia)] += fnet[ia]; 
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  hydro_halo_size
 *
 *****************************************************************************/

int hydro_halo_size(cs_limits_t lim) {

  int szx = 1 + lim.imax - lim.imin;
  int szy = 1 + lim.jmax - lim.jmin;
  int szz = 1 + lim.kmax - lim.kmin;

  return szx*szy*szz;
}

/*****************************************************************************
 *
 *  hydro_halo_enqueue_send
 *
 *****************************************************************************/

int hydro_halo_enqueue_send(const hydro_t * hydro, hydro_halo_t * h, int ireq) {

  assert(hydro);
  assert(h);
  assert(1 <= ireq && ireq < h->nvel);

  int nx = 1 + h->slim[ireq].imax - h->slim[ireq].imin;
  int ny = 1 + h->slim[ireq].jmax - h->slim[ireq].jmin;
  int nz = 1 + h->slim[ireq].kmax - h->slim[ireq].kmin;

  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  #pragma omp for nowait
  for (int ih = 0; ih < nx*ny*nz; ih++) {
    int ic = h->slim[ireq].imin + ih/strx;
    int jc = h->slim[ireq].jmin + (ih % strx)/stry;
    int kc = h->slim[ireq].kmin + (ih % stry)/strz;
    int index = cs_index(hydro->cs, ic, jc, kc);

    for (int ibf = 0; ibf < NHDIM; ibf++) {
      int uaddr = addr_rank1(hydro->nsite, NHDIM, index, ibf);
      h->send[ireq][ih*NHDIM + ibf] = hydro->u[uaddr];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_halo_dequeue_recv
 *
 *****************************************************************************/

int hydro_halo_dequeue_recv(hydro_t * hydro, const hydro_halo_t * h, int ireq) {
  assert(hydro);
  assert(h);
  assert(1 <= ireq && ireq < h->nvel);

  int nx = 1 + h->rlim[ireq].imax - h->rlim[ireq].imin;
  int ny = 1 + h->rlim[ireq].jmax - h->rlim[ireq].jmin;
  int nz = 1 + h->rlim[ireq].kmax - h->rlim[ireq].kmin;

  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  double * recv = h->recv[ireq];

  /* Check if this a copy from our own send buffer */
  {
    int i = 1 + h->cv[h->nvel - ireq][X];
    int j = 1 + h->cv[h->nvel - ireq][Y];
    int k = 1 + h->cv[h->nvel - ireq][Z];

    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) recv = h->send[ireq];
  }

  #pragma omp for nowait
  for (int ih = 0; ih < nx*ny*nz; ih++) {
    int ic = h->rlim[ireq].imin + ih/strx;
    int jc = h->rlim[ireq].jmin + (ih % strx)/stry;
    int kc = h->rlim[ireq].kmin + (ih % stry)/strz;
    int index = cs_index(hydro->cs, ic, jc, kc);

    for (int ibf = 0; ibf < NHDIM; ibf++) {
      int uaddr = addr_rank1(hydro->nsite, NHDIM, index, ibf);
      hydro->u[uaddr] = recv[ih*NHDIM + ibf];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_halo_create
 *
 *  It's convenient to borrow the velocity notation from the lb for
 *  the commnunication directions.
 *
 *****************************************************************************/

#include "lb_d3q27.h"

int hydro_halo_create(const hydro_t * hydro, hydro_halo_t * h) {

  int nlocal[3] = {0};

  assert(hydro);
  assert(h);

  *h = (hydro_halo_t) {0};

  /* Communictation model */

  cs_cart_comm(hydro->cs, &h->comm);

  {
    LB_CV_D3Q27(cv27);

    h->nvel = 27;
    for (int p = 0; p < h->nvel; p++) {
      h->cv[p][X] = cv27[p][X];
      h->cv[p][Y] = cv27[p][Y];
      h->cv[p][Z] = cv27[p][Z];
    }
  }

  /* Ranks of Cartesian neighbours */

  {
    int dims[3] = {0};
    int periods[3] = {0};
    int coords[3] = {0};

    MPI_Cart_get(h->comm, 3, dims, periods, coords);

    for (int p = 0; p < h->nvel; p++) {
      int nbr[3] = {0};
      int out[3] = {0};  /* Out-of-range is erroneous for non-perioidic dims */
      int i = 1 + h->cv[p][X];
      int j = 1 + h->cv[p][Y];
      int k = 1 + h->cv[p][Z];

      nbr[X] = coords[X] + h->cv[p][X];
      nbr[Y] = coords[Y] + h->cv[p][Y];
      nbr[Z] = coords[Z] + h->cv[p][Z];
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
    assert(h->nbrrank[1][1][1] == cs_cart_rank(hydro->cs));
  }

  /* Set out limits for send and recv regions. */

  cs_nlocal(hydro->cs, nlocal);

  for (int p = 1; p < h->nvel; p++) {

    int8_t cx = h->cv[p][X];
    int8_t cy = h->cv[p][Y];
    int8_t cz = h->cv[p][Z];
    int nhalo = hydro->nhcomm;

    cs_limits_t send = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    cs_limits_t recv = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};

    if (cx == -1) send.imax = nhalo;
    if (cx == +1) send.imin = send.imax - (nhalo - 1);
    if (cy == -1) send.jmax = nhalo;
    if (cy == +1) send.jmin = send.jmax - (nhalo - 1);
    if (cz == -1) send.kmax = nhalo;
    if (cz == +1) send.kmin = send.kmax - (nhalo - 1);

    /* For recv, direction is reversed cf. send */
    if (cx == +1) { recv.imin = 1 - nhalo;     recv.imax = 0;}
    if (cx == -1) { recv.imin = recv.imax + 1; recv.imax = recv.imax + nhalo;}
    if (cy == +1) { recv.jmin = 1 - nhalo;     recv.jmax = 0;}
    if (cy == -1) { recv.jmin = recv.jmax + 1; recv.jmax = recv.jmax + nhalo;}
    if (cz == +1) { recv.kmin = 1 - nhalo;     recv.kmax = 0;}
    if (cz == -1) { recv.kmin = recv.kmax + 1; recv.kmax = recv.kmax + nhalo;}

    h->slim[p] = send;
    h->rlim[p] = recv;
  }

  /* Message count and buffers (NHDIM is always 3 for u) */

  for (int p = 1; p < h->nvel; p++) {

    int scount = NHDIM*hydro_halo_size(h->slim[p]);
    int rcount = NHDIM*hydro_halo_size(h->rlim[p]);

    h->send[p] = (double *) calloc(scount, sizeof(double));
    h->recv[p] = (double *) calloc(rcount, sizeof(double));
    assert(h->send[p]);
    assert(h->recv[p]);
  }

  for (int ireq = 0; ireq < 2*27; ireq++) {
    h->request[ireq] = MPI_REQUEST_NULL;
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_halo_post
 *
 *****************************************************************************/

int hydro_halo_post(hydro_t * hydro) {

  assert(hydro);

  const int tagbase = 2022;
  hydro_halo_t * h = &hydro->h;

  /* Post recvs */

  TIMER_start(TIMER_HYDRO_HALO_IRECV);

  for (int ireq = 1; ireq < h->nvel; ireq++) {

    int i = 1 + h->cv[h->nvel - ireq][X];
    int j = 1 + h->cv[h->nvel - ireq][Y];
    int k = 1 + h->cv[h->nvel - ireq][Z];
    int mcount = NHDIM*hydro_halo_size(h->rlim[ireq]);

    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) mcount = 0;

    MPI_Irecv(h->recv[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
	      tagbase + ireq, h->comm, h->request + ireq);
  }

  TIMER_stop(TIMER_HYDRO_HALO_IRECV);

  /* Load send buffers; post sends */

  TIMER_start(TIMER_HYDRO_HALO_PACK);

  #pragma omp parallel
  {
    for (int ireq = 1; ireq < h->nvel; ireq++) {
      hydro_halo_enqueue_send(hydro, h, ireq);
    }
  }

  TIMER_stop(TIMER_HYDRO_HALO_PACK);

  TIMER_start(TIMER_HYDRO_HALO_ISEND);

  for (int ireq = 1; ireq < h->nvel; ireq++) {
    int i = 1 + h->cv[ireq][X];
    int j = 1 + h->cv[ireq][Y];
    int k = 1 + h->cv[ireq][Z];
    int mcount = NHDIM*hydro_halo_size(h->slim[ireq]);

    if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) mcount = 0;

    MPI_Isend(h->send[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
	      tagbase + ireq, h->comm, h->request + 27 + ireq);
  }

  TIMER_stop(TIMER_HYDRO_HALO_ISEND);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_halo_wait
 *
 *****************************************************************************/

int hydro_halo_wait(hydro_t * hydro) {

  assert(hydro);

  hydro_halo_t * h = &hydro->h;

  TIMER_start(TIMER_HYDRO_HALO_WAITALL);

  MPI_Waitall(2*h->nvel, h->request, MPI_STATUSES_IGNORE);

  TIMER_stop(TIMER_HYDRO_HALO_WAITALL);

  TIMER_start(TIMER_HYDRO_HALO_UNPACK);

  #pragma omp parallel
  {
    for (int ireq = 1; ireq < h->nvel; ireq++) {
      hydro_halo_dequeue_recv(hydro, h, ireq);
    }
  }

  TIMER_stop(TIMER_HYDRO_HALO_UNPACK);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_halo_free
 *
 *****************************************************************************/

int hydro_halo_free(hydro_halo_t * h) {

  assert(h);

  for (int p = 1; p < h->nvel; p++) {
    free(h->send[p]);
    free(h->recv[p]);
  }

  *h = (hydro_halo_t) {0};

  return 0;
}
