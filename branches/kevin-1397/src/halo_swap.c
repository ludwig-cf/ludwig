/*****************************************************************************
 *
 *  halo_swap.c
 *
 *  Lattice halo swap machinery.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "coords.h"
#include "halo_swap.h"

typedef struct halo_swap_param_s halo_swap_param_t;

struct halo_swap_s {
  halo_swap_param_t * param; 
  double * fxlo;
  double * fxhi;
  double * fylo;
  double * fyhi;
  double * fzlo;
  double * fzhi;
  double * hxlo;
  double * hxhi;
  double * hylo;
  double * hyhi;
  double * hzlo;
  double * hzhi;
  f_pack_t data_pack;       /* Pack buffer kernel function */
  f_unpack_t data_unpack;   /* Unpack buffer kernel function */
  cudaStream_t stream[3];   /* Stream for each of X,Y,Z */
  halo_swap_t * target;     /* Device memory */
};

/* Note nsite != naddr if extra memory has been allocated for LE
 * plane buffers. */

struct halo_swap_param_s {
  int nhalo;                /* coords_nhalo() */
  int nswap;                /* Width of actual halo swap <= nhalo */
  int nsite;                /* Total nall[X]*nall[Y]*nall[Z] */
  int na;                   /* Extent (rank 1 fields) */
  int nb;                   /* Extent (rank2 fields) */
  int naddr;                /* Extenet (nsite for address calculation) */
  int nfel;                 /* Field elements per site (double) */
  int nlocal[3];            /* local domain extent */
  int nall[3];              /* ... including 2*coords_nhalo */
  int hext[3][3];           /* halo extents ... see below */
  int hsz[3];               /* halo size in lattice sites each direction */
};

static __constant__ halo_swap_param_t const_param;

__host__ int halo_swap_create(int nhcomm, int naddr, int na, int nb, halo_swap_t ** phalo);
__host__ __target__ void halo_swap_coords(halo_swap_t * halo, int id, int index, int * ic, int * jc, int * kc);
__host__ __target__ int halo_swap_index(halo_swap_t * halo, int ic, int jc, int kc);
__host__ __target__ int halo_swap_bufindex(halo_swap_t * halo, int id, int ic, int jc, int kc);

/*****************************************************************************
 *
 *  halo_swap_create_r1
 *
 *  Rank1 addressable objects.
 *
 *****************************************************************************/

__host__ int halo_swap_create_r1(int nhcomm, int naddr, int na,
				 halo_swap_t ** p) {

  return halo_swap_create(nhcomm, naddr, na, 1, p);
}

/*****************************************************************************
 *
 *  halo_swap_create_r2
 *
 *  Rank2 addressable objects
 *
 *****************************************************************************/

__host__ int halo_swap_create_r2(int nhcomm, int naddr, int na, int nb,
				 halo_swap_t ** p) {

  return halo_swap_create(nhcomm, naddr, na, nb, p);
}

/*****************************************************************************
 *
 *  halo_swap_create
 *
 *****************************************************************************/

__host__ int halo_swap_create(int nhcomm, int naddr, int na, int nb,
			      halo_swap_t ** phalo) {

  int sz;
  int nhalo;
  int ndevice;
  unsigned int mflag = cudaHostAllocDefault;

  halo_swap_t * halo = NULL;

  assert(phalo);

  halo = (halo_swap_t *) calloc(1, sizeof(halo_swap_t));
  assert(halo);

  halo->param = (halo_swap_param_t *) calloc(1, sizeof(halo_swap_param_t));
  assert(halo->param);

  /* Template for distributions, which is used to allocate buffers;
   * assumed to be large enough for any halo transfer... */

  nhalo = coords_nhalo();

  halo->param->na = na;
  halo->param->nb = nb;
  halo->param->nhalo = nhalo;
  halo->param->nswap = nhcomm;
  halo->param->nfel = na*nb;
  halo->param->naddr = naddr;
  coords_nlocal(halo->param->nlocal);
  coords_nall(halo->param->nall);

  halo->param->nsite = halo->param->nall[X]*halo->param->nall[Y]*halo->param->nall[Z];

  /* Halo extents:  hext[X] = {1, nall[Y], nall[Z]}
                    hext[Y] = {nall[X], 1, nall[Z]}
                    hext[Z] = {nall[X], nall[Y], 1} */

  halo->param->hext[X][X] = halo->param->nswap;
  halo->param->hext[X][Y] = halo->param->nall[Y];
  halo->param->hext[X][Z] = halo->param->nall[Z];
  halo->param->hext[Y][X] = halo->param->nall[X];
  halo->param->hext[Y][Y] = halo->param->nswap;
  halo->param->hext[Y][Z] = halo->param->nall[Z];
  halo->param->hext[Z][X] = halo->param->nall[X];
  halo->param->hext[Z][Y] = halo->param->nall[Y];
  halo->param->hext[Z][Z] = halo->param->nswap;

  halo->param->hsz[X] = nhcomm*halo->param->hext[X][Y]*halo->param->hext[X][Z];
  halo->param->hsz[Y] = nhcomm*halo->param->hext[Y][X]*halo->param->hext[Y][Z];
  halo->param->hsz[Z] = nhcomm*halo->param->hext[Z][X]*halo->param->hext[Z][Y];

  /* Host buffers, actual and halo regions */

  sz = halo->param->hsz[X]*na*nb*sizeof(double);
  cudaHostAlloc((void **) &halo->fxlo, sz, mflag);
  cudaHostAlloc((void **) &halo->fxhi, sz, mflag);
  cudaHostAlloc((void **) &halo->hxlo, sz, mflag);
  cudaHostAlloc((void **) &halo->hxhi, sz, mflag);

  sz = halo->param->hsz[Y]*na*nb*sizeof(double);
  cudaHostAlloc((void **) &halo->fylo, sz, mflag);
  cudaHostAlloc((void **) &halo->fyhi, sz, mflag);
  cudaHostAlloc((void **) &halo->hylo, sz, mflag);
  cudaHostAlloc((void **) &halo->hyhi, sz, mflag);

  sz = halo->param->hsz[Z]*na*nb*sizeof(double);
  cudaHostAlloc((void **) &halo->fzlo, sz, mflag);
  cudaHostAlloc((void **) &halo->fzhi, sz, mflag);
  cudaHostAlloc((void **) &halo->hzlo, sz, mflag);
  cudaHostAlloc((void **) &halo->hzhi, sz, mflag);

  /* Device buffers: allocate or alias */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    halo->target = halo;
  }
  else {
    double * tmp;

    /* Target structure */
    targetCalloc((void **) &halo->target, sizeof(halo_swap_t));

    /* Buffers */
    sz = halo->param->hsz[X]*na*nb*sizeof(double);

    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fxlo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fxhi, &tmp, sizeof(double *));

    targetCalloc((void **) & tmp, sz);
    copyToTarget(&halo->target->hxlo, &tmp, sizeof(double *));
    targetCalloc((void **) & tmp, sz);
    copyToTarget(&halo->target->hxhi, &tmp, sizeof(double *));

    sz = halo->param->hsz[Y]*na*nb*sizeof(double);

    targetCalloc((void ** ) &tmp, sz);
    copyToTarget(&halo->target->fylo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fyhi, &tmp, sizeof(double *));

    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->hylo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->hyhi, &tmp, sizeof(double *));

    sz = halo->param->hsz[Z]*na*nb*sizeof(double);

    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fzlo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fzhi, &tmp, sizeof(double *));

    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->hzlo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->hzhi, &tmp, sizeof(double *));

    /* Device constants */
    halo_swap_commit(halo);
  }

  *phalo = halo;

  return 0;
}

/*****************************************************************************
 *
 *  halo_swap_free
 *
 *****************************************************************************/

__host__ int halo_swap_free(halo_swap_t * halo) {

  int ndevice;

  assert(halo);

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    double * tmp;

    copyFromTarget(&tmp, &halo->target->fxlo, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->fxhi, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->fylo, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->fyhi, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->fzlo, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->fzhi, sizeof(double *));
    targetFree(tmp);

    copyFromTarget(&tmp, &halo->target->hxlo, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->hxhi, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->hylo, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->hyhi, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->hzlo, sizeof(double *));
    targetFree(tmp);
    copyFromTarget(&tmp, &halo->target->hzhi, sizeof(double *));
    targetFree(tmp);

    targetFree(halo->target);
  }

  cudaFreeHost(halo->fxlo);
  cudaFreeHost(halo->fxhi);
  cudaFreeHost(halo->fylo);
  cudaFreeHost(halo->fyhi);
  cudaFreeHost(halo->fzlo);
  cudaFreeHost(halo->fzhi);

  cudaFreeHost(halo->hxlo);
  cudaFreeHost(halo->hxhi);
  cudaFreeHost(halo->hylo);
  cudaFreeHost(halo->hyhi);
  cudaFreeHost(halo->hzlo);
  cudaFreeHost(halo->hzhi);
  free(halo->param);
  free(halo);

  return 0;
}

/*****************************************************************************
 *
 *  halo_swap_handlers_set
 *
 *****************************************************************************/

__host__ int halo_swap_handlers_set(halo_swap_t * halo, f_pack_t pack,
				    f_unpack_t unpack) {

  assert(halo);

  halo->data_pack = pack;
  halo->data_unpack = unpack;

  return 0;
}

/*****************************************************************************
 *
 *  halo_swap_commit
 *
 *****************************************************************************/

__host__ int halo_swap_commit(halo_swap_t * halo) {

  assert(halo);

  copyConstToTarget(&const_param, halo->param, sizeof(halo_swap_param_t));

  return 0;
}

/*****************************************************************************
 *
 *  halo_swap_driver
 *
 *  "data" needs to be a device pointer
 *
 *****************************************************************************/

__host__ int halo_swap_driver(halo_swap_t * halo, double * data) {

  int ncount;
  int ndevice;
  int ic, jc, kc;
  int ih, jh, kh;
  int ixlo, ixhi;
  int iylo, iyhi;
  int izlo, izhi;  
  int m, mc, p;
  int nd, nh;
  int hsz[3];
  dim3 nblk, ntpb;

  MPI_Comm comm = cart_comm();

  MPI_Request req_x[4];
  MPI_Request req_y[4];
  MPI_Request req_z[4];
  MPI_Status  status[4];

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;

  assert(halo);

  targetGetDeviceCount(&ndevice);
  halo_swap_commit(halo);

  /* hsz[] is just shorthand for local halo sizes */
  /* An offset nd is required if nswap < nhalo */

  hsz[X] = halo->param->hsz[X];
  hsz[Y] = halo->param->hsz[Y];
  hsz[Z] = halo->param->hsz[Z];
  nh = halo->param->nhalo;
  nd = nh - halo->param->nswap;

  /* POST ALL RELEVANT Irecv() ahead of time */

  for (p = 0; p < 4; p++) {
    req_x[p] = MPI_REQUEST_NULL;
    req_y[p] = MPI_REQUEST_NULL;
    req_z[p] = MPI_REQUEST_NULL;
  }

  if (cart_size(X) > 1) {
    ncount = halo->param->hsz[X]*halo->param->nfel;
    MPI_Irecv(halo->hxlo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), ftagx, comm, req_x);
    MPI_Irecv(halo->hxhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), btagx, comm, req_x + 1);
  }

  if (cart_size(Y) > 1) {
    ncount = halo->param->hsz[Y]*halo->param->nfel;
    MPI_Irecv(halo->hylo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), ftagy, comm, req_y);
    MPI_Irecv(halo->hyhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), btagy, comm, req_y + 1);
  }

  if (cart_size(Z) > 1) {
    ncount = halo->param->hsz[Z]*halo->param->nfel;
    MPI_Irecv(halo->hzlo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), ftagz, comm, req_z);
    MPI_Irecv(halo->hzhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), btagz, comm, req_z + 1);
  }

  /* pack X edges on accelerator */
  /* pack Y edges on accelerator */
  /* pack Z edges on accelerator */

  kernel_launch_param(hsz[X], &nblk, &ntpb);
  __host_launch4s(halo->data_pack, nblk, ntpb, 0, halo->stream[X],
		  halo->target, X, data);

  if (ndevice > 0) {
    ncount = halo->param->hsz[X]*halo->param->nfel;
    cudaMemcpyAsync(halo->fxlo, halo->target->fxlo, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[X]);
    cudaMemcpyAsync(halo->fxhi, halo->target->fxhi, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[X]);
  }

  kernel_launch_param(hsz[Y], &nblk, &ntpb);
  __host_launch4s(halo->data_pack, nblk, ntpb, 0, halo->stream[Y],
		  halo->target, Y, data);

  if (ndevice > 0) {
    ncount = halo->param->hsz[Y]*halo->param->nfel;
    cudaMemcpyAsync(halo->fylo, halo->target->fylo, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[Y]);
    cudaMemcpyAsync(halo->fyhi, halo->target->fyhi, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[Y]);
  }

  kernel_launch_param(hsz[Z], &nblk, &ntpb);
  __host_launch4s(halo->data_pack, nblk, ntpb, 0, halo->stream[Z],
		  halo->target, Z, data);

  if (ndevice > 0) {
    ncount = halo->param->hsz[Z]*halo->param->nfel;
    cudaMemcpyAsync(halo->fzlo, halo->target->fzlo, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[Z]);
    cudaMemcpyAsync(halo->fzhi, halo->fzhi, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[Z]);
  }


 /* Wait for X; copy or MPI recvs; put X halos back on device, and unpack */

  cudaStreamSynchronize(halo->stream[X]);
  ncount = halo->param->hsz[X]*halo->param->nfel;

  if (cart_size(X) == 1) {
    /* note these copies do not alias for ndevice == 1 */
    /* fxhi -> hxlo */
    cudaMemcpyAsync(halo->target->hxlo, halo->fxhi, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[X]);
    /* fxlo -> hxhi */
    cudaMemcpyAsync(halo->target->hxhi, halo->fxlo, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[X]);
  }
  else {
    MPI_Isend(halo->fxhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), ftagx, comm, req_x + 2);
    MPI_Isend(halo->fxlo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), btagx, comm, req_x + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_x, &mc, status);
      if (mc == 0 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hxlo, halo->hxlo, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[X]);
      }
      if (mc == 1 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hxhi, halo->hxhi, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[X]);
      }
    }
  }

  kernel_launch_param(hsz[X], &nblk, &ntpb);
  __host_launch4s(halo->data_unpack, nblk, ntpb, 0, halo->stream[X],
		  halo->target, X, data);

  /* Now wait for Y data to arrive from device */
  /* Fill in 4 corners of Y edge data from X halo */

  cudaStreamSynchronize(halo->stream[Y]);

  ih = halo->param->hext[Y][X] - nh;
  jh = halo->param->hext[X][Y] - nh - halo->param->nswap;

  for (ic = 0; ic < halo->param->nswap; ic++) {
    for (jc = 0; jc < halo->param->nswap; jc++) {
      for (kc = 0; kc < halo->param->nall[Z]; kc++) {

	/* This looks a bit odd, but iylo and ixhi relate to Y halo,
	 * and ixlo and iyhi relate to X halo buffers */
        ixlo = halo_swap_bufindex(halo, X,      ic, nh + jc, kc);
        iylo = halo_swap_bufindex(halo, Y, nd + ic,      jc, kc);
        ixhi = halo_swap_bufindex(halo, Y, ih + ic,      jc, kc);
        iyhi = halo_swap_bufindex(halo, X, ic,      jh + jc, kc);

        for (p = 0; p < halo->param->nfel; p++) {
          halo->fylo[hsz[Y]*p + iylo] = halo->hxlo[hsz[X]*p + ixlo];
          halo->fyhi[hsz[Y]*p + iylo] = halo->hxlo[hsz[X]*p + iyhi];
          halo->fylo[hsz[Y]*p + ixhi] = halo->hxhi[hsz[X]*p + ixlo];
          halo->fyhi[hsz[Y]*p + ixhi] = halo->hxhi[hsz[X]*p + iyhi];
        }
      }
    }
  }


  /* Swap in Y, send data back to device and unpack */

  ncount = halo->param->hsz[Y]*halo->param->nfel;

  if (cart_size(Y) == 1) {
    /* fyhi -> hylo */
    cudaMemcpyAsync(halo->target->hylo, halo->fyhi, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[Y]);
    /* fylo -> hyhi */
    cudaMemcpyAsync(halo->target->hyhi, halo->fylo,ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[Y]);
  }
  else {
    MPI_Isend(halo->fyhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), ftagy, comm, req_y + 2);
    MPI_Isend(halo->fylo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), btagy, comm, req_y + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_y, &mc, status);
      if (mc == 0 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hylo, halo->hylo, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[Y]);
      }
      if (mc == 1 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hyhi, halo->hyhi, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[Y]);
      }
    }
  }

  kernel_launch_param(hsz[Y], &nblk, &ntpb);
  __host_launch4s(halo->data_unpack, nblk, ntpb, 0, halo->stream[Y],
		  halo->target, Y, data);

  /* Wait for Z data from device */
  /* Fill in 4 corners of Z edge data from X halo  */

  cudaStreamSynchronize(halo->stream[Z]);

  ih = halo->param->hext[Z][X] - nh;
  kh = halo->param->hext[X][Z] - nh - halo->param->nswap;

  for (ic = 0; ic < halo->param->nswap; ic++) {
    for (jc = 0; jc < halo->param->nall[Y]; jc++) {
      for (kc = 0; kc < halo->param->nswap; kc++) {

        ixlo = halo_swap_bufindex(halo, X,      ic, jc, nh + kc);
        izlo = halo_swap_bufindex(halo, Z, nd + ic, jc,      kc);
        ixhi = halo_swap_bufindex(halo, X,      ic, jc, kh + kc);
	izhi = halo_swap_bufindex(halo, Z, ih + ic, jc,      kc);

        for (p = 0; p < halo->param->nfel; p++) {
          halo->fzlo[hsz[Z]*p + izlo] = halo->hxlo[hsz[X]*p + ixlo];
          halo->fzhi[hsz[Z]*p + izlo] = halo->hxlo[hsz[X]*p + ixhi];
          halo->fzlo[hsz[Z]*p + izhi] = halo->hxhi[hsz[X]*p + ixlo];
          halo->fzhi[hsz[Z]*p + izhi] = halo->hxhi[hsz[X]*p + ixhi];
        }
      }
    }
  }

  /* Fill in 4 strips in X of Z edge data: from Y halo  */

  jh = halo->param->hext[Z][Y] - nh;
  kh = halo->param->hext[Y][Z] - nh - halo->param->nswap;
  
  for (ic = 0; ic < halo->param->nall[X]; ic++) {
    for (jc = 0; jc < halo->param->nswap; jc++) {
      for (kc = 0; kc < halo->param->nswap; kc++) {

        iylo = halo_swap_bufindex(halo, Y, ic,      jc, nh + kc);
        izlo = halo_swap_bufindex(halo, Z, ic, nd + jc,      kc);
        iyhi = halo_swap_bufindex(halo, Y, ic,      jc, kh + kc);
        izhi = halo_swap_bufindex(halo, Z, ic, jh + jc,      kc);

        for (p = 0; p < halo->param->nfel; p++) {
          halo->fzlo[hsz[Z]*p + izlo] = halo->hylo[hsz[Y]*p + iylo];
          halo->fzhi[hsz[Z]*p + izlo] = halo->hylo[hsz[Y]*p + iyhi];
          halo->fzlo[hsz[Z]*p + izhi] = halo->hyhi[hsz[Y]*p + iylo];
          halo->fzhi[hsz[Z]*p + izhi] = halo->hyhi[hsz[Y]*p + iyhi];
        }
      }
    }
  }

  /* The z-direction swap  */

  ncount = halo->param->hsz[Z]*halo->param->nfel;

  if (cart_size(Z) == 1) {
    /* fzhi -> hzlo */
    cudaMemcpyAsync(halo->target->hzlo, halo->fzhi, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[Z]);
    /* fzlo -> hzhi */
    cudaMemcpyAsync(halo->target->hzhi, halo->fzlo, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[Z]);
  }
  else {
    MPI_Isend(halo->fzhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), ftagz, comm, req_z + 2);
    MPI_Isend(halo->fzlo,  ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), btagz, comm, req_z + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_z, &mc, status);
      if (mc == 0 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hzlo, halo->hzlo, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[Z]);
      }
    }
    if (mc == 1 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hzhi, halo->hzhi, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[Z]);
    }
  }

  kernel_launch_param(hsz[Z], &nblk, &ntpb);
  __host_launch4s(halo->data_unpack, nblk, ntpb, 0, halo->stream[Z],
		  halo->target, Z, data);

  cudaStreamSynchronize(halo->stream[X]);
  cudaStreamSynchronize(halo->stream[Y]);
  cudaStreamSynchronize(halo->stream[Z]);

  return 0;
}

/*****************************************************************************
 *
 *  halo_swap_pack_rank1
 *
 *  Move data to halo buffer on device for coordinate
 *  direction id at both low and high ends.
 *
 *****************************************************************************/

__global__
void halo_swap_pack_rank1(halo_swap_t * halo, int id, double * data) {

  int kindex;

  __target_simt_parallel_for(kindex, halo->param->hsz[id], 1) {

    int nh;
    int na;
    int naddr;
    int ia, indexl, indexh, ic, jc, kc;
    int hsz;
    int ho; /* high end offset */
    double * __restrict__ buflo;
    double * __restrict__ bufhi;

    naddr = halo->param->naddr;
    na = halo->param->na;
    hsz = halo->param->hsz[id];

    /* Load two buffers for this site */
    /* Use full nhalo to address full data */

    nh = halo->param->nhalo;
    halo_swap_coords(halo, id, kindex, &ic, &jc, &kc);

    if (id == X) {
      ho = nh + halo->param->nlocal[X] - halo->param->nswap;
      indexl = halo_swap_index(halo, nh + ic, jc, kc);
      indexh = halo_swap_index(halo, ho + ic, jc, kc);
      buflo = halo->fxlo;
      bufhi = halo->fxhi;
    }
    if (id == Y) {
      ho = nh + halo->param->nlocal[Y] - halo->param->nswap;
      indexl = halo_swap_index(halo, ic, nh + jc, kc);
      indexh = halo_swap_index(halo, ic, ho + jc, kc);
      buflo = halo->fylo;
      bufhi = halo->fyhi;
    }
    if (id == Z) {
      ho = nh + halo->param->nlocal[Z] - halo->param->nswap;
      indexl = halo_swap_index(halo, ic, jc, nh + kc);
      indexh = halo_swap_index(halo, ic, jc, ho + kc);
      buflo = halo->fzlo;
      bufhi = halo->fzhi;
    }

    if (halo->param->nb == 1) {

      /* Rank 1 */

      /* Low end, and high end */

      for (ia = 0; ia < na; ia++) {
	buflo[hsz*ia + kindex] = data[addr_rank1(naddr, na, indexl, ia)];
      }

      for (ia = 0; ia < na; ia++) {
	bufhi[hsz*ia + kindex] = data[addr_rank1(naddr, na, indexh, ia)];
      }
    }
    else {
      int nb = halo->param->nb;
      int ib, nel;

      nel = 0;
      for (ia = 0; ia < na; ia++) {
	for (ib = 0; ib < nb; ib++) {
	  buflo[hsz*nel + kindex] = data[addr_rank2(naddr, na, nb, indexl, ia, ib)];
	  nel += 1;
	}
      }

      nel = 0;
      for (ia = 0; ia < na; ia++) {
	for (ib = 0; ib < nb; ib++) {
	  bufhi[hsz*nel + kindex] = data[addr_rank2(naddr, na, nb, indexh, ia, ib)];
	  nel += 1;
	}
      }

    }
  }

  return;
}

/*****************************************************************************
 *
 *  halo_swap_unpack_rank1
 *
 *  Unpack halo buffers to the distribution on device for direction id.
 *
 *****************************************************************************/

__global__
void halo_swap_unpack_rank1(halo_swap_t * halo, int id, double * data) {

  int kindex;

  /* Unpack buffer this site. */

  __target_simt_parallel_for(kindex, halo->param->hsz[id], 1) {

    int naddr;
    int na;
    int hsz;
    int ia, indexl, indexh;
    int nh;                          /* Full halo width */
    int ic, jc, kc;                  /* Lattice ooords */
    int lo, ho;                      /* Offset for low, high end */
    double * __restrict__ buflo;
    double * __restrict__ bufhi;

    naddr = halo->param->naddr;
    na = halo->param->na;
    hsz = halo->param->hsz[id];

    nh = halo->param->nhalo;
    halo_swap_coords(halo, id, kindex, &ic, &jc, &kc);

    if (id == X) {
      lo = nh - halo->param->nswap;
      ho = nh + halo->param->nlocal[X];
      indexl = halo_swap_index(halo, lo + ic, jc, kc);
      indexh = halo_swap_index(halo, ho + ic, jc, kc);
      buflo = halo->hxlo;
      bufhi = halo->hxhi;
    }

    if (id == Y) {
      lo = nh - halo->param->nswap;
      ho = nh + halo->param->nlocal[Y];
      indexl = halo_swap_index(halo, ic, lo + jc, kc);
      indexh = halo_swap_index(halo, ic, ho + jc, kc);
      buflo = halo->hylo;
      bufhi = halo->hyhi;
    }

    if (id == Z) {
      lo = nh - halo->param->nswap;
      ho = nh + halo->param->nlocal[Z];
      indexl = halo_swap_index(halo, ic, jc, lo + kc);
      indexh = halo_swap_index(halo, ic, jc, ho + kc);
      buflo = halo->hzlo;
      bufhi = halo->hzhi;
    } 


    if (halo->param->nb == 1) {

      /* Rank 1 */
      /* Low end, then high end */

      for (ia = 0; ia < na; ia++) {
	data[addr_rank1(naddr, na, indexl, ia)] = buflo[hsz*ia + kindex];
      }

      for (ia = 0; ia < na; ia++) {
	data[addr_rank1(naddr, na, indexh, ia)] = bufhi[hsz*ia + kindex];
      }

    }
    else {

      int nb = halo->param->nb;
      int ib, nel;

      nel = 0;
      for (ia = 0; ia < na; ia++) {
	for (ib = 0; ib < nb; ib++) {
	  data[addr_rank2(naddr, na, nb, indexl, ia, ib)] = buflo[hsz*nel + kindex];
	  nel += 1;
	}
      }

      nel = 0;
      for (ia = 0; ia < na; ia++) {
	for (ib = 0; ib < nb; ib++) {
	  data[addr_rank2(naddr, na, nb, indexh, ia, ib)] = bufhi[hsz*nel + kindex];
	  nel += 1;
	}
      }

    }
  }

  return;
}

/*****************************************************************************
 *
 *  halo_swap_coords
 *
 *  For given kernel index, work out where we are in (ic, jc, kc)
 *
 *****************************************************************************/

__host__ __target__
void halo_swap_coords(halo_swap_t * halo, int id, int index,
		      int * ic, int * jc, int * kc) {
  int xstr;
  int ystr;

  assert(halo);

  ystr = halo->param->hext[id][Z];
  xstr = ystr*halo->param->hext[id][Y];

  *ic = index/xstr;
  *jc = (index - *ic*xstr)/ystr;
  *kc = index - *ic*xstr - *jc*ystr;

  return;
}

/*****************************************************************************
 *
 *  halo_swap_index
 *
 *  A special case of coords_index().
 *
 *****************************************************************************/

__host__ __target__
int halo_swap_index(halo_swap_t * halo, int ic, int jc, int kc) {

  int xstr;
  int ystr;

  assert(halo);

  ystr = halo->param->nall[Z];
  xstr = ystr*halo->param->nall[Y];

  return (ic*xstr + jc*ystr + kc);
}

/*****************************************************************************
 *
 *  halo_swap_bufindex
 *
 *  Computes index for buffer direction id
 *
 *****************************************************************************/

__host__ __target__
int halo_swap_bufindex(halo_swap_t * halo, int id, int ic, int jc, int kc) {

  int xstr;
  int ystr;

  assert(halo);

  ystr = halo->param->hext[id][Z];
  xstr = ystr*halo->param->hext[id][Y];

  return (ic*xstr + jc*ystr + kc);
}
