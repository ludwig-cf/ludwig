/*****************************************************************************
 *
 *  halo_swap.c
 *
 *  Lattice halo swap machinery.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "halo_swap.h"

typedef struct halo_swap_param_s halo_swap_param_t;

struct halo_swap_s {
  pe_t * pe;
  cs_t * cs;
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
  tdpStream_t stream[3];    /* Stream for each of X,Y,Z */
  halo_swap_t * target;     /* Device memory */
};

/* Note nsite != naddr if extra memory has been allocated for LE
 * plane buffers. */

struct halo_swap_param_s {
  int nhalo;                /* cs_nhalo */
  int nswap;                /* Width of actual halo swap <= nhalo */
  int nsite;                /* Total nall[X]*nall[Y]*nall[Z] */
  int na;                   /* Extent (rank 1 fields) */
  int nb;                   /* Extent (rank2 fields) */
  int naddr;                /* Extenet (nsite for address calculation) */
  int nfel;                 /* Field elements per site (double) */
  int nlocal[3];            /* local domain extent */
  int nall[3];              /* ... including 2*cs_nhalo */
  int hext[3][3];           /* halo extents ... see below */
  int hsz[3];               /* halo size in lattice sites each direction */
};

static __constant__ halo_swap_param_t const_param;

__host__ int halo_swap_create(pe_t * pe, cs_t * cs, int nhcomm, int naddr,
			      int na, int nb, halo_swap_t ** phalo);
__host__ __device__ void halo_swap_coords(halo_swap_t * halo, int id, int index, int * ic, int * jc, int * kc);
__host__ __device__ int halo_swap_index(halo_swap_t * halo, int ic, int jc, int kc);
__host__ __device__ int halo_swap_bufindex(halo_swap_t * halo, int id, int ic, int jc, int kc);

/*****************************************************************************
 *
 *  halo_swap_create_r1
 *
 *  Rank1 addressable objects.
 *
 *****************************************************************************/

__host__ int halo_swap_create_r1(pe_t * pe, cs_t * cs, int nhcomm, int naddr,
				 int na,
				 halo_swap_t ** p) {

  return halo_swap_create(pe, cs, nhcomm, naddr, na, 1, p);
}

/*****************************************************************************
 *
 *  halo_swap_create_r2
 *
 *  Rank2 addressable objects
 *
 *****************************************************************************/

__host__ int halo_swap_create_r2(pe_t * pe, cs_t *cs, int nhcomm, int naddr,
				 int na, int nb,
				 halo_swap_t ** p) {

  return halo_swap_create(pe, cs, nhcomm, naddr, na, nb, p);
}

/*****************************************************************************
 *
 *  halo_swap_create
 *
 *****************************************************************************/

__host__ int halo_swap_create(pe_t * pe, cs_t * cs, int nhcomm, int naddr,
			      int na, int nb,
			      halo_swap_t ** phalo) {

  int sz;
  int nhalo;
  int ndevice;
  unsigned int mflag = tdpHostAllocDefault;

  halo_swap_t * halo = NULL;

  assert(pe);
  assert(cs);
  assert(phalo);

  halo = (halo_swap_t *) calloc(1, sizeof(halo_swap_t));
  assert(halo);

  halo->param = (halo_swap_param_t *) calloc(1, sizeof(halo_swap_param_t));
  assert(halo->param);

  /* Template for distributions, which is used to allocate buffers;
   * assumed to be large enough for any halo transfer... */

  halo->pe = pe;
  halo->cs = cs;

  cs_nhalo(cs, &nhalo);

  halo->param->na = na;
  halo->param->nb = nb;
  halo->param->nhalo = nhalo;
  halo->param->nswap = nhcomm;
  halo->param->nfel = na*nb;
  halo->param->naddr = naddr;
  cs_nlocal(cs, halo->param->nlocal);
  cs_nall(cs, halo->param->nall);

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
  tdpHostAlloc((void **) &halo->fxlo, sz, mflag);
  tdpHostAlloc((void **) &halo->fxhi, sz, mflag);
  tdpHostAlloc((void **) &halo->hxlo, sz, mflag);
  tdpHostAlloc((void **) &halo->hxhi, sz, mflag);

  sz = halo->param->hsz[Y]*na*nb*sizeof(double);
  tdpHostAlloc((void **) &halo->fylo, sz, mflag);
  tdpHostAlloc((void **) &halo->fyhi, sz, mflag);
  tdpHostAlloc((void **) &halo->hylo, sz, mflag);
  tdpHostAlloc((void **) &halo->hyhi, sz, mflag);

  sz = halo->param->hsz[Z]*na*nb*sizeof(double);
  tdpHostAlloc((void **) &halo->fzlo, sz, mflag);
  tdpHostAlloc((void **) &halo->fzhi, sz, mflag);
  tdpHostAlloc((void **) &halo->hzlo, sz, mflag);
  tdpHostAlloc((void **) &halo->hzhi, sz, mflag);

  tdpStreamCreate(&halo->stream[X]);
  tdpStreamCreate(&halo->stream[Y]);
  tdpStreamCreate(&halo->stream[Z]);

  /* Device buffers: allocate or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    halo->target = halo;
  }
  else {
    double * tmp;
    halo_swap_param_t * tmpp;

    /* Target structure */
    tdpMalloc((void **) &halo->target, sizeof(halo_swap_t));
    tdpMemset(halo->target, 0, sizeof(halo_swap_t));

    /* Buffers */
    sz = halo->param->hsz[X]*na*nb*sizeof(double);

    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->fxlo, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);
    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->fxhi, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);

    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->hxlo, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);
    tdpMalloc((void **) & tmp, sz);
    tdpMemcpy(&halo->target->hxhi, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);

    sz = halo->param->hsz[Y]*na*nb*sizeof(double);

    tdpMalloc((void ** ) &tmp, sz);
    tdpMemcpy(&halo->target->fylo, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);
    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->fyhi, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);

    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->hylo, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);
    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->hyhi, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);

    sz = halo->param->hsz[Z]*na*nb*sizeof(double);

    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->fzlo, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);
    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->fzhi, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);

    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->hzlo, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);
    tdpMalloc((void **) &tmp, sz);
    tdpMemcpy(&halo->target->hzhi, &tmp, sizeof(double *),
	      tdpMemcpyHostToDevice);

    tdpGetSymbolAddress((void **) &tmpp, tdpSymbol(const_param));
    tdpMemcpy(&halo->target->param, &tmpp, sizeof(halo_swap_param_t *),
	      tdpMemcpyHostToDevice); 

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

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    double * tmp;

    tdpMemcpy(&tmp, &halo->target->fxlo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->fxhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->fylo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->fyhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->fzlo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->fzhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);

    tdpMemcpy(&tmp, &halo->target->hxlo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->hxhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->hylo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->hyhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->hzlo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);
    tdpMemcpy(&tmp, &halo->target->hzhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpFree(tmp);

    tdpFree(halo->target);
  }

  tdpFreeHost(halo->fxlo);
  tdpFreeHost(halo->fxhi);
  tdpFreeHost(halo->fylo);
  tdpFreeHost(halo->fyhi);
  tdpFreeHost(halo->fzlo);
  tdpFreeHost(halo->fzhi);

  tdpFreeHost(halo->hxlo);
  tdpFreeHost(halo->hxhi);
  tdpFreeHost(halo->hylo);
  tdpFreeHost(halo->hyhi);
  tdpFreeHost(halo->hzlo);
  tdpFreeHost(halo->hzhi);

  tdpStreamDestroy(halo->stream[X]);
  tdpStreamDestroy(halo->stream[Y]);
  tdpStreamDestroy(halo->stream[Z]);

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

  tdpMemcpyToSymbol(tdpSymbol(const_param), halo->param,
		    sizeof(halo_swap_param_t), 0, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  halo_swap_host_rank1
 *
 *****************************************************************************/

__host__ int halo_swap_host_rank1(halo_swap_t * halo, void * mbuf,
				  MPI_Datatype mpidata) {

  int sz;
  int ic, jc, kc;
  int ia, index;
  int nh;
  int ireal, ihalo;
  int icount, nsend;
  int pforw, pback;
  int nlocal[3];
  int mpicartsz[3];

  unsigned char * buf;
  unsigned char * sendforw;
  unsigned char * sendback;
  unsigned char * recvforw;
  unsigned char * recvback;

  MPI_Comm comm;
  MPI_Request req[4];
  MPI_Status status[2];

  const int tagf = 2015;
  const int tagb = 2016;

  halo_swap_param_t * hp;

  assert(halo);
  assert(mbuf);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);

  buf = (unsigned char *) mbuf;

  cs_cart_comm(halo->cs, &comm);
  cs_cartsz(halo->cs, mpicartsz);

  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);

  hp = halo->param;
  cs_nlocal(halo->cs, nlocal);

  /* X-direction */

  nsend = hp->nswap*hp->na*nlocal[Y]*nlocal[Z];
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(halo->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(halo->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(halo->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(halo->pe, "malloc(recvback) failed\n");

  /* Load send buffers */

  icount = 0;

  for (nh = 0; nh < hp->nswap; nh++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < hp->na; ia++) {
	  /* Backward going... */
	  index = cs_index(halo->cs, 1 + nh, jc, kc);
	  ireal = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  /* ...and forward going. */
	  index = cs_index(halo->cs, nlocal[X] - nh, jc, kc);
	  ireal = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (mpicartsz[X] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs_cart_neighb(halo->cs, FORWARD, X);
    pback = cs_cart_neighb(halo->cs, BACKWARD, X);
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait for receives */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < hp->nswap; nh++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < hp->na; ia++) {
	  index = cs_index(halo->cs, nlocal[X] + 1 + nh, jc, kc);
	  ihalo = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  index = cs_index(halo->cs, 0 - nh, jc, kc);
	  ihalo = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  /* Y direction */

  nsend = hp->nswap*hp->na*(nlocal[X] + 2*hp->nswap)*nlocal[Z];
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(halo->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(halo->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(halo->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(halo->pe, "malloc(recvback) failed\n");

  /* Load buffers */

  icount = 0;

  for (nh = 0; nh < hp->nswap; nh++) {
    for (ic = 1 - hp->nswap; ic <= nlocal[X] + hp->nswap; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < hp->na; ia++) {
	  index = cs_index(halo->cs, ic, 1 + nh, kc);
	  ireal = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  index = cs_index(halo->cs, ic, nlocal[Y] - nh, kc);
	  ireal = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (mpicartsz[Y] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs_cart_neighb(halo->cs, FORWARD, Y);
    pback = cs_cart_neighb(halo->cs, BACKWARD, Y);
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait for receives */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < hp->nswap; nh++) {
    for (ic = 1 - hp->nswap; ic <= nlocal[X] + hp->nswap; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < hp->na; ia++) {
	  index = cs_index(halo->cs, ic, 0 - nh, kc);
	  ihalo = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  index = cs_index(halo->cs, ic, nlocal[Y] + 1 + nh, kc);
	  ihalo = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  /* Z direction */

  nsend = hp->nswap*hp->na*(nlocal[X] + 2*hp->nswap)*(nlocal[Y] + 2*hp->nswap);
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(halo->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(halo->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(halo->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(halo->pe, "malloc(recvback) failed\n");

  /* Load */
  /* Some adjustment in the load required for 2d systems (X-Y) */

  icount = 0;

  for (nh = 0; nh < hp->nswap; nh++) {
    for (ic = 1 - hp->nswap; ic <= nlocal[X] + hp->nswap; ic++) {
      for (jc = 1 - hp->nswap; jc <= nlocal[Y] + hp->nswap; jc++) {
	for (ia = 0; ia < hp->na; ia++) {
	  kc = imin(1 + nh, nlocal[Z]);
	  index = cs_index(halo->cs, ic, jc, kc);
	  ireal = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  kc = imax(nlocal[Z] - nh, 1);
	  index = cs_index(halo->cs, ic, jc, kc);
	  ireal = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (mpicartsz[Z] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs_cart_neighb(halo->cs, FORWARD, Z);
    pback = cs_cart_neighb(halo->cs, BACKWARD, Z);
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait before unloading */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < hp->nswap; nh++) {
    for (ic = 1 - hp->nswap; ic <= nlocal[X] + hp->nswap; ic++) {
      for (jc = 1 - hp->nswap; jc <= nlocal[Y] + hp->nswap; jc++) {
	for (ia = 0; ia < hp->na; ia++) {
	  index = cs_index(halo->cs, ic, jc, 0 - nh);
	  ihalo = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  index = cs_index(halo->cs, ic, jc, nlocal[Z] + 1 + nh);
	  ihalo = addr_rank1(hp->naddr, hp->na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  return 0;
}


/*****************************************************************************
 *
 *  halo_swap_packed
 *
 *  Version allowing for host/device copies; data must be packed
 *  and unpacked to/from device via appropriate kernels.
 *
 *  "data" must be a device pointer
 *
 *****************************************************************************/

__host__ int halo_swap_packed(halo_swap_t * halo, double * data) {

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
  int mpicartsz[3];
  dim3 nblk, ntpb;
  double * tmp;

  MPI_Comm comm;
  MPI_Request req_x[4];
  MPI_Request req_y[4];
  MPI_Request req_z[4];
  MPI_Status  status[4];

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;

  assert(halo);

  /* 2D systems require fix... in the meantime...*/
  assert(halo->param->nlocal[Z] >= halo->param->nswap);

  tdpGetDeviceCount(&ndevice);
  halo_swap_commit(halo);

  cs_cart_comm(halo->cs, &comm);
  cs_cartsz(halo->cs, mpicartsz);

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

  if (mpicartsz[X] > 1) {
    ncount = halo->param->hsz[X]*halo->param->nfel;
    MPI_Irecv(halo->hxlo, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,BACKWARD,X), ftagx, comm, req_x);
    MPI_Irecv(halo->hxhi, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,FORWARD,X), btagx, comm, req_x + 1);
  }

  if (mpicartsz[Y] > 1) {
    ncount = halo->param->hsz[Y]*halo->param->nfel;
    MPI_Irecv(halo->hylo, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,BACKWARD,Y), ftagy, comm, req_y);
    MPI_Irecv(halo->hyhi, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,FORWARD,Y), btagy, comm, req_y + 1);
  }

  if (mpicartsz[Z] > 1) {
    ncount = halo->param->hsz[Z]*halo->param->nfel;
    MPI_Irecv(halo->hzlo, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,BACKWARD,Z), ftagz, comm, req_z);
    MPI_Irecv(halo->hzhi, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,FORWARD,Z), btagz, comm, req_z + 1);
  }

  /* pack X edges on accelerator */

  kernel_launch_param(hsz[X], &nblk, &ntpb);
  tdpLaunchKernel(halo->data_pack, nblk, ntpb, 0, halo->stream[X],
		  halo->target, X, data);

  if (ndevice > 0) {
    ncount = hsz[X]*halo->param->nfel;
    tdpMemcpy(&tmp, &halo->target->fxlo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(halo->fxlo, tmp, ncount*sizeof(double),
		   tdpMemcpyDeviceToHost, halo->stream[X]);
    tdpMemcpy(&tmp, &halo->target->fxhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(halo->fxhi, tmp, ncount*sizeof(double),
		   tdpMemcpyDeviceToHost, halo->stream[X]);
  }

  /* pack Y edges on accelerator */

  kernel_launch_param(hsz[Y], &nblk, &ntpb);
  tdpLaunchKernel(halo->data_pack, nblk, ntpb, 0, halo->stream[Y],
		  halo->target, Y, data);

  if (ndevice > 0) {
    ncount = hsz[Y]*halo->param->nfel;
    tdpMemcpy(&tmp, &halo->target->fylo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(halo->fylo, tmp, ncount*sizeof(double),
		   tdpMemcpyDeviceToHost, halo->stream[Y]);
    tdpMemcpy(&tmp, &halo->target->fyhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(halo->fyhi, tmp, ncount*sizeof(double),
		   tdpMemcpyDeviceToHost, halo->stream[Y]);
  }

  /* pack Z edges on accelerator */

  kernel_launch_param(hsz[Z], &nblk, &ntpb);
  tdpLaunchKernel(halo->data_pack, nblk, ntpb, 0, halo->stream[Z],
		  halo->target, Z, data);

  if (ndevice > 0) {
    ncount = hsz[Z]*halo->param->nfel;
    tdpMemcpy(&tmp, &halo->target->fzlo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(halo->fzlo, tmp, ncount*sizeof(double),
		   tdpMemcpyDeviceToHost, halo->stream[Z]);
    tdpMemcpy(&tmp, &halo->target->fzhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(halo->fzhi, tmp, ncount*sizeof(double),
		   tdpMemcpyDeviceToHost, halo->stream[Z]);
  }


  /* Wait for X; copy or MPI recvs; put X halos back on device, and unpack */

  tdpStreamSynchronize(halo->stream[X]);
  ncount = hsz[X]*halo->param->nfel;

  if (mpicartsz[X] == 1) {
    /* note these copies do not alias for ndevice == 1 */
    /* fxhi -> hxlo */
    memcpy(halo->hxlo, halo->fxhi, ncount*sizeof(double));
    tdpMemcpy(&tmp, &halo->target->hxlo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(tmp, halo->fxhi, ncount*sizeof(double),
		    tdpMemcpyHostToDevice, halo->stream[X]);
    /* fxlo -> hxhi */
    memcpy(halo->hxhi, halo->fxlo, ncount*sizeof(double));
    tdpMemcpy(&tmp, &halo->target->hxhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(tmp, halo->fxlo, ncount*sizeof(double),
		    tdpMemcpyHostToDevice, halo->stream[X]);
  }
  else {
    MPI_Isend(halo->fxhi, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,FORWARD,X), ftagx, comm, req_x + 2);
    MPI_Isend(halo->fxlo, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,BACKWARD,X), btagx, comm, req_x + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_x, &mc, status);
      if (mc == 0 && ndevice > 0) {
	tdpMemcpy(&tmp, &halo->target->hxlo, sizeof(double *),
		  tdpMemcpyDeviceToHost);
	tdpMemcpyAsync(tmp, halo->hxlo, ncount*sizeof(double),
		       tdpMemcpyHostToDevice, halo->stream[X]);
      }
      if (mc == 1 && ndevice > 0) {
	tdpMemcpy(&tmp, &halo->target->hxhi, sizeof(double *),
		  tdpMemcpyDeviceToHost);
	tdpMemcpyAsync(tmp, halo->hxhi, ncount*sizeof(double),
		       tdpMemcpyHostToDevice, halo->stream[X]);
      }
    }
  }

  kernel_launch_param(hsz[X], &nblk, &ntpb);
  tdpLaunchKernel(halo->data_unpack, nblk, ntpb, 0, halo->stream[X],
		  halo->target, X, data);

  /* Now wait for Y data to arrive from device */
  /* Fill in 4 corners of Y edge data from X halo */

  tdpStreamSynchronize(halo->stream[Y]);

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

  if (mpicartsz[Y] == 1) {
    /* fyhi -> hylo */
    memcpy(halo->hylo, halo->fyhi, ncount*sizeof(double));
    tdpMemcpy(&tmp, &halo->target->hylo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(tmp, halo->fyhi, ncount*sizeof(double),
		   tdpMemcpyHostToDevice, halo->stream[Y]);
    /* fylo -> hyhi */
    memcpy(halo->hyhi, halo->fylo, ncount*sizeof(double));
    tdpMemcpy(&tmp, &halo->target->hyhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(tmp, halo->fylo,ncount*sizeof(double),
		   tdpMemcpyHostToDevice, halo->stream[Y]);
  }
  else {
    MPI_Isend(halo->fyhi, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs, FORWARD,Y), ftagy, comm, req_y + 2);
    MPI_Isend(halo->fylo, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs, BACKWARD,Y), btagy, comm, req_y + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_y, &mc, status);
      if (mc == 0 && ndevice > 0) {
	tdpMemcpy(&tmp, &halo->target->hylo, sizeof(double *),
		  tdpMemcpyDeviceToHost);
	tdpMemcpyAsync(tmp, halo->hylo, ncount*sizeof(double),
		       tdpMemcpyHostToDevice, halo->stream[Y]);
      }
      if (mc == 1 && ndevice > 0) {
	tdpMemcpy(&tmp, &halo->target->hyhi, sizeof(double *),
		  tdpMemcpyDeviceToHost);
	tdpMemcpyAsync(tmp, halo->hyhi, ncount*sizeof(double),
			tdpMemcpyHostToDevice, halo->stream[Y]);
      }
    }
  }


  kernel_launch_param(hsz[Y], &nblk, &ntpb);
  tdpLaunchKernel(halo->data_unpack, nblk, ntpb, 0, halo->stream[Y],
		  halo->target, Y, data);

  /* Wait for Z data from device */
  /* Fill in 4 corners of Z edge data from X halo  */

  tdpStreamSynchronize(halo->stream[Z]);

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

  if (mpicartsz[Z] == 1) {
    /* fzhi -> hzlo */
    tdpMemcpy(&tmp, &halo->target->hzlo, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(tmp, halo->fzhi, ncount*sizeof(double),
		   tdpMemcpyHostToDevice, halo->stream[Z]);
    /* fzlo -> hzhi */
    tdpMemcpy(&tmp, &halo->target->hzhi, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    tdpMemcpyAsync(tmp, halo->fzlo, ncount*sizeof(double),
		   tdpMemcpyHostToDevice, halo->stream[Z]);
  }
  else {
    MPI_Isend(halo->fzhi, ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,FORWARD,Z), ftagz, comm, req_z + 2);
    MPI_Isend(halo->fzlo,  ncount, MPI_DOUBLE,
	      cs_cart_neighb(halo->cs,BACKWARD,Z), btagz, comm, req_z + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_z, &mc, status);
      if (mc == 0 && ndevice > 0) {
	tdpMemcpy(&tmp, &halo->target->hzlo, sizeof(double *),
		  tdpMemcpyDeviceToHost);
	tdpMemcpyAsync(tmp, halo->hzlo, ncount*sizeof(double),
		       tdpMemcpyHostToDevice, halo->stream[Z]);
      }
      if (mc == 1 && ndevice > 0) {
	tdpMemcpy(&tmp, &halo->target->hzhi, sizeof(double *),
		  tdpMemcpyDeviceToHost);
	tdpMemcpyAsync(tmp, halo->hzhi, ncount*sizeof(double),
		       tdpMemcpyHostToDevice, halo->stream[Z]);
      }
    }
  }

  kernel_launch_param(hsz[Z], &nblk, &ntpb);
  tdpLaunchKernel(halo->data_unpack, nblk, ntpb, 0, halo->stream[Z],
		  halo->target, Z, data);

  tdpStreamSynchronize(halo->stream[X]);
  tdpStreamSynchronize(halo->stream[Y]);
  tdpStreamSynchronize(halo->stream[Z]);

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

  assert(halo);
  assert(id == X || id == Y || id == Z);
  assert(data);

  for_simt_parallel(kindex, halo->param->hsz[id], 1) {

    int nh;
    int hsz;
    int ia, indexl, indexh, ic, jc, kc;
    int hi; /* high end offset */
    double * __restrict__ buflo = NULL;
    double * __restrict__ bufhi = NULL;
    halo_swap_param_t * hp;

    hp = halo->param;
    hsz = halo->param->hsz[id];

    /* Load two buffers for this site */
    /* Use full nhalo to address full data */

    nh = halo->param->nhalo;
    halo_swap_coords(halo, id, kindex, &ic, &jc, &kc);

    indexl = 0;
    indexh = 0;

    if (id == X) {
      hi = nh + hp->nlocal[X] - hp->nswap;
      indexl = halo_swap_index(halo, hp->nhalo + ic, jc, kc);
      indexh = halo_swap_index(halo, hi + ic, jc, kc);
      buflo = halo->fxlo;
      bufhi = halo->fxhi;
    }
    if (id == Y) {
      hi = nh + hp->nlocal[Y] - hp->nswap;
      indexl = halo_swap_index(halo, ic, nh + jc, kc);
      indexh = halo_swap_index(halo, ic, hi + jc, kc);
      buflo = halo->fylo;
      bufhi = halo->fyhi;
    }
    if (id == Z) {
      hi = nh + hp->nlocal[Z] - hp->nswap;
      indexl = halo_swap_index(halo, ic, jc, nh + kc);
      indexh = halo_swap_index(halo, ic, jc, hi + kc);
      buflo = halo->fzlo;
      bufhi = halo->fzhi;
    }

    if (halo->param->nb == 1) {

      /* Rank 1 */

      /* Low end, and high end */

      for (ia = 0; ia < hp->na; ia++) {
	buflo[hsz*ia + kindex] = data[addr_rank1(hp->naddr, hp->na, indexl, ia)];
      }

      for (ia = 0; ia < hp->na; ia++) {
	bufhi[hsz*ia + kindex] = data[addr_rank1(hp->naddr, hp->na, indexh, ia)];
      }
    }
    else {
      int ib, nel;

      nel = 0;
      for (ia = 0; ia < hp->na; ia++) {
	for (ib = 0; ib < hp->nb; ib++) {
	  buflo[hsz*nel + kindex] =
	    data[addr_rank2(hp->naddr, hp->na, hp->nb, indexl, ia, ib)];
	  nel += 1;
	}
      }

      nel = 0;
      for (ia = 0; ia < hp->na; ia++) {
	for (ib = 0; ib < hp->nb; ib++) {
	  bufhi[hsz*nel + kindex] =
	    data[addr_rank2(hp->naddr, hp->na, hp->nb, indexh, ia, ib)];
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

  assert(halo);
  assert(id == X || id == Y || id == Z);
  assert(data);

  /* Unpack buffer this site. */

  for_simt_parallel(kindex, halo->param->hsz[id], 1) {

    int hsz;
    int ia, indexl, indexh;
    int nh;                          /* Full halo width */
    int ic, jc, kc;                  /* Lattice ooords */
    int lo, hi;                      /* Offset for low, high end */
    double * __restrict__ buflo = NULL;
    double * __restrict__ bufhi = NULL;
    halo_swap_param_t * hp;

    hp = halo->param;
    hsz = halo->param->hsz[id];

    nh = halo->param->nhalo;
    halo_swap_coords(halo, id, kindex, &ic, &jc, &kc);

    indexl = 0;
    indexh = 0;

    if (id == X) {
      lo = nh - hp->nswap;
      hi = nh + hp->nlocal[X];
      indexl = halo_swap_index(halo, lo + ic, jc, kc);
      indexh = halo_swap_index(halo, hi + ic, jc, kc);
      buflo = halo->hxlo;
      bufhi = halo->hxhi;
    }

    if (id == Y) {
      lo = nh - hp->nswap;
      hi = nh + hp->nlocal[Y];
      indexl = halo_swap_index(halo, ic, lo + jc, kc);
      indexh = halo_swap_index(halo, ic, hi + jc, kc);
      buflo = halo->hylo;
      bufhi = halo->hyhi;
    }

    if (id == Z) {
      lo = nh - hp->nswap;
      hi = nh + hp->nlocal[Z];
      indexl = halo_swap_index(halo, ic, jc, lo + kc);
      indexh = halo_swap_index(halo, ic, jc, hi + kc);
      buflo = halo->hzlo;
      bufhi = halo->hzhi;
    } 


    if (halo->param->nb == 1) {

      /* Rank 1 */
      /* Low end, then high end */

      for (ia = 0; ia < hp->na; ia++) {
	data[addr_rank1(hp->naddr, hp->na, indexl, ia)] = buflo[hsz*ia + kindex];
      }

      for (ia = 0; ia < hp->na; ia++) {
	data[addr_rank1(hp->naddr, hp->na, indexh, ia)] = bufhi[hsz*ia + kindex];
      }

    }
    else {
      int ib, nel;

      nel = 0;
      for (ia = 0; ia < hp->na; ia++) {
	for (ib = 0; ib < hp->nb; ib++) {
	  data[addr_rank2(hp->naddr, hp->na, hp->nb, indexl, ia, ib)] =
	    buflo[hsz*nel + kindex];
	  nel += 1;
	}
      }

      nel = 0;
      for (ia = 0; ia < hp->na; ia++) {
	for (ib = 0; ib < hp->nb; ib++) {
	  data[addr_rank2(hp->naddr, hp->na, hp->nb, indexh, ia, ib)] =
	    bufhi[hsz*nel + kindex];
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
 *  relative to buffer region for direction id.
 *
 *****************************************************************************/

__host__ __device__
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
 *  A special case of cs_index().
 *
 *****************************************************************************/

__host__ __device__
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

__host__ __device__
int halo_swap_bufindex(halo_swap_t * halo, int id, int ic, int jc, int kc) {

  int xstr;
  int ystr;

  assert(halo);

  ystr = halo->param->hext[id][Z];
  xstr = ystr*halo->param->hext[id][Y];

  return (ic*xstr + jc*ystr + kc);
}
