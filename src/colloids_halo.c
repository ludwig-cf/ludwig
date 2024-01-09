/*****************************************************************************
 *
 *  colloids_halo.c
 *
 *  Halo exchange of colloid state information.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "colloids_halo.h"
#include "util.h"

struct colloid_halo_s {
  pe_t * pe;               /* Parallel environment */
  cs_t * cs;               /* Coordinate system */
  colloids_info_t * cinfo;
  colloid_state_t * send;
  colloid_state_t * recv;
  int nsend[2];
  int nrecv[2];
};

static const int tagf_ = 1061;
static const int tagb_ = 1062;

static int colloids_halo_load(colloid_halo_t * halo, int dim);
static int colloids_halo_unload(colloid_halo_t * halo, int nrecv);
static int colloids_halo_number(colloid_halo_t * halo, int dim);
static int colloids_halo_irecv(colloid_halo_t * halo, int dim, MPI_Request req[2]);
static int colloids_halo_isend(colloid_halo_t * halo, int dim, MPI_Request req[2]);
static int colloids_halo_load_list(colloid_halo_t * halo,
				   int ic, int jc, int kc,
				   const double rperiod[3], int noff);

/*****************************************************************************
 *
 *  colloid_halo_create
 *
 *  There are two ways to operate the halo swap.
 *  (1) Allocate colloid_halo_t ahead of time and use colloid_halo_dim
 *  (2) Just use colloid_halo_state(), where this is allocated internally.
 *
 *****************************************************************************/

int colloids_halo_create(colloids_info_t * cinfo, colloid_halo_t ** phalo) {

  colloid_halo_t * halo = NULL;

  assert(cinfo);

  halo = (colloid_halo_t *) calloc(1, sizeof(colloid_halo_t));
  assert(halo);
  if (halo == NULL) pe_fatal(cinfo->pe, "calloc(colloid_halo_t) failed\n");

  halo->pe = cinfo->pe;
  halo->cs = cinfo->cs;
  halo->cinfo = cinfo;

  *phalo = halo;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_halo_free
 *
 *****************************************************************************/

void colloids_halo_free(colloid_halo_t * halo) {

  assert(halo);

  free(halo);

  return;
}

/*****************************************************************************
 *
 *  colloids_halo_state
 *
 *  A limited choice of periodic conditions is available:
 *    {1, 1, 1}, {1, 1, 0}, {1, 0, 0}, or {0, 0, 0}
 *
 *  These are trapped explicitly in serial here. In parallel,
 *  communications at the boundaries involve MPI_PROC_NULL and are
 *  trapped later by setting the number of incoming colloids to zero.
 *
 *****************************************************************************/

int colloids_halo_state(colloids_info_t * cinfo) {

  colloid_halo_t * halo = NULL;

  assert(cinfo);

  halo = (colloid_halo_t *) calloc(1, sizeof(colloid_halo_t));
  assert(halo);
  if (halo == NULL) pe_fatal(cinfo->pe, "calloc(colloid_halo_t) failed\n");

  halo->pe = cinfo->pe;
  halo->cs = cinfo->cs;
  halo->cinfo = cinfo;

  colloids_halo_dim(halo, X);
  colloids_halo_dim(halo, Y);
  colloids_halo_dim(halo, Z);

  free(halo);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_halo_dim
 *
 *  Swap in one dimension 'dim'.
 *
 *****************************************************************************/

int colloids_halo_dim(colloid_halo_t * halo, int dim) {

  int n;

  MPI_Request request_send[2];
  MPI_Request request_recv[2];
  MPI_Status  status[2];

  assert(halo);
  assert(halo->cinfo);

  /* Work out how many are currently in the 'send' region, and
   * communicate the information to work out recv count */

  colloids_halo_send_count(halo, dim, NULL);
  colloids_halo_number(halo, dim);

  /* Allocate the send and recv buffer, and post recvs */

  n = halo->nsend[FORWARD] + halo->nsend[BACKWARD];
  halo->send = (colloid_state_t *) malloc(imax(1,n)*sizeof(colloid_state_t));
  assert(halo->send);

  n = halo->nrecv[FORWARD] + halo->nrecv[BACKWARD];
  halo->recv = (colloid_state_t *) malloc(imax(1,n)*sizeof(colloid_state_t));
  assert(halo->recv);

  if (halo->send == NULL) pe_fatal(halo->pe, "halo malloc(send_) failed\n");
  if (halo->recv == NULL) pe_fatal(halo->pe, "halo malloc(recv_) failed\n");

  colloids_halo_irecv(halo, dim, request_recv);

  /* Load the send buffer and send */

  colloids_halo_load(halo, dim);
  colloids_halo_isend(halo, dim, request_send);

  /* Wait for the receives, unload the recv buffer, and finish */

  MPI_Waitall(2, request_recv, status);
  colloids_halo_unload(halo, n);
  free(halo->recv);

  MPI_Waitall(2, request_send, status);
  free(halo->send);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_halo_send_count
 *
 *  The array nreturn is available to return the count to an outside
 *  concern. It may be NULL.
 *
 *****************************************************************************/

int colloids_halo_send_count(colloid_halo_t * halo, int dim, int * nreturn) {

  int ic, jc, kc;
  int nback, nforw;
  int ncell[3];

  assert(halo);
  assert(halo->cinfo);

  colloids_info_ncell(halo->cinfo, ncell);

  halo->nsend[BACKWARD] = 0;
  halo->nsend[FORWARD] = 0;

  if (dim == X) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	colloids_info_cell_count(halo->cinfo,        1, jc, kc, &nback);
	colloids_info_cell_count(halo->cinfo, ncell[X], jc, kc, &nforw);
	halo->nsend[BACKWARD] += nback;
	halo->nsend[FORWARD]  += nforw;
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	colloids_info_cell_count(halo->cinfo, ic,        1, kc, &nback);
	colloids_info_cell_count(halo->cinfo, ic, ncell[Y], kc, &nforw);
	halo->nsend[BACKWARD] += nback;
	halo->nsend[FORWARD]  += nforw;
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (jc = 0; jc <= ncell[Y] + 1; jc++) {
	colloids_info_cell_count(halo->cinfo, ic, jc,        1, &nback);
	colloids_info_cell_count(halo->cinfo, ic, jc, ncell[Z], &nforw);
	halo->nsend[BACKWARD] += nback;
	halo->nsend[FORWARD]  += nforw;
      }
    }
  }

  if (nreturn) {
    nreturn[BACKWARD] = halo->nsend[BACKWARD];
    nreturn[FORWARD]  = halo->nsend[FORWARD];
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_halo_load
 *
 *****************************************************************************/

static int colloids_halo_load(colloid_halo_t * halo, int dim) {

  int p;
  int ic, jc, kc;
  int noff;         /* Offset in the send buffer to fill */
  int nsent_forw;   /* Counter for particles put into send forw buffer */
  int nsent_back;   /* Counter for particles put into send back buffer */
  int ncell[3];
  double rforw[3];
  double rback[3];

  assert(halo);

  colloids_info_ncell(halo->cinfo, ncell);

  nsent_forw = 0;
  nsent_back = 0;

  for (p = 0; p < 3; p++) {
    rback[p] = 0.0;
    rforw[p] = 0.0;
  }

  /* The factor (1-epsilon) here is to prevent problems associated
   * with a colloid position *exactly* on a cell boundary. */

  p = halo->cs->param->mpi_cartcoords[dim];
  if (p == 0) {
    rback[dim] = (1.0-DBL_EPSILON)*halo->cs->param->ntotal[dim];
  }
  if (p == halo->cs->param->mpi_cartsz[dim] - 1) {
    rforw[dim] = -1.0*halo->cs->param->ntotal[dim];
  }

  if (dim == X) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	noff = nsent_back;
	nsent_back += colloids_halo_load_list(halo, 1, jc, kc, rback, noff);
	noff = halo->nsend[BACKWARD] + nsent_forw;
	nsent_forw += colloids_halo_load_list(halo, ncell[X], jc, kc, rforw, noff);
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	noff = nsent_back;
	nsent_back += colloids_halo_load_list(halo, ic, 1, kc, rback, noff);
	noff = halo->nsend[BACKWARD] + nsent_forw;
	nsent_forw += colloids_halo_load_list(halo, ic, ncell[Y], kc, rforw, noff); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (jc = 0; jc <= ncell[Y] + 1; jc++) {
	noff = nsent_back;
	nsent_back += colloids_halo_load_list(halo, ic, jc, 1, rback, noff);
	noff = halo->nsend[BACKWARD] + nsent_forw;
	nsent_forw += colloids_halo_load_list(halo, ic, jc, ncell[Z], rforw, noff); 
      }
    }
  }

  assert(nsent_forw == halo->nsend[FORWARD]);
  assert(nsent_back == halo->nsend[BACKWARD]);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_halo_load_list
 *
 *  Insert a particle into the halo message structure. Apply the
 *  periodic boundary conditions to the copy before it leaves.
 *  noff is the offset in the buffer for this cell list.
 *
 *  Return the number of particles loaded.
 *
 *****************************************************************************/

static int colloids_halo_load_list(colloid_halo_t * halo,
				   int ic, int jc, int kc,
				   const double rperiod[3], int noff) {
  int n;
  colloid_t * pc = NULL;

  assert(halo);

  n = 0;
  colloids_info_cell_list_head(halo->cinfo, ic, jc, kc, &pc);

  while (pc) {
    halo->send[noff + n] = pc->s;
    halo->send[noff + n].r[X] = pc->s.r[X] + rperiod[X];
    halo->send[noff + n].r[Y] = pc->s.r[Y] + rperiod[Y];
    halo->send[noff + n].r[Z] = pc->s.r[Z] + rperiod[Z];
    /* Because delta phi is accumulated across copies at each time step,
     * we must zero the outgoing copy here to avoid overcounting */
    halo->send[noff + n].deltaphi = 0.0;
    n++;
    pc = pc->next;
  }

  return n;
}

/*****************************************************************************
 *
 *  colloids_halo_unload
 *
 *  See where this particle wants to go in the cell list.
 *  Have a look at the existing particles in the list location:
 *  if it's already present, make a copy of the state; if it's new,
 *  add the incoming particle to the list.
 *
 *****************************************************************************/

static int colloids_halo_unload(colloid_halo_t * halo, int nrecv) {

  int n;
  int exists;
  int index;
  int cell[3];
  colloid_t * pc = NULL;

  assert(halo);

  for (n = 0; n < nrecv; n++) {

    exists = 0;
    index = halo->recv[n].index;
    colloids_info_cell_coords(halo->cinfo, halo->recv[n].r, cell);
    colloids_info_cell_list_head(halo->cinfo, cell[X], cell[Y], cell[Z], &pc);

    while (pc) {

      if (pc->s.index == index) {
	/* kludge: don't update deltaphi */
	double phi;
	phi = pc->s.deltaphi;
	pc->s = halo->recv[n];
	pc->s.deltaphi = phi;
	exists = 1;
      }

      pc = pc->next;
    }

    if (exists == 0) {
      colloids_info_add(halo->cinfo, index, halo->recv[n].r, &pc);
      assert(pc);
      pc->s = halo->recv[n];
      pc->s.rebuild = 1;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_halo_irecv
 *
 *****************************************************************************/

static int colloids_halo_irecv(colloid_halo_t * halo, int dim,
			       MPI_Request req[2]) {
  int n;
  int pforw;
  int pback;

  MPI_Comm comm;

  assert(halo);

  req[0] = MPI_REQUEST_NULL;
  req[1] = MPI_REQUEST_NULL;

  if (halo->cs->param->mpi_cartsz[dim] > 1) {
    comm  = halo->cs->commcart;
    pforw = halo->cs->mpi_cart_neighbours[CS_FORW][dim];
    pback = halo->cs->mpi_cart_neighbours[CS_BACK][dim];

    n = halo->nrecv[CS_FORW]*sizeof(colloid_state_t);
    MPI_Irecv(halo->recv, n, MPI_BYTE, pforw, tagb_, comm, req);

    n = halo->nrecv[CS_BACK]*sizeof(colloid_state_t);
    MPI_Irecv(halo->recv + halo->nrecv[CS_FORW], n, MPI_BYTE, pback, tagf_,
	      comm, req + 1);
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_halo_isend
 *
 *  This 'progresses the message' in serial (for periodic boundaries).
 *
 *****************************************************************************/

static int colloids_halo_isend(colloid_halo_t * halo, int dim,
			       MPI_Request req[2]) {
  int n;
  int pforw;
  int pback;

  MPI_Comm comm;

  assert(halo);

  if (halo->cs->param->mpi_cartsz[dim] == 1) {

    if (halo->cs->param->periodic[dim]) {
      n = halo->nsend[CS_FORW] + halo->nsend[CS_BACK];
      memcpy(halo->recv, halo->send, n*sizeof(colloid_state_t));
    }

    req[0] = MPI_REQUEST_NULL;
    req[1] = MPI_REQUEST_NULL;
  }
  else {

    comm = halo->cs->commcart;
    pforw = halo->cs->mpi_cart_neighbours[CS_FORW][dim];
    pback = halo->cs->mpi_cart_neighbours[CS_BACK][dim];

    n = halo->nsend[CS_FORW]*sizeof(colloid_state_t);
    MPI_Issend(halo->send + halo->nsend[CS_BACK], n, MPI_BYTE, pforw, tagf_,
	       comm, req);

    n = halo->nsend[CS_BACK]*sizeof(colloid_state_t);
    MPI_Issend(halo->send, n, MPI_BYTE, pback, tagb_, comm, req + 1);
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_halo_number
 *
 *****************************************************************************/

static int colloids_halo_number(colloid_halo_t * halo, int dim) {

  int pforw, pback;

  MPI_Comm    comm;
  MPI_Request request[4];
  MPI_Status  status[4];

  assert(halo);

  if (halo->cs->param->mpi_cartsz[dim] == 1) {
    halo->nrecv[CS_BACK] = halo->nsend[CS_FORW];
    halo->nrecv[CS_FORW] = halo->nsend[CS_BACK];
  }
  else {

    comm = halo->cs->commcart;
    pforw = halo->cs->mpi_cart_neighbours[CS_FORW][dim];
    pback = halo->cs->mpi_cart_neighbours[CS_BACK][dim];

    MPI_Irecv(halo->nrecv + FORWARD, 1, MPI_INT, pforw, tagb_, comm,
	      request);
    MPI_Irecv(halo->nrecv + BACKWARD, 1, MPI_INT, pback, tagf_, comm,
	      request + 1);
    MPI_Issend(halo->nsend + FORWARD, 1, MPI_INT, pforw, tagf_, comm,
	       request + 2);
    MPI_Issend(halo->nsend + BACKWARD, 1, MPI_INT, pback, tagb_, comm,
	       request + 3);

    MPI_Waitall(4, request, status);
  }

  /* Non periodic boundaries receive no particles */

  if (halo->cs->param->periodic[dim] == 0) {
    if (halo->cs->param->mpi_cartcoords[dim] == 0) halo->nrecv[CS_BACK] = 0;
    if (halo->cs->param->mpi_cartcoords[dim]
	== halo->cs->param->mpi_cartsz[dim] - 1) halo->nrecv[CS_FORW] = 0;
  }

  return 0;
}
