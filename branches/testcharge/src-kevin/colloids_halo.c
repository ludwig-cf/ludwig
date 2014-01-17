/*****************************************************************************
 *
 *  colloids_halo.c
 *
 *  Halo exchange of colloid state information.
 *
 *  $Id: colloids_halo.c,v 1.4 2010-11-29 17:03:16 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
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

#ifndef OLD_ONLY

struct colloid_halo_s {
  colloids_info_t * cinfo;
  colloid_state_t * send;
  colloid_state_t * recv;
  int nsend[2];
  int nrecv[2];
};

#else

static colloid_state_t * send_;   /* Send buffer */
static colloid_state_t * recv_;   /* Recv buffer */
#endif
static const int tagf_ = 1061;
static const int tagb_ = 1062;

#ifdef OLD_ONLY
static void colloids_halo_load(int dim, const int nsend[2]);
static void colloids_halo_unload(int nrecv);
static void colloids_halo_number(int dim, int nsend[2], int nrecv[2]);
static void colloids_halo_irecv(int dim, int nrecv[2], MPI_Request req[2]);
static void colloids_halo_isend(int dim, int nsend[2], MPI_Request req[2]);
static  int colloids_halo_load_list(int ic, int jc, int kc,
				    const double rperiod[3], int noff);
#else
static int colloids_halo_load(colloid_halo_t * halo, int dim);
static int colloids_halo_unload(colloid_halo_t * halo, int nrecv);
static int colloids_halo_number(colloid_halo_t * halo, int dim);
static int colloids_halo_irecv(colloid_halo_t * halo, int dim, MPI_Request req[2]);
static int colloids_halo_isend(colloid_halo_t * halo, int dim, MPI_Request req[2]);
static int colloids_halo_load_list(colloid_halo_t * halo,
				   int ic, int jc, int kc,
				   const double rperiod[3], int noff);
#endif

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
#ifdef OLD_ONLY
void colloids_halo_state(void) {

  colloids_halo_dim(X);
  colloids_halo_dim(Y);
  colloids_halo_dim(Z);

  return;
}
#else

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
  if (halo == NULL) fatal("calloc(colloid_halo_t) failed\n");

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

int colloids_halo_state(colloids_info_t * cinfo) {

  colloid_halo_t * halo = NULL;

  assert(cinfo);

  halo = (colloid_halo_t *) calloc(1, sizeof(colloid_halo_t));
  if (halo == NULL) fatal("calloc(colloid_halo_t) failed\n");

  halo->cinfo = cinfo;

  colloids_halo_dim(halo, X);
  colloids_halo_dim(halo, Y);
  colloids_halo_dim(halo, Z);

  free(halo);

  return 0;
}
#endif

/*****************************************************************************
 *
 *  colloids_halo_dim
 *
 *  Swap in one dimension 'dim'.
 *
 *****************************************************************************/
#ifdef OLD_ONLY
void colloids_halo_dim(int dim) {

  int n;
  int nsend[2];
  int nrecv[2];

  MPI_Request request_send[2];
  MPI_Request request_recv[2];
  MPI_Status  status[2];

  /* Work out how many are currently in the 'send' region, and
   * communicate the information to work out recv count */

  colloids_halo_send_count(dim, nsend);
  colloids_halo_number(dim, nsend, nrecv);

  /* Allocate the send and recv buffer, and post recvs */

  n = nsend[FORWARD] + nsend[BACKWARD];
  send_ = (colloid_state_t *) malloc(n*sizeof(colloid_state_t));
  n = nrecv[FORWARD] + nrecv[BACKWARD];
  recv_ = (colloid_state_t *) malloc(n*sizeof(colloid_state_t));

  if (send_ == NULL) fatal("colloids halo malloc(send_) failed\n");
  if (recv_ == NULL) fatal("colloids halo malloc(recv_) failed\n");

  colloids_halo_irecv(dim, nrecv, request_recv);

  /* Load the send buffer and send */

  colloids_halo_load(dim, nsend);
  colloids_halo_isend(dim, nsend, request_send);

  /* Wait for the receives, unload the recv buffer, and finish */

  MPI_Waitall(2, request_recv, status);
  colloids_halo_unload(n);
  free(recv_);

  MPI_Waitall(2, request_send, status);
  free(send_);

  return;
}
#else
int colloids_halo_dim(colloid_halo_t * halo, int dim) {

  int n;

  MPI_Request request_send[2];
  MPI_Request request_recv[2];
  MPI_Status  status[2];

  assert(halo);
  assert(halo->cinfo);

  /* Work out how many are currently in the 'send' region, and
   * communicate the information to work out recv count */

  colloids_halo_send_count(halo, dim);
  colloids_halo_number(halo, dim);

  /* Allocate the send and recv buffer, and post recvs */

  n = halo->nsend[FORWARD] + halo->nsend[BACKWARD];
  halo->send = (colloid_state_t *) malloc(n*sizeof(colloid_state_t));
  n = halo->nrecv[FORWARD] + halo->nrecv[BACKWARD];
  halo->recv = (colloid_state_t *) malloc(n*sizeof(colloid_state_t));

  if (halo->send == NULL) fatal("colloids halo malloc(send_) failed\n");
  if (halo->recv == NULL) fatal("colloids halo malloc(recv_) failed\n");

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
#endif
/*****************************************************************************
 *
 *  colloids_halo_send_count
 *
 *****************************************************************************/
#ifdef OLD_ONLY
void colloids_halo_send_count(int dim, int nsend[2]) {

  int ic, jc, kc;
  int ncell[3];

  colloids_cell_ncell(ncell);

  nsend[BACKWARD] = 0;
  nsend[FORWARD] = 0;

  if (dim == X) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	nsend[BACKWARD] += colloids_cell_count(1, jc, kc);
	nsend[FORWARD] += colloids_cell_count(ncell[X], jc, kc); 
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	nsend[BACKWARD] += colloids_cell_count(ic, 1, kc);
	nsend[FORWARD] += colloids_cell_count(ic, ncell[Y], kc); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (jc = 0; jc <= ncell[Y] + 1; jc++) {
	nsend[BACKWARD] += colloids_cell_count(ic, jc, 1);
	nsend[FORWARD] += colloids_cell_count(ic, jc, ncell[Z]); 
      }
    }
  }

  return;
}
#else
int colloids_halo_send_count(colloid_halo_t * halo, int dim) {

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

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloids_halo_load
 *
 *****************************************************************************/
#ifdef OLD_ONLY
static void colloids_halo_load(int dim, const int nsend[2]) {

  int p;
  int ic, jc, kc;
  int noff;         /* Offset in the send buffer to fill */
  int nsent_forw;   /* Counter for particles put into send forw buffer */
  int nsent_back;   /* Counter for particles put into send back buffer */
  int ncell[3];
  double rforw[3];
  double rback[3];

  colloids_cell_ncell(ncell);

  nsent_forw = 0;
  nsent_back = 0;

  for (p = 0; p < 3; p++) {
    rback[p] = 0.0;
    rforw[p] = 0.0;
  }

  /* The factor (1-epsilon) here is to prevent problems associated
   * with a colloid position *exactly* on a cell boundary. */

  p = cart_coords(dim);
  if (p == 0) rback[dim] = (1.0-DBL_EPSILON)*L(dim);
  if (p == cart_size(dim) - 1) rforw[dim] = -L(dim);

  if (dim == X) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	noff = nsent_back;
	nsent_back += colloids_halo_load_list(1, jc, kc, rback, noff);
	noff = nsend[BACKWARD] + nsent_forw;
	nsent_forw += colloids_halo_load_list(ncell[X], jc, kc, rforw, noff);
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	noff = nsent_back;
	nsent_back += colloids_halo_load_list(ic, 1, kc, rback, noff);
	noff = nsend[BACKWARD] + nsent_forw;
	nsent_forw += colloids_halo_load_list(ic, ncell[Y], kc, rforw, noff); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (jc = 0; jc <= ncell[Y] + 1; jc++) {
	noff = nsent_back;
	nsent_back += colloids_halo_load_list(ic, jc, 1, rback, noff);
	noff = nsend[BACKWARD] + nsent_forw;
	nsent_forw += colloids_halo_load_list(ic, jc, ncell[Z], rforw, noff); 
      }
    }
  }

  assert(nsent_forw == nsend[FORWARD]);
  assert(nsent_back == nsend[BACKWARD]);

  return;
}
#else
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

  p = cart_coords(dim);
  if (p == 0) rback[dim] = (1.0-DBL_EPSILON)*L(dim);
  if (p == cart_size(dim) - 1) rforw[dim] = -L(dim);

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
#endif
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
#ifdef OLD_ONLY
static int colloids_halo_load_list(int ic, int jc, int kc,
				    const double * rperiod, int noff) {
  int n;
  colloid_t * pc;

  n = 0;
  pc = colloids_cell_list(ic, jc, kc);

  while (pc) {
    send_[noff + n] = pc->s;
    send_[noff + n].r[X] = pc->s.r[X] + rperiod[X];
    send_[noff + n].r[Y] = pc->s.r[Y] + rperiod[Y];
    send_[noff + n].r[Z] = pc->s.r[Z] + rperiod[Z];
    /* Because delta phi is accumulated across copies at each time step,
     * we must zero the outgoing copy here to avoid overcounting */
    send_[noff + n].deltaphi = 0.0;
    n++;
    pc = pc->next;
  }

  return n;
}
#else
static int colloids_halo_load_list(colloid_halo_t * halo, int ic, int jc, int kc,
				    const double * rperiod, int noff) {
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
#endif
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
#ifdef OLD_ONLY
static void colloids_halo_unload(int nrecv) {

  int n;
  int exists;
  int cell[3];
  colloid_t * pc;

  for (n = 0; n < nrecv; n++) {

    colloids_cell_coords(recv_[n].r, cell);
    exists = 0;
    pc = colloids_cell_list(cell[X], cell[Y], cell[Z]);

    while (pc) {

      if (pc->s.index == recv_[n].index) {
	/* kludge: don't update deltaphi */
	double phi;
	phi = pc->s.deltaphi;
	pc->s = recv_[n];
	pc->s.deltaphi = phi;
	exists = 1;
      }

      pc = pc->next;
    }

    if (exists == 0) {
      pc = colloid_add(recv_[n].index, recv_[n].r);
      assert(pc);
      pc->s = recv_[n];
      pc->s.rebuild = 1;
    }
  }

  return;
}
#else
static int colloids_halo_unload(colloid_halo_t * halo, int nrecv) {

  int n;
  int exists;
  int cell[3];
  colloid_t * pc;

  assert(halo);

  for (n = 0; n < nrecv; n++) {

    colloids_info_cell_coords(halo->cinfo, halo->recv[n].r, cell);
    exists = 0;
    colloids_info_cell_list_head(halo->cinfo, cell[X], cell[Y], cell[Z], &pc);

    while (pc) {

      if (pc->s.index == halo->recv[n].index) {
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
      colloids_info_add(halo->cinfo, halo->recv[n].index, halo->recv[n].r, &pc);
      assert(pc);
      pc->s = halo->recv[n];
      pc->s.rebuild = 1;
    }
  }

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloids_halo_irecv
 *
 *****************************************************************************/
#ifdef OLD_ONLY
static void colloids_halo_irecv(int dim, int nrecv[2], MPI_Request req[2]) {

  int n;
  int pforw;
  int pback;

  MPI_Comm comm;

  req[0] = MPI_REQUEST_NULL;
  req[1] = MPI_REQUEST_NULL;

  if (cart_size(dim) > 1) {
    comm  = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    n = nrecv[FORWARD]*sizeof(colloid_state_t);
    MPI_Irecv(recv_, n, MPI_BYTE, pforw, tagb_, comm, req);

    n = nrecv[BACKWARD]*sizeof(colloid_state_t);
    MPI_Irecv(recv_ + nrecv[FORWARD], n, MPI_BYTE, pback, tagf_, comm, req+1);
  }

  return;
}
#else
static int colloids_halo_irecv(colloid_halo_t * halo, int dim,
			       MPI_Request req[2]) {
  int n;
  int pforw;
  int pback;

  MPI_Comm comm;

  assert(halo);

  req[0] = MPI_REQUEST_NULL;
  req[1] = MPI_REQUEST_NULL;

  if (cart_size(dim) > 1) {
    comm  = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    n = halo->nrecv[FORWARD]*sizeof(colloid_state_t);
    MPI_Irecv(halo->recv, n, MPI_BYTE, pforw, tagb_, comm, req);

    n = halo->nrecv[BACKWARD]*sizeof(colloid_state_t);
    MPI_Irecv(halo->recv + halo->nrecv[FORWARD], n, MPI_BYTE, pback, tagf_,
	      comm, req + 1);
  }

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloids_halo_isend
 *
 *  This 'progresses the message' in serial (for periodic boundaries).
 *
 *****************************************************************************/
#ifdef OLD_ONLY
static void colloids_halo_isend(int dim, int nsend[2], MPI_Request req[2]) {

  int n;
  int pforw;
  int pback;

  MPI_Comm comm;

  if (cart_size(dim) == 1) {

    if (is_periodic(dim)) {
      n = nsend[FORWARD] + nsend[BACKWARD];
      memcpy(recv_, send_, n*sizeof(colloid_state_t));
    }

    req[0] = MPI_REQUEST_NULL;
    req[1] = MPI_REQUEST_NULL;
  }
  else {

    comm = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    n = nsend[FORWARD]*sizeof(colloid_state_t);
    MPI_Issend(send_ + nsend[BACKWARD], n, MPI_BYTE, pforw, tagf_, comm, req);

    n = nsend[BACKWARD]*sizeof(colloid_state_t);
    MPI_Issend(send_, n, MPI_BYTE, pback, tagb_, comm, req + 1);
  }

  return;
}
#else
static int colloids_halo_isend(colloid_halo_t * halo, int dim,
			       MPI_Request req[2]) {
  int n;
  int pforw;
  int pback;

  MPI_Comm comm;

  assert(halo);

  if (cart_size(dim) == 1) {

    if (is_periodic(dim)) {
      n = halo->nsend[FORWARD] + halo->nsend[BACKWARD];
      memcpy(halo->recv, halo->send, n*sizeof(colloid_state_t));
    }

    req[0] = MPI_REQUEST_NULL;
    req[1] = MPI_REQUEST_NULL;
  }
  else {

    comm = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    n = halo->nsend[FORWARD]*sizeof(colloid_state_t);
    MPI_Issend(halo->send + halo->nsend[BACKWARD], n, MPI_BYTE, pforw, tagf_,
	       comm, req);

    n = halo->nsend[BACKWARD]*sizeof(colloid_state_t);
    MPI_Issend(halo->send, n, MPI_BYTE, pback, tagb_, comm, req + 1);
  }

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloids_halo_number
 *
 *****************************************************************************/
#ifdef OLD_ONLY
static void colloids_halo_number(int dim, int nsend[2], int nrecv[2]) {

  int       pforw, pback;

  MPI_Comm    comm;
  MPI_Request request[4];
  MPI_Status  status[4];

  if (cart_size(dim) == 1) {
    nrecv[BACKWARD] = nsend[FORWARD];
    nrecv[FORWARD] = nsend[BACKWARD];
  }
  else {

    comm = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    MPI_Irecv(nrecv + FORWARD, 1, MPI_INT, pforw, tagb_, comm, request);
    MPI_Irecv(nrecv + BACKWARD, 1, MPI_INT, pback, tagf_, comm, request + 1);

    MPI_Issend(nsend + FORWARD, 1, MPI_INT, pforw, tagf_, comm, request + 2);
    MPI_Issend(nsend + BACKWARD, 1, MPI_INT, pback, tagb_, comm, request + 3);

    MPI_Waitall(4, request, status);
  }

  /* Non periodic boundaries receive no particles */

  if (is_periodic(dim) == 0) {
    if (cart_coords(dim) == 0) nrecv[BACKWARD] = 0;
    if (cart_coords(dim) == cart_size(dim) - 1) nrecv[FORWARD] = 0;
  }

  return;
}
#else
static int colloids_halo_number(colloid_halo_t * halo, int dim) {

  int pforw, pback;

  MPI_Comm    comm;
  MPI_Request request[4];
  MPI_Status  status[4];

  assert(halo);

  if (cart_size(dim) == 1) {
    halo->nrecv[BACKWARD] = halo->nsend[FORWARD];
    halo->nrecv[FORWARD] = halo->nsend[BACKWARD];
  }
  else {

    comm = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

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

  if (is_periodic(dim) == 0) {
    if (cart_coords(dim) == 0) halo->nrecv[BACKWARD] = 0;
    if (cart_coords(dim) == cart_size(dim) - 1) halo->nrecv[FORWARD] = 0;
  }

  return 0;
}
#endif
