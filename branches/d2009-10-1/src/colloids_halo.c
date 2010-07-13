/*****************************************************************************
 *
 *  colloids_halo.c
 *
 *  Halo exchange of colloid state information.
 *
 *  $Id: colloids_halo.c,v 1.1.2.3 2010-07-13 18:20:53 kevin Exp $
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

static colloid_state_t * send_;   /* Send buffer */
static colloid_state_t * recv_;   /* Recv buffer */
static const int tag_ = 1065;

static void colloids_halo_load(int dim, int nload[2]);
static void colloids_halo_unload(int nrecv);
static void colloids_halo_number(int dim, int nsend[2], int nrecv[2]);
static void colloids_halo_irecv(int dim, int nrecv[2], MPI_Request req[2]);
static void colloids_halo_isend(int dim, int nsend[2], MPI_Request req[2]);
static void colloids_halo_load_list(int ic, int jc, int kc,
				    const double rperiod[3], int * n);

/*****************************************************************************
 *
 *  colloids_halo_state
 *
 *  A limited choice of periodic conditions is available:
 *    {1, 1, 1}, {1, 1, 0}, {1, 0, 0}, or {0, 0, 0}
 *
 *  These are trapped explicitly in serial here. In parallel,
 *  communications at the boundaries involve MPI_PROC_NULL.
 *
 *****************************************************************************/

void colloids_halo_state(void) {

  if (cart_size(X) == 1 && is_periodic(X) == 0) return;
  colloids_halo_dim(X);

  if (cart_size(Y) == 1 && is_periodic(Y) == 0) return;
  colloids_halo_dim(Y);

  if (cart_size(Z) == 1 && is_periodic(Z) == 0) return;
  colloids_halo_dim(Z);

  return;
}

/*****************************************************************************
 *
 *  colloids_halo_dim
 *
 *  Swap in one dimension 'dim'.
 *
 *****************************************************************************/

void colloids_halo_dim(int dim) {

  int n;
  int nsend[2];
  int nrecv[2];
  int nload[2];

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

  if (send_ == NULL) fatal("malloc(send_) failed\n");
  if (recv_ == NULL) fatal("malloc(recv_) failed\n");

  colloids_halo_irecv(dim, nrecv, request_recv);

  /* Load the send buffer and send */

  colloids_halo_load(dim, nload);

  assert(nload[FORWARD] == nsend[FORWARD]);
  assert(nload[BACKWARD] == nsend[BACKWARD]);

  colloids_halo_isend(dim, nsend, request_send);

  /* Wait for the receives, unload the recv buffer, and finish */

  MPI_Waitall(2, request_recv, status);
  colloids_halo_unload(nrecv[FORWARD] + nrecv[BACKWARD]);
  free(recv_);

  MPI_Waitall(2, request_send, status);
  free(send_);

  return;
}

/*****************************************************************************
 *
 *  colloids_halo_send_count
 *
 *****************************************************************************/

void colloids_halo_send_count(int dim, int nsend[2]) {

  int ic, jc, kc;

  nsend[BACKWARD] = 0;
  nsend[FORWARD] = 0;

  if (dim == X) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
	nsend[BACKWARD] += colloids_cell_count(1, jc, kc);
	nsend[FORWARD] += colloids_cell_count(Ncell(X), jc, kc); 
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
	nsend[BACKWARD] += colloids_cell_count(ic, 1, kc);
	nsend[FORWARD] += colloids_cell_count(ic, Ncell(Y), kc); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
	nsend[BACKWARD] += colloids_cell_count(ic, jc, 1);
	nsend[FORWARD] += colloids_cell_count(ic, jc, Ncell(Z)); 
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_halo_load
 *
 *****************************************************************************/

static void colloids_halo_load(int dim, int nload[2]) {

  int p;
  int ic, jc, kc;
  double rforw[3];
  double rback[3];

  nload[FORWARD] = 0;
  nload[BACKWARD] = 0;

  for (p = 0; p < 3; p++) {
    rback[p] = 0.0;
    rforw[p] = 0.0;
  }

  p = cart_coords(dim);
  if (p == 0) rback[dim] = L(dim);
  if (p == cart_size(dim) - 1) rforw[dim] = -L(dim);

  if (dim == X) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
	colloids_halo_load_list(1, jc, kc, rback, nload + BACKWARD);
	colloids_halo_load_list(Ncell(X), jc, kc, rforw, nload + FORWARD); 
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
	colloids_halo_load_list(ic, 1, kc, rback, nload + BACKWARD);
	colloids_halo_load_list(ic, Ncell(Y), kc, rforw, nload + FORWARD); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
	colloids_halo_load_list(ic, jc, 1, rback, nload + BACKWARD);
	colloids_halo_load_list(ic, jc, Ncell(Z), rforw, nload + FORWARD); 
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_halo_load_list
 *
 *  Insert a particle into the halo message structure. Apply the
 *  periodic boundary conditions to the copy before it leaves.
 *
 *****************************************************************************/

static void colloids_halo_load_list(int ic, int jc, int kc,
				    const double * rperiod, int * n) {
  colloid_t * pc;

  pc = colloids_cell_list(ic, jc, kc);

  while (pc) {
    send_[*n] = pc->s;
    send_[*n].r[X] = pc->s.r[X] + rperiod[X];
    send_[*n].r[Y] = pc->s.r[Y] + rperiod[Y];
    send_[*n].r[Z] = pc->s.r[Z] + rperiod[Z];

    (*n)++;
    pc = pc->next;
  }

  return;
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
	pc->s = recv_[n];
	exists = 1;
      }

      pc = pc->next;
    }

    if (exists == 0) {
      pc = colloid_allocate();
      pc->s = recv_[n];
      colloids_cell_insert_colloid(pc);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_halo_irecv
 *
 *****************************************************************************/

static void colloids_halo_irecv(int dim, int nrecv[2], MPI_Request req[2]) {

  int n;
  int pforw;
  int pback;

  MPI_Comm comm;

  if (cart_size(dim) > 1) {
    comm  = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    n = nrecv[FORWARD]*sizeof(colloid_state_t);
    MPI_Irecv(recv_, n, MPI_BYTE, pforw, tag_, comm, req);

    n = nrecv[BACKWARD]*sizeof(colloid_state_t);
    MPI_Irecv(recv_ + nrecv[FORWARD], n, MPI_BYTE, pback, tag_, comm, req + 1);
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_halo_isend
 *
 *  This 'progresses the message' in serial.
 *
 *****************************************************************************/

static void colloids_halo_isend(int dim, int nsend[2], MPI_Request req[2]) {

  int n;
  int pforw;
  int pback;

  MPI_Comm comm;

  if (cart_size(dim) == 1) {
    n = nsend[FORWARD] + nsend[BACKWARD];
    memcpy(recv_, send_, n*sizeof(colloid_state_t));
  }
  else {

    comm = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    n = nsend[FORWARD]*sizeof(colloid_state_t);
    MPI_Issend(send_ + nsend[BACKWARD], n, MPI_BYTE, pforw, tag_, comm, req);

    n = nsend[BACKWARD]*sizeof(colloid_state_t);
    MPI_Issend(send_, n, MPI_BYTE, pback, tag_, comm, req + 1);
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_halo_number
 *
 *****************************************************************************/

static void colloids_halo_number(int dim, int nsend[2], int nrecv[2]) {

  int       pforw, pback;
  const int tag = 1066;

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

    MPI_Irecv(nrecv + FORWARD, 1, MPI_INT, pforw, tag, comm, request);
    MPI_Irecv(nrecv + BACKWARD, 1, MPI_INT, pback, tag, comm, request + 1);

    MPI_Issend(nsend + FORWARD, 1, MPI_INT, pforw, tag, comm, request + 2);
    MPI_Issend(nsend + BACKWARD, 1, MPI_INT, pback, tag, comm, request + 3);

    MPI_Waitall(4, request, status);
  }

  return;
}
