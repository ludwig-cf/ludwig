/*****************************************************************************
 *
 *  colloid_sums.c
 *
 *  Communication for sums over colloid links.
 *
 *  $Id: colloid_sums.c,v 1.2 2010-10-15 12:40:02 kevin Exp $
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

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "colloid_sums.h"

static void colloid_sums_count(const int dim, int ncount[2]);
static void colloid_sums_irecv(int dim, int n[2], int s, MPI_Request req[2]);
static void colloid_sums_isend(int dim, int n[2], int s, MPI_Request req[2]);
static void colloid_sums_process(int dim, const int ncount[2]);

/*****************************************************************************
 *
 *  Sum / message types
 *
 *  1. Structural components related to links: sumw, cbar, rxcbar;
 *     includes deficits from previous time step: dmass, dphi
 *
 *  2. Dynamic quantities required for implicit update:
 *     external force and torque fex, tex; zero-velocity
 *     force and torque f0, t0; upper triangle of symmetric
 *     drag matrix zeta; active squirmer mass correction mactive.
 *
 *  3. Active squirmer force and torque corrections: fc0, tc0
 *
 *  For each message type, the index is passed as a double
 *  for simplicity. The following keep track of the different
 *  messages...
 *
 *****************************************************************************/

static int colloid_sums_m1(int, int, int, int); /* STRUCTURE */
static int colloid_sums_m2(int, int, int, int); /* DYNAMICS */
static int colloid_sums_m3(int, int, int, int); /* ACTIVE */

static const int msize_[3] = {10, 35, 7};       /* Message sizes (doubles) */

/* The following are used for internal communication */

enum load_unload {MESSAGE_LOAD, MESSAGE_UNLOAD};

static int mtype_;                              /* Current message type */
static int mload_;                              /* Load / unload flag */
static int tagf_ = 1070;                        /* Message tag */
static int tagb_ = 1071;                        /* Message tag */

static double * send_;                          /* Send buffer */
static double * recv_;                          /* Receive buffer */

/*****************************************************************************
 *
 *  colloid_sums_halo
 *
 *  There are a limited number of acceptable non-periodic boundaries.
 *  The order of the calls is therefore important, as X messages must
 *  be complete, etc. These must be trapped explicitly in serial.
 *
 *****************************************************************************/

void colloid_sums_halo(const int mtype) {

  if (cart_size(X) == 1 && is_periodic(X) == 0) return;
  colloid_sums_dim(X, mtype);

  if (cart_size(Y) == 1 && is_periodic(Y) == 0) return;
  colloid_sums_dim(Y, mtype);

  if (cart_size(Z) == 1 && is_periodic(Z) == 0) return;
  colloid_sums_dim(Z, mtype);

  return;
}

/*****************************************************************************
 *
 *  colloid_sums_dim
 *
 *****************************************************************************/

void colloid_sums_dim(const int dim, const int mtype) {

  int n, nsize;
  int ncount[2];
 
  MPI_Request recv_req[2];
  MPI_Request send_req[2];
  MPI_Status  status[2];

  assert(mtype >=0 && mtype < 3);
  mtype_ = mtype;
  nsize = msize_[mtype];

  colloid_sums_count(dim, ncount);

  /* Allocate send and receive buffer */

  n = ncount[BACKWARD] + ncount[FORWARD];
  send_ = (double *) malloc(n*nsize*sizeof(double));
  recv_ = (double *) malloc(n*nsize*sizeof(double));
 
  if (send_ == NULL) fatal("malloc(send_) failed\n");
  if (recv_ == NULL) fatal("malloc(recv_) failed\n");

  /* Post receives */

  colloid_sums_irecv(dim, ncount, nsize, recv_req);

  /* load send buffer with appropriate message type and send */

  mload_ = MESSAGE_LOAD;
  colloid_sums_process(dim, ncount);
  colloid_sums_isend(dim, ncount, nsize, send_req);

  /* Wait for receives and unload the sum */

  MPI_Waitall(2, recv_req, status);
  mload_ = MESSAGE_UNLOAD;
  colloid_sums_process(dim, ncount);
  free(recv_);

  /* Finish */

  MPI_Waitall(2, send_req, status);
  free(send_);

  return;
}

/*****************************************************************************
 *
 *  colloid_sums_count
 *
 *  Note that we need the full extent of the cell list is each
 *  direction perpendicular to the transfer.
 *
 *****************************************************************************/

static void colloid_sums_count(const int dim, int ncount[2]) {

  int ic, jc, kc;

  ncount[BACKWARD] = 0;
  ncount[FORWARD] = 0;

  if (dim == X) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {
	ncount[BACKWARD] += colloids_cell_count(0, jc, kc);
	ncount[BACKWARD] += colloids_cell_count(1, jc, kc);
	ncount[FORWARD]  += colloids_cell_count(Ncell(X), jc, kc);
	ncount[FORWARD]  += colloids_cell_count(Ncell(X)+1, jc, kc);
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {
	ncount[BACKWARD] += colloids_cell_count(ic, 0, kc);
	ncount[BACKWARD] += colloids_cell_count(ic, 1, kc);
	ncount[FORWARD]  += colloids_cell_count(ic, Ncell(Y), kc); 
	ncount[FORWARD]  += colloids_cell_count(ic, Ncell(Y)+1, kc); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
	ncount[BACKWARD] += colloids_cell_count(ic, jc, 0);
	ncount[BACKWARD] += colloids_cell_count(ic, jc, 1);
	ncount[FORWARD]  += colloids_cell_count(ic, jc, Ncell(Z)); 
	ncount[FORWARD]  += colloids_cell_count(ic, jc, Ncell(Z)+1); 
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_sums_irecv
 *
 *****************************************************************************/

static void colloid_sums_irecv(int dim, int ncount[2], int nsize,
			       MPI_Request req[2]) {
  int nf, pforw;
  int nb, pback;

  MPI_Comm comm;

  req[0] = MPI_REQUEST_NULL;
  req[1] = MPI_REQUEST_NULL;

  if (cart_size(dim) > 1) {
    comm  = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    nb = nsize*ncount[BACKWARD];
    nf = nsize*ncount[FORWARD];

    if (nb > 0) MPI_Irecv(recv_ + nf, nb, MPI_DOUBLE, pback, tagf_, comm, req);
    if (nf > 0) MPI_Irecv(recv_, nf, MPI_DOUBLE, pforw, tagb_, comm, req + 1);
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_sums_isend
 *
 *****************************************************************************/

static void colloid_sums_isend(int dim, int ncount[2], int nsize,
			       MPI_Request req[2]) {
  int nf, pforw;
  int nb, pback;

  MPI_Comm comm;

  nf = nsize*ncount[FORWARD];
  nb = nsize*ncount[BACKWARD];

  req[0] = MPI_REQUEST_NULL;
  req[1] = MPI_REQUEST_NULL;

  if (cart_size(dim) == 1) {
    memcpy(recv_, send_, (nf + nb)*sizeof(double));
  }
  else {

    comm = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    if (nb > 0) MPI_Issend(send_, nb, MPI_DOUBLE, pback, tagb_, comm, req);
    if (nf > 0) MPI_Issend(send_ + nb, nf, MPI_DOUBLE, pforw, tagf_,
			   comm, req + 1);
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_sums_process
 *
 *  The send buffer is loaded with all backward going particles, then
 *  all forward going particles.
 *
 *  If it's the unload stage, we must arrange for the particles to be
 *  extracted from the opposite part of the buffer, ie., particles
 *  at the back receive from the forward part of the buffer and
 *  vice-versa.
 *
 *****************************************************************************/

static void colloid_sums_process(int dim, const int ncount[2]) {

  int nb, nf;
  int ic, jc, kc;

  int (* message_loader)(int, int, int, int) = NULL;

  if (mtype_ == COLLOID_SUM_STRUCTURE) message_loader = colloid_sums_m1;
  if (mtype_ == COLLOID_SUM_DYNAMICS) message_loader = colloid_sums_m2;
  if (mtype_ == COLLOID_SUM_ACTIVE) message_loader = colloid_sums_m3;
  assert(message_loader);

  if (mload_ == MESSAGE_LOAD) {
    nb = 0;
    nf = ncount[BACKWARD];
  }
  else {
    nb = ncount[FORWARD];
    nf = 0;
  }

  if (dim == X) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {
	nb += message_loader(0, jc, kc, nb);
	nb += message_loader(1, jc, kc, nb);
	nf += message_loader(Ncell(X), jc, kc, nf);
	nf += message_loader(Ncell(X)+1, jc, kc, nf);
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {
	nb += message_loader(ic, 0, kc, nb);
	nb += message_loader(ic, 1, kc, nb);
	nf += message_loader(ic, Ncell(Y), kc, nf); 
	nf += message_loader(ic, Ncell(Y)+1, kc, nf); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
	nb += message_loader(ic, jc, 0, nb);
	nb += message_loader(ic, jc, 1, nb);
	nf += message_loader(ic, jc, Ncell(Z), nf); 
	nf += message_loader(ic, jc, Ncell(Z)+1, nf); 
      }
    }
  }

  if (mload_ == MESSAGE_LOAD) {
    assert(nb == ncount[BACKWARD]);
    assert(nf == ncount[BACKWARD] + ncount[FORWARD]);
  }
  else {
    assert(nb == ncount[FORWARD] + ncount[BACKWARD]);
    assert(nf == ncount[FORWARD]);
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_sums_m1
 *
 *  'Structure' messages cbar, rxcbar etc
 *
 *  The supplied offset for the start of the message is number of
 *  particles, so must take account of the size of the message.
 *
 *****************************************************************************/

static int colloid_sums_m1(int ic, int jc, int kc, int noff) {

  int n, npart;
  int ia;
  int index;
  colloid_t * pc;

  n = msize_[mtype_]*noff;
  npart = 0;
  pc = colloids_cell_list(ic, jc, kc);

  while (pc) {

    if (mload_ == MESSAGE_LOAD) {
      send_[n++] = 1.0*pc->s.index;
      send_[n++] = pc->sumw;
      for (ia = 0; ia < 3; ia++) {
	send_[n++] = pc->cbar[ia];
	send_[n++] = pc->rxcbar[ia];
      }
      send_[n++] = pc->deltam;
      send_[n++] = pc->s.deltaphi;

      assert(n == (noff + npart + 1)*msize_[mtype_]);
    }
    else {
      /* unload and check incoming index (a fatal error) */
      index = (int) recv_[n++];

      if (index != pc->s.index) {
	fatal("Sum mismatch m1 (%d)\n", index);
      }

      pc->sumw += recv_[n++];
      for (ia = 0; ia < 3; ia++) {
	pc->cbar[ia] += recv_[n++];
	pc->rxcbar[ia] += recv_[n++];
      }
      pc->deltam += recv_[n++];
      pc->s.deltaphi += recv_[n++];
      assert(n == (noff + npart + 1)*msize_[mtype_]);
    }

    npart++;
    pc = pc->next;
  }

  return npart;
}

/*****************************************************************************
 *
 *  colloid_sums_m2
 *
 *  'Dynamics' message f0, t0, fex, tex, etc
 *
 *****************************************************************************/

static int colloid_sums_m2(int ic, int jc, int kc, int noff) {

  int n, npart;
  int ia;
  int index;
  colloid_t * pc;

  n = msize_[mtype_]*noff;
  npart = 0;
  pc = colloids_cell_list(ic, jc, kc);

  while (pc) {

    if (mload_ == MESSAGE_LOAD) {
      send_[n++] = 1.0*pc->s.index;
      send_[n++] = pc->sump;
      for (ia = 0; ia < 3; ia++) {
	send_[n++] = pc->f0[ia];
	send_[n++] = pc->t0[ia];
	send_[n++] = pc->force[ia];
	send_[n++] = pc->torque[ia];
      }
      for (ia = 0; ia < 21; ia++) {
	send_[n++] = pc->zeta[ia];
      }
      assert(n == (noff + npart + 1)*msize_[mtype_]);
    }
    else {
      /* unload and check incoming index (a fatal error) */
      index = (int) recv_[n++];
      if (index != pc->s.index) fatal("Sum mismatch m2 (%d)\n", index);

      pc->sump += recv_[n++];
      for (ia = 0; ia < 3; ia++) {
	pc->f0[ia] += recv_[n++];
	pc->t0[ia] += recv_[n++];
	pc->force[ia] += recv_[n++];
	pc->torque[ia] += recv_[n++];
      }
      for (ia = 0; ia < 21; ia++) {
	pc->zeta[ia] += recv_[n++];
      }
      assert(n == (noff + npart + 1)*msize_[mtype_]);
    }

    npart++;
    pc = pc->next;
  }

  return npart;
}

/*****************************************************************************
 *
 *  colloid_sums_m3
 *
 *  See comments for m1 above.
 *
 *****************************************************************************/

static int colloid_sums_m3(int ic, int jc, int kc, int noff) {

  int n, npart;
  int ia;
  int index;
  colloid_t * pc;

  n = msize_[mtype_]*noff;
  npart = 0;
  pc = colloids_cell_list(ic, jc, kc);

  while (pc) {

    if (mload_ == MESSAGE_LOAD) {
      send_[n++] = 1.0*pc->s.index;
      for (ia = 0; ia < 3; ia++) {
	send_[n++] = pc->fc0[ia];
	send_[n++] = pc->tc0[ia];
      }
      assert(n == (noff + npart + 1)*msize_[mtype_]);
    }
    else {
      /* unload and check incoming index (a fatal error) */
      index = (int) recv_[n++];
      if (index != pc->s.index) fatal("Sum mismatch m2 (%d)\n", index);

      for (ia = 0; ia < 3; ia++) {
	pc->fc0[ia] += recv_[n++];
	pc->tc0[ia] += recv_[n++];
      }
      assert(n == (noff + npart + 1)*msize_[mtype_]);
    }

    npart++;
    pc = pc->next;
  }

  return npart;
}
