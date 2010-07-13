/*****************************************************************************
 *
 *  colloid_sums.c
 *
 *  Communication for sums over colloid links.
 *
 *  $Id: colloid_sums.c,v 1.1.2.1 2010-07-13 18:23:00 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "colloid_sums.h"

static double * send_;
static double * recv_;

static void colloid_sums_count(const int dim, int ncount[2]);

/*****************************************************************************
 *
 *  colloid_sums_dim
 *
 *****************************************************************************/

void colloid_sums_sim(const int dim) {

  int ncount[2];

  colloid_sums_count(dim, ncount);

  /* Allocate send and receive buffer, and load */

  n = ncount[BACKWARD] + ncount[FORWARD];
  send_ = (double *) malloc(n*sizeof(double));
  recv_ = (double *) malloc(n*sizeof(double));

  if (send_ == NULL) fatal("malloc(send_) failed\n");
  if (recv_ == NULL) fatal("malloc(recv_) failed\n");

  /* Post receives */

  /* load send buffer with appropriate message type and send */

  /* Wait for receives and unload the sum */

  /* Finish */

  free(recv_);
  free(send_);

  return;
}

/*****************************************************************************
 *
 *  colloid_sums_count
 *
 *****************************************************************************/

static void colloid_sum_count(connst int dim, int ncount[2]) {

  int ic, jc, kc;

  ncount[BACKWARD] = 0;
  ncount[FORWARD] = 0;

  if (dim == X) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
	ncount[BACKWARD] += colloids_cell_count(0, jc, kc);
	ncount[BACKWARD] += colloids_cell_count(1, jc, kc);
	ncount[FORWARD]  += colloids_cell_count(Ncell(X), jc, kc);
	ncount[FORWARD]  += colloids_cell_count(Ncell(X)+1, jc, kc);
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
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

static void colloid_sums_irecv(int dim, int nrecv[2], MPI_Request req[2]) {

  int pforw;
  int pback;

  MPI_Comm comm;

  if (cart_size(dim) > 1) {
    comm  = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    MPI_Irecv(recv_, nrecv[FORWARD], MPI_DOUBLE, pforw, tag_, comm, req);
    MPI_Irecv(recv_ + nrecv[FORWARD], nrecv[BACKWARD], MPI_DOUBLE, pback,
	      tag_, comm, req + 1);
  }

  return;
}
