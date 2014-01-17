/*****************************************************************************
 *
 *  colloid_sums.c
 *
 *  Communication for sums over colloid links.
 *
 *  $Id$
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

#ifdef OLD_ONLY
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
 *     Also used for subgrid total force: fc0
 *
 *  4. Used to work out order parameter / charge correction
 *     for conserved quantities to be replaced after colloid
 *     movement.
 *
 *  For each message type, the index is passed as a double
 *  for simplicity. The following keep track of the different
 *  messages...
 *
 *****************************************************************************/

static int colloid_sums_m0(int, int, int, int); /* DUMMY */
static int colloid_sums_m1(int, int, int, int); /* STRUCTURE */
static int colloid_sums_m2(int, int, int, int); /* DYNAMICS */
static int colloid_sums_m3(int, int, int, int); /* ACTIVE and SUBGRID */
static int colloid_sums_m4(int, int, int, int); /* CONSERVATION */

/* Message sizes (doubles) */

enum message_type_enum {MTYPE_MAX = 4};
static const int msize_[MTYPE_MAX] = {10, 35, 7, 6};

/* The following are used for internal communication */

enum load_unload {MESSAGE_LOAD, MESSAGE_UNLOAD};

static int mtype_;                              /* Current message type */
static int mload_;                              /* Load / unload flag */
static int tagf_ = 1070;                        /* Message tag */
static int tagb_ = 1071;                        /* Message tag */

static double * send_;                          /* Send buffer */
static double * recv_;                          /* Receive buffer */
#else

struct colloid_sum_s {
  colloids_info_t * cinfo;                /* Temporary reference */
  int mtype;                              /* Current message type */
  int mload;                              /* Load / unload flag */
  int msize;                              /* Current message size */
  int ncount[2];                          /* forward / backward */
  double * send;                          /* Send buffer */
  double * recv;                          /* Receive buffer */
};

static int colloid_sums_count(colloid_sum_t * sum, const int dim);
static int colloid_sums_irecv(colloid_sum_t * sum, int dim, MPI_Request rq[2]);
static int colloid_sums_isend(colloid_sum_t * sum, int dim, MPI_Request rq[2]);
static int colloid_sums_process(colloid_sum_t * sum, int dim);

static int colloid_sums_m0(colloid_sum_t * sum, int, int, int, int);
static int colloid_sums_m1(colloid_sum_t * sum, int, int, int, int);
static int colloid_sums_m2(colloid_sum_t * sum, int, int, int, int);
static int colloid_sums_m3(colloid_sum_t * sum, int, int, int, int);
static int colloid_sums_m4(colloid_sum_t * sum, int, int, int, int);

/* Message sizes (doubles) */

enum message_type_enum {MTYPE_MAX = 4};
static const int msize_[MTYPE_MAX] = {10, 35, 7, 6};

/* The following are used for internal communication */

enum load_unload {MESSAGE_LOAD, MESSAGE_UNLOAD};
static int tagf_ = 1070;                        /* Message tag */
static int tagb_ = 1071;                        /* Message tag */

/*****************************************************************************
 *
 *  colloid_sum_create
 *
 *****************************************************************************/

int colloid_sums_create(colloids_info_t * cinfo, colloid_sum_t ** psum) {

  colloid_sum_t * sum = NULL;

  assert(cinfo);

  sum = (colloid_sum_t *) calloc(1, sizeof(colloid_sum_t));
  if (sum == NULL) fatal("calloc(colloid_sum_t) failed\n");

  sum->cinfo = cinfo;
  *psum = sum;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_sum_free
 *
 *****************************************************************************/

void colloid_sums_free(colloid_sum_t * sum) {

  assert(sum);

  free(sum);

  return;
}

#endif

/*****************************************************************************
 *
 *  colloid_sums_halo
 *
 *  There are a limited number of acceptable non-periodic boundaries.
 *  The order of the calls is therefore important, as X messages must
 *  be complete, etc. These must be trapped explicitly in serial.
 *
 *****************************************************************************/
#ifdef OLD_ONLY
void colloid_sums_halo(const int mtype) {

  colloid_sums_dim(X, mtype);
  colloid_sums_dim(Y, mtype);
  colloid_sums_dim(Z, mtype);

  return;
}
#else
int colloid_sums_halo(colloids_info_t * cinfo, colloid_sum_enum_t mtype) {

  colloid_sum_t * sum = NULL;

  assert(cinfo);

  sum = (colloid_sum_t * ) calloc(1, sizeof(colloid_sum_t));
  if (sum == NULL) fatal("calloc(colloid_sum_t) failed\n");
  sum->cinfo = cinfo;
  sum->mtype = mtype;
  sum->msize = msize_[mtype];

  colloid_sums_1d(sum, X, mtype);
  colloid_sums_1d(sum, Y, mtype);
  colloid_sums_1d(sum, Z, mtype);

  free(sum);

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloid_sums_dim
 *
 *****************************************************************************/
#ifdef OLD_ONLY
void colloid_sums_dim(const int dim, const int mtype) {

  int n, nsize;
  int ncount[2];
 
  MPI_Request recv_req[2];
  MPI_Request send_req[2];
  MPI_Status  status[2];

  assert(mtype >=0 && mtype < MTYPE_MAX);
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
#else
int colloid_sums_1d(colloid_sum_t * sum, int dim, colloid_sum_enum_t mtype) {

  int n;
 
  MPI_Request recv_req[2];
  MPI_Request send_req[2];
  MPI_Status  status[2];

  assert(sum);
  assert(sum->cinfo);
  assert(mtype >=0 && mtype < MTYPE_MAX);

  /* Count how many colloids are relevant */

  sum->mtype = mtype;
  sum->msize = msize_[mtype];
  colloid_sums_count(sum, dim);

  /* Allocate send and receive buffer */

  n = sum->ncount[BACKWARD] + sum->ncount[FORWARD];

  sum->send = (double *) malloc(n*msize_[mtype]*sizeof(double));
  sum->recv = (double *) malloc(n*msize_[mtype]*sizeof(double));
 
  if (sum->send == NULL) fatal("malloc(sum->send) failed\n");
  if (sum->recv == NULL) fatal("malloc(sum->recv) failed\n");

  /* Post receives */

  colloid_sums_irecv(sum, dim, recv_req);

  /* load send buffer with appropriate message type and send */

  sum->mload = MESSAGE_LOAD;
  colloid_sums_process(sum, dim);
  colloid_sums_isend(sum, dim, send_req);

  /* Wait for receives and unload the sum */

  MPI_Waitall(2, recv_req, status);
  sum->mload = MESSAGE_UNLOAD;
  colloid_sums_process(sum, dim);
  free(sum->recv);

  /* Finish */

  MPI_Waitall(2, send_req, status);
  free(sum->send);

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloid_sums_count
 *
 *  Note that we need the full extent of the cell list is each
 *  direction perpendicular to the transfer.
 *
 *****************************************************************************/
#ifdef OLD_ONLY
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

  if (is_periodic(dim) == 0) {
    if (cart_coords(dim) == 0) ncount[BACKWARD] = 0;
    if (cart_coords(dim) == cart_size(dim) - 1) ncount[FORWARD] = 0;
  }

  return;
}
#else
static int colloid_sums_count(colloid_sum_t * sum, const int dim) {

  int ic, jc, kc;
  int n0, n1;
  int ncell[3];

  sum->ncount[BACKWARD] = 0;
  sum->ncount[FORWARD] = 0;

  colloids_info_ncell(sum->cinfo, ncell);

  if (dim == X) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {
	colloids_info_cell_count(sum->cinfo, 0, jc, kc, &n0);
	colloids_info_cell_count(sum->cinfo, 1, jc, kc, &n1);
	sum->ncount[BACKWARD] += (n0 + n1);
	colloids_info_cell_count(sum->cinfo, ncell[X],     jc, kc, &n0);
	colloids_info_cell_count(sum->cinfo, ncell[X] + 1, jc, kc, &n1);
	sum->ncount[FORWARD] += (n0 + n1);
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {
	colloids_info_cell_count(sum->cinfo, ic, 0, kc, &n0);
	colloids_info_cell_count(sum->cinfo, ic, 1, kc, &n1);
	sum->ncount[BACKWARD] += (n0 + n1);
	colloids_info_cell_count(sum->cinfo, ic, ncell[Y],     kc, &n0);
	colloids_info_cell_count(sum->cinfo, ic, ncell[Y] + 1, kc, &n1);
	sum->ncount[FORWARD] += (n0 + n1);
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (jc = 0; jc <= ncell[Y] + 1; jc++) {
	colloids_info_cell_count(sum->cinfo, ic, jc, 0, &n0);
	colloids_info_cell_count(sum->cinfo, ic, jc, 1, &n1);
	sum->ncount[BACKWARD] += (n0 + n1);
	colloids_info_cell_count(sum->cinfo, ic, jc, ncell[Z],     &n0);
	colloids_info_cell_count(sum->cinfo, ic, jc, ncell[Z] + 1, &n1);
	sum->ncount[FORWARD] += (n0 + n1);
      }
    }
  }

  if (is_periodic(dim) == 0) {
    if (cart_coords(dim) == 0) sum->ncount[BACKWARD] = 0;
    if (cart_coords(dim) == cart_size(dim) - 1) sum->ncount[FORWARD] = 0;
  }

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloid_sums_irecv
 *
 *****************************************************************************/
#ifdef OLD_ONLY
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
#else
static int colloid_sums_irecv(colloid_sum_t * sum, int dim,
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

    nb = sum->msize*sum->ncount[BACKWARD];
    nf = sum->msize*sum->ncount[FORWARD];

    if (nb > 0) MPI_Irecv(sum->recv + nf, nb, MPI_DOUBLE, pback, tagf_, comm,
			  req);
    if (nf > 0) MPI_Irecv(sum->recv, nf, MPI_DOUBLE, pforw, tagb_, comm,
			  req + 1);
  }

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloid_sums_isend
 *
 *****************************************************************************/
#ifdef OLD_ONLY
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
#else
static int colloid_sums_isend(colloid_sum_t * sum, int dim,
			       MPI_Request req[2]) {
  int nf, pforw;
  int nb, pback;

  MPI_Comm comm;

  nf = sum->msize*sum->ncount[FORWARD];
  nb = sum->msize*sum->ncount[BACKWARD];

  req[0] = MPI_REQUEST_NULL;
  req[1] = MPI_REQUEST_NULL;

  if (cart_size(dim) == 1) {
    memcpy(sum->recv, sum->send, (nf + nb)*sizeof(double));
  }
  else {

    comm = cart_comm();
    pforw = cart_neighb(FORWARD, dim);
    pback = cart_neighb(BACKWARD, dim);

    if (nb > 0) MPI_Issend(sum->send, nb, MPI_DOUBLE, pback, tagb_, comm, req);
    if (nf > 0) MPI_Issend(sum->send + nb, nf, MPI_DOUBLE, pforw, tagf_,
			   comm, req + 1);
  }

  return 0;
}
#endif
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
#ifdef OLD_ONLY
static void colloid_sums_process(int dim, const int ncount[2]) {

  int nb, nf;
  int ic, jc, kc;

  int (* message_loaderf)(int, int, int, int) = NULL;
  int (* message_loaderb)(int, int, int, int) = NULL;

  if (mtype_ == COLLOID_SUM_STRUCTURE) message_loaderf = colloid_sums_m1;
  if (mtype_ == COLLOID_SUM_DYNAMICS) message_loaderf = colloid_sums_m2;
  if (mtype_ == COLLOID_SUM_ACTIVE) message_loaderf = colloid_sums_m3;
  if (mtype_ == COLLOID_SUM_CONSERVATION) message_loaderf = colloid_sums_m4;

  assert(message_loaderf);
  message_loaderb = message_loaderf;

  if (mload_ == MESSAGE_LOAD) {
    nb = 0;
    nf = ncount[BACKWARD];
  }
  else {
    nb = ncount[FORWARD];
    nf = 0;
  }

  /* Eliminate messages at non-perioidic boundaries via dummy loader m0 */

  if (is_periodic(dim) == 0) {
    ic = cart_coords(dim);
    if (ic == 0) message_loaderb = colloid_sums_m0;
    if (ic == cart_size(dim) - 1) message_loaderf = colloid_sums_m0;
  }

  if (dim == X) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {
	nb += message_loaderb(0, jc, kc, nb);
	nb += message_loaderb(1, jc, kc, nb);
	nf += message_loaderf(Ncell(X), jc, kc, nf);
	nf += message_loaderf(Ncell(X)+1, jc, kc, nf);
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {
	nb += message_loaderb(ic, 0, kc, nb);
	nb += message_loaderb(ic, 1, kc, nb);
	nf += message_loaderf(ic, Ncell(Y), kc, nf); 
	nf += message_loaderf(ic, Ncell(Y)+1, kc, nf); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= Ncell(X) + 1; ic++) {
      for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
	nb += message_loaderb(ic, jc, 0, nb);
	nb += message_loaderb(ic, jc, 1, nb);
	nf += message_loaderf(ic, jc, Ncell(Z), nf); 
	nf += message_loaderf(ic, jc, Ncell(Z)+1, nf); 
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
#else
static int colloid_sums_process(colloid_sum_t * sum, int dim) {

  int nb, nf;
  int ic, jc, kc;
  int ncell[3];

  int (* mloader_forw)(colloid_sum_t *, int, int, int, int) = NULL;
  int (* mloader_back)(colloid_sum_t *, int, int, int, int) = NULL;

  colloids_info_ncell(sum->cinfo, ncell);

  if (sum->mtype == COLLOID_SUM_STRUCTURE) mloader_forw = colloid_sums_m1;
  if (sum->mtype == COLLOID_SUM_DYNAMICS) mloader_forw = colloid_sums_m2;
  if (sum->mtype == COLLOID_SUM_ACTIVE) mloader_forw = colloid_sums_m3;
  if (sum->mtype == COLLOID_SUM_CONSERVATION) mloader_forw = colloid_sums_m4;

  assert(mloader_forw);
  mloader_back = mloader_forw;

  if (sum->mload == MESSAGE_LOAD) {
    nb = 0;
    nf = sum->ncount[BACKWARD];
  }
  else {
    nb = sum->ncount[FORWARD];
    nf = 0;
  }

  /* Eliminate messages at non-perioidic boundaries via dummy loader m0.
   * This is where loader_forw and loader_back can differ. */

  if (is_periodic(dim) == 0) {
    ic = cart_coords(dim);
    if (ic == 0) mloader_back = colloid_sums_m0;
    if (ic == cart_size(dim) - 1) mloader_forw = colloid_sums_m0;
  }

  if (dim == X) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {
	nb += mloader_back(sum, 0, jc, kc, nb);
	nb += mloader_back(sum, 1, jc, kc, nb);
	nf += mloader_forw(sum, ncell[X], jc, kc, nf);
	nf += mloader_forw(sum, ncell[X] + 1, jc, kc, nf);
      }
    }
  }

  if (dim == Y) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {
	nb += mloader_back(sum, ic, 0, kc, nb);
	nb += mloader_back(sum, ic, 1, kc, nb);
	nf += mloader_forw(sum, ic, ncell[Y], kc, nf); 
	nf += mloader_forw(sum, ic, ncell[Y] + 1, kc, nf); 
      }
    }
  }

  if (dim == Z) {
    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (jc = 0; jc <= ncell[Y] + 1; jc++) {
	nb += mloader_back(sum, ic, jc, 0, nb);
	nb += mloader_back(sum, ic, jc, 1, nb);
	nf += mloader_forw(sum, ic, jc, ncell[Z], nf); 
	nf += mloader_forw(sum, ic, jc, ncell[Z] + 1, nf); 
      }
    }
  }

  if (sum->mload == MESSAGE_LOAD) {
    assert(nb == sum->ncount[BACKWARD]);
    assert(nf == sum->ncount[BACKWARD] + sum->ncount[FORWARD]);
  }
  else {
    assert(nb == sum->ncount[FORWARD] + sum->ncount[BACKWARD]);
    assert(nf == sum->ncount[FORWARD]);
  }

  return 0;
}
#endif
/*****************************************************************************
 *
 *  colloid_sums_m0
 *
 *  Null message for non-perioidic boundaries.
 *
 *****************************************************************************/
#ifdef OLD_ONLY
static int colloid_sums_m0(int ic, int jc, int kc, int noff) {

  return 0;
}
#else
static int colloid_sums_m0(colloid_sum_t * sum, int ic, int jc, int kc,
			   int noff) {

  return 0;
}
#endif
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
#ifdef OLD_ONLY
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
#else
static int colloid_sums_m1(colloid_sum_t * sum, int ic, int jc, int kc,
			   int noff) {

  int n, npart;
  int ia;
  int index;
  colloid_t * pc;

  n = sum->msize*noff;
  npart = 0;
  colloids_info_cell_list_head(sum->cinfo, ic, jc, kc, &pc);

  while (pc) {

    if (sum->mload == MESSAGE_LOAD) {
      sum->send[n++] = 1.0*pc->s.index;
      sum->send[n++] = pc->sumw;
      for (ia = 0; ia < 3; ia++) {
	sum->send[n++] = pc->cbar[ia];
	sum->send[n++] = pc->rxcbar[ia];
      }
      sum->send[n++] = pc->deltam;
      sum->send[n++] = pc->s.deltaphi;

      assert(n == (noff + npart + 1)*sum->msize);
    }
    else {
      /* unload and check incoming index (a fatal error) */
      index = (int) sum->recv[n++];

      if (index != pc->s.index) {
	fatal("Sum mismatch m1 (%d)\n", index);
      }

      pc->sumw += sum->recv[n++];
      for (ia = 0; ia < 3; ia++) {
	pc->cbar[ia] += sum->recv[n++];
	pc->rxcbar[ia] += sum->recv[n++];
      }
      pc->deltam += sum->recv[n++];
      pc->s.deltaphi += sum->recv[n++];
      assert(n == (noff + npart + 1)*sum->msize);
    }

    npart++;
    pc = pc->next;
  }

  return npart;
}
#endif
/*****************************************************************************
 *
 *  colloid_sums_m2
 *
 *  'Dynamics' message f0, t0, fex, tex, etc
 *
 *****************************************************************************/
#ifdef OLD_ONLY
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
#else
static int colloid_sums_m2(colloid_sum_t * sum, int ic, int jc, int kc,
			   int noff) {

  int n, npart;
  int ia;
  int index;
  colloid_t * pc = NULL;

  n = sum->msize*noff;
  npart = 0;
  colloids_info_cell_list_head(sum->cinfo, ic, jc, kc, &pc);

  while (pc) {

    if (sum->mload == MESSAGE_LOAD) {
      sum->send[n++] = 1.0*pc->s.index;
      sum->send[n++] = pc->sump;
      for (ia = 0; ia < 3; ia++) {
	sum->send[n++] = pc->f0[ia];
	sum->send[n++] = pc->t0[ia];
	sum->send[n++] = pc->force[ia];
	sum->send[n++] = pc->torque[ia];
      }
      for (ia = 0; ia < 21; ia++) {
	sum->send[n++] = pc->zeta[ia];
      }
      assert(n == (noff + npart + 1)*sum->msize);
    }
    else {
      /* unload and check incoming index (a fatal error) */
      index = (int) sum->recv[n++];
      if (index != pc->s.index) fatal("Sum mismatch m2 (%d)\n", index);

      pc->sump += sum->recv[n++];
      for (ia = 0; ia < 3; ia++) {
	pc->f0[ia] += sum->recv[n++];
	pc->t0[ia] += sum->recv[n++];
	pc->force[ia] += sum->recv[n++];
	pc->torque[ia] += sum->recv[n++];
      }
      for (ia = 0; ia < 21; ia++) {
	pc->zeta[ia] += sum->recv[n++];
      }
      assert(n == (noff + npart + 1)*sum->msize);
    }

    npart++;
    pc = pc->next;
  }

  return npart;
}
#endif
/*****************************************************************************
 *
 *  colloid_sums_m3
 *
 *  See comments for m1 above.
 *
 *****************************************************************************/
#ifdef OLD_ONLY
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
#else
static int colloid_sums_m3(colloid_sum_t * sum, int ic, int jc, int kc,
			   int noff) {

  int n, npart;
  int ia;
  int index;
  colloid_t * pc;

  n = sum->msize*noff;
  npart = 0;
  colloids_info_cell_list_head(sum->cinfo, ic, jc, kc, &pc);

  while (pc) {

    if (sum->mload == MESSAGE_LOAD) {
      sum->send[n++] = 1.0*pc->s.index;
      for (ia = 0; ia < 3; ia++) {
	sum->send[n++] = pc->fc0[ia];
	sum->send[n++] = pc->tc0[ia];
      }
      assert(n == (noff + npart + 1)*sum->msize);
    }
    else {
      /* unload and check incoming index (a fatal error) */
      index = (int) sum->recv[n++];
      if (index != pc->s.index) fatal("Sum mismatch m2 (%d)\n", index);

      for (ia = 0; ia < 3; ia++) {
	pc->fc0[ia] += sum->recv[n++];
	pc->tc0[ia] += sum->recv[n++];
      }
      assert(n == (noff + npart + 1)*sum->msize);
    }

    npart++;
    pc = pc->next;
  }

  return npart;
}
#endif
/*****************************************************************************
 *
 *  colloid_sums_m4
 *
 *  See comments for m1 above.
 *
 *  This is for conserved order parameters and related information.
 *
 *****************************************************************************/
#ifdef OLD_ONLY
static int colloid_sums_m4(int ic, int jc, int kc, int noff) {

  int n, npart;
  int index;
  colloid_t * pc;

  n = msize_[mtype_]*noff;
  npart = 0;
  pc = colloids_cell_list(ic, jc, kc);

  while (pc) {

    if (mload_ == MESSAGE_LOAD) {
      send_[n++] = 1.0*pc->s.index;
      send_[n++] = pc->s.deltaphi;
      send_[n++] = pc->s.deltaq0;
      send_[n++] = pc->s.deltaq1;
      send_[n++] = pc->s.sa;
      send_[n++] = pc->s.saf;

      assert(n == (noff + npart + 1)*msize_[mtype_]);
    }
    else {

      /* unload and check incoming index (a fatal error) */
      index = (int) recv_[n++];
      if (index != pc->s.index) fatal("Sum mismatch m4 (%d)\n", index);

      pc->s.deltaphi += recv_[n++];
      pc->s.deltaq0  += recv_[n++];
      pc->s.deltaq1  += recv_[n++];
      pc->s.sa       += recv_[n++];
      pc->s.saf      += recv_[n++];

      assert(n == (noff + npart + 1)*msize_[mtype_]);
    }

    npart++;
    pc = pc->next;
  }

  return npart;
}
#else
static int colloid_sums_m4(colloid_sum_t * sum, int ic, int jc, int kc,
			   int noff) {

  int n, npart;
  int index;
  colloid_t * pc;

  n = msize_[sum->mtype]*noff;
  npart = 0;
  colloids_info_cell_list_head(sum->cinfo, ic, jc, kc, &pc);

  while (pc) {

    if (sum->mload == MESSAGE_LOAD) {
      sum->send[n++] = 1.0*pc->s.index;
      sum->send[n++] = pc->s.deltaphi;
      sum->send[n++] = pc->s.deltaq0;
      sum->send[n++] = pc->s.deltaq1;
      sum->send[n++] = pc->s.sa;
      sum->send[n++] = pc->s.saf;

      assert(n == (noff + npart + 1)*msize_[sum->mtype]);
    }
    else {

      /* unload and check incoming index (a fatal error) */
      index = (int) sum->recv[n++];
      if (index != pc->s.index) fatal("Sum mismatch m4 (%d)\n", index);

      pc->s.deltaphi += sum->recv[n++];
      pc->s.deltaq0  += sum->recv[n++];
      pc->s.deltaq1  += sum->recv[n++];
      pc->s.sa       += sum->recv[n++];
      pc->s.saf      += sum->recv[n++];

      assert(n == (noff + npart + 1)*msize_[sum->mtype]);
    }

    npart++;
    pc = pc->next;
  }

  return npart;
}
#endif
