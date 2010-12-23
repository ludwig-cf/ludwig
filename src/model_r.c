/*****************************************************************************
 *
 *  model_r.c
 *
 *  Distributions using 'reversed' storage (intended for GPU).
 *  This is an implementation of the interface defined in model.h.
 *
 *  Uses f[NVEL][NDIST][index] where index is the usual spatial index
 *  via coords_index(ic, jc, kc). 
 *
 *  The LB model is either D2Q9, D3Q15 or D3Q19, as included in model.h.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "io_harness.h"
#include "model.h"

const double cs2  = (1.0/3.0);
const double rcs2 = 3.0;

double * f_;

static int ndist_ = 1;
static int nsite_ = 0;
static int initialised_ = 0;
static struct io_info_t * io_info_distribution_; 

static int distributions_read(FILE *, const int, const int, const int);
static int distributions_write(FILE *, const int, const int, const int);

/***************************************************************************
 *
 *  distribution_init
 *
 *  Irrespective of the value of nhalo associated with coords.c,
 *  we only ever at the moment pass one plane worth of distribution
 *  values. This is nhalolocal.
 *
 ***************************************************************************/
 
void distribution_init(void) {

  int ndata;

  nsite_ = coords_nsites();

  /* The total number of distribution data is then... */

  ndata = NVEL*ndist_*nsite_;

  f_ = (double  *) malloc(ndata*sizeof(double));
  if (f_ == NULL) fatal("malloc(distributions) failed\n");

  initialised_ = 1;

  distribution_halo_set_complete();
  distribution_init_f();

  return;
}

/*****************************************************************************
 *
 *  distribution_init_f
 *
 *  Fluid uniformly at rest.
 *
 *****************************************************************************/

void distribution_init_f(void) {

  int nlocal[3];
  int ic, jc, kc, index;

  assert(initialised_);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	/* should be rho0 */
	distribution_zeroth_moment_set_equilibrium(index, 0, 1.0);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_finish
 *
 *  Clean up.
 *
 *****************************************************************************/

void distribution_finish(void) {

  if (io_info_distribution_) io_info_destroy(io_info_distribution_);
  free(f_);

  initialised_ = 0;
  ndist_ = 1;

  return;
}

/*****************************************************************************
 *
 *  distribution_io_info_set
 *
 *****************************************************************************/

void distribution_io_info_set(struct io_info_t * io_info) {

  char string[FILENAME_MAX];

  assert(io_info);
  io_info_distribution_ = io_info;

  sprintf(string, "%1d x Distribution: d%dq%d reverse", ndist_, NDIM, NVEL);

  io_info_set_name(io_info_distribution_, string);
  io_info_set_read(io_info_distribution_, distributions_read);
  io_info_set_write(io_info_distribution_, distributions_write);
  io_info_set_bytesize(io_info_distribution_, ndist_*NVEL*sizeof(double));

  return;
}

/*****************************************************************************
 *
 *  distribution_io_info
 *
 *****************************************************************************/

struct io_info_t * distribution_io_info(void) {

  return io_info_distribution_;
}

/*****************************************************************************
 *
 *  distribution_get_stress_at_site
 *
 *  Return the (deviatoric) stress at index.
 *
 *****************************************************************************/

void distribution_get_stress_at_site(int index, double s[3][3]) {

  int p, ia, ib;

  /* Want to ensure index is ok */
  assert(index >= 0  && index < nsite_);

  for (ia = 0; ia < NDIM; ia++) {
    for (ib = 0; ib < NDIM; ib++) {
      s[ia][ib] = 0.0;
    }
  }

  for (p = 0; p < NVEL; p++) {
    for (ia = 0; ia < NDIM; ia++) {
      for (ib = 0; ib < NDIM; ib++) {
	s[ia][ib] += f_[p*ndist_*nsite_ + index]*q_[p][ia][ib];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_halo
 *
 *  Swap the distributions at the periodic/processor boundaries
 *  in each direction.
 *
 *****************************************************************************/

void distribution_halo() {

  int ic, jc, kc;
  int n, p;
  int pforw, pback;
  int ihalo, ireal;
  int indexhalo, indexreal;
  int nsend, count;
  int nlocal[3];

  const int tagf = 900;
  const int tagb = 901;

  double * sendforw;
  double * sendback;
  double * recvforw;
  double * recvback;

  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm = cart_comm();

  assert(initialised_);

  coords_nlocal(nlocal);

  /* The x-direction (YZ plane) */

  nsend = NVEL*ndist_*nlocal[Y]*nlocal[Z];
  sendforw = (double *) malloc(nsend*sizeof(double));
  sendback = (double *) malloc(nsend*sizeof(double));
  recvforw = (double *) malloc(nsend*sizeof(double));
  recvback = (double *) malloc(nsend*sizeof(double));
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvforw == NULL) fatal("malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < ndist_; n++) {
      ireal = p*ndist_*nsite_ + n*nsite_;
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  indexreal = ireal + coords_index(nlocal[X], jc, kc);
	  sendforw[count] = f_[indexreal];

	  indexreal = ireal + coords_index(1, jc, kc);
	  sendback[count] = f_[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);

  if (cart_size(X) == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cart_neighb(FORWARD, X);
    pback = cart_neighb(BACKWARD, X);

    MPI_Irecv(recvforw, nsend, MPI_DOUBLE, pforw, tagb, comm, request);
    MPI_Irecv(recvback, nsend, MPI_DOUBLE, pback, tagf, comm, request + 1);

    MPI_Issend(sendback, nsend, MPI_DOUBLE, pback, tagb, comm, request + 2);
    MPI_Issend(sendforw, nsend, MPI_DOUBLE, pforw, tagf, comm, request + 3);

    /* Wait for receives */
    MPI_Waitall(2, request, status);
  }

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < ndist_; n++) {
      ihalo = p*ndist_*nsite_ + n*nsite_;
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  indexhalo = ihalo + coords_index(0, jc, kc);
	  f_[indexhalo] = recvback[count];

	  indexhalo = ihalo + coords_index(nlocal[X]+1, jc, kc);
	  f_[indexhalo] = recvforw[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);

  free(recvback);
  free(recvforw);

  if (cart_size(X) > 1) {
    /* Wait for sends */
    MPI_Waitall(2, request + 2, status);
  }

  free(sendback);
  free(sendforw);
  

  /* The y-direction (XZ plane) */

  nsend = NVEL*ndist_*(nlocal[X] + 2)*nlocal[Z];
  sendforw = (double *) malloc(nsend*sizeof(double));
  sendback = (double *) malloc(nsend*sizeof(double));
  recvforw = (double *) malloc(nsend*sizeof(double));
  recvback = (double *) malloc(nsend*sizeof(double));
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvforw == NULL) fatal("malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < ndist_; n++) {
      ireal = p*ndist_*nsite_ + n*nsite_;
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  indexreal = ireal + coords_index(ic, nlocal[Y], kc);
	  sendforw[count] = f_[indexreal];

	  indexreal = ireal + coords_index(ic, 1, kc);
	  sendback[count] = f_[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);


  if (cart_size(Y) == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cart_neighb(FORWARD, Y);
    pback = cart_neighb(BACKWARD, Y);

    MPI_Irecv(recvforw, nsend, MPI_DOUBLE, pforw, tagb, comm, request);
    MPI_Irecv(recvback, nsend, MPI_DOUBLE, pback, tagf, comm, request + 1);

    MPI_Issend(sendback, nsend, MPI_DOUBLE, pback, tagb, comm, request + 2);
    MPI_Issend(sendforw, nsend, MPI_DOUBLE, pforw, tagf, comm, request + 3);

    /* Wait of receives */
    MPI_Waitall(2, request, status);
  }

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < ndist_; n++) {
      ihalo = p*ndist_*nsite_ + n*nsite_;
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  indexhalo = ihalo + coords_index(ic, 0, kc);
	  f_[indexhalo] = recvback[count];

	  indexhalo = ihalo + coords_index(ic, nlocal[Y] + 1, kc);
	  f_[indexhalo] = recvforw[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);
  free(recvback);
  free(recvforw);

  if (cart_size(Y) > 1) {
    /* Wait for sends */
    MPI_Waitall(2, request + 2, status);
  }

  free(sendback);
  free(sendforw);

  /* Finally, z-direction (XY plane) */

  nsend = NVEL*ndist_*(nlocal[X] + 2)*(nlocal[Y] + 2);
  sendforw = (double *) malloc(nsend*sizeof(double));
  sendback = (double *) malloc(nsend*sizeof(double));
  recvforw = (double *) malloc(nsend*sizeof(double));
  recvback = (double *) malloc(nsend*sizeof(double));
  if (sendforw == NULL) fatal("malloc(sendforw) failed\n");
  if (sendback == NULL) fatal("malloc(sendback) failed\n");
  if (recvforw == NULL) fatal("malloc(recvforw) failed\n");
  if (recvforw == NULL) fatal("malloc(recvback) failed\n");

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < ndist_; n++) {
      ireal = p*ndist_*nsite_ + n*nsite_;
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (jc = 0; jc <= nlocal[Y] + 1; jc++) {

	  indexreal = ireal + coords_index(ic, jc, nlocal[Z]);
	  sendforw[count] = f_[indexreal];

	  indexreal = ireal + coords_index(ic, jc, 1);
	  sendback[count] = f_[indexreal];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);

  if (cart_size(Z) == 1) {
    memcpy(recvback, sendforw, nsend*sizeof(double));
    memcpy(recvforw, sendback, nsend*sizeof(double));
  }
  else {

    pforw = cart_neighb(FORWARD, Z);
    pback = cart_neighb(BACKWARD, Z);

    MPI_Irecv(recvforw, nsend, MPI_DOUBLE, pforw, tagb, comm, request);
    MPI_Irecv(recvback, nsend, MPI_DOUBLE, pback, tagf, comm, request + 1);

    MPI_Issend(sendback, nsend, MPI_DOUBLE, pback, tagb, comm, request + 2);
    MPI_Issend(sendforw, nsend, MPI_DOUBLE, pforw, tagf, comm, request + 3);

    /* Wait for receives */
    MPI_Waitall(2, request, status);
  }

  count = 0;
  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < ndist_; n++) {
      ihalo = p*ndist_*nsite_ + n*nsite_;
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
	for (jc = 0; jc <= nlocal[Y] + 1; jc++) {

	  indexhalo = ihalo + coords_index(ic, jc, 0);
	  f_[indexhalo] = recvback[count];

	  indexhalo = ihalo + coords_index(ic, jc, nlocal[Z] + 1);
	  f_[indexhalo] = recvforw[count];
	  ++count;
	}
      }
    }
  }
  assert(count == nsend);
  free(recvback);
  free(recvforw);

  if (cart_size(Z) > 1) {
    /* Wait for sends */
    MPI_Waitall(2, request + 2, status);
  }

  free(sendback);
  free(sendforw);
 
  return;
}

/*****************************************************************************
 *
 *  read_distributions
 *
 *  Read one lattice site (ic, jc, kc) worth of distributions.
 *
 *****************************************************************************/

static int distributions_read(FILE * fp, const int ic, const int jc,
			      const int kc) {

  int index, n, nread, p;

  index = coords_index(ic, jc, kc);
  nread = 0;

  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < ndist_; n++) {
      nread += fread(f_ + p*ndist_*nsite_ + n*nsite_ + index,
		     sizeof(double), ndist_*NVEL, fp);
    }
  }

  if (nread != ndist_*NVEL) {
    fatal("fread(distribution) failed at %d %d %d\n", ic, jc,kc);
  }

  return n;
}

/*****************************************************************************
 *
 *  distributions_write
 *
 *  Write one lattice site (ic, jc, kc) worth of distributions.
 *
 *****************************************************************************/

static int distributions_write(FILE * fp, const int ic , const int jc,
			       const int kc) {

  int index, n, nwrite, p;

  index = coords_index(ic, jc, kc);
  nwrite = 0;

  for (p = 0; p < NVEL; p++) {
    for (n = 0; n < ndist_; n++) {
      nwrite += fwrite(f_ + p*ndist_*nsite_ + n*nsite_ + index,
		       sizeof(double), 1, fp);
    }
  }

  if (nwrite != ndist_*NVEL) {
    fatal("fwrite(distribution) failed at %d %d %d\n", ic, jc, kc);
  }

  return n;
}

/*****************************************************************************
 *
 *  distribution_halo_set_complete
 *
 *  Set the actual halo datatype to the full type to swap
 *  all NVEL distributions.
 *
 *****************************************************************************/

void distribution_halo_set_complete(void) {

  assert(initialised_);

  return;
}

/*****************************************************************************
 *
 *  distribution_halo_set_reduced
 *
 *  Set the actual halo datatype to the reduced type to send only the
 *  propagating elements of the distribution in a given direction.
 *
 *****************************************************************************/

void distribution_halo_set_reduced(void) {

  assert(initialised_);

  info("distribution_halo_set_reduced not yet implemented\n");

  return;
}

/*****************************************************************************
 *
 *  distribution_ndist
 *
 *  Return the number of distribution functions.
 *
 *****************************************************************************/

int distribution_ndist(void) {

  return ndist_;
}

/*****************************************************************************
 *
 *  distribution_ndist_set
 *
 *  Set the number of distribution functions to be used.
 *
 *****************************************************************************/

void distribution_ndist_set(const int n) {

  assert(initialised_ == 0);
  assert(n > 0);
  ndist_ = n;

  return;
}

/*****************************************************************************
 *
 *  distribution_f
 *
 *  Get the distribution at site index, velocity p, distribution n.
 *
 *****************************************************************************/

double distribution_f(const int index, const int p, const int n) {

  assert(initialised_);
  assert(index >= 0 && index < nsite_);
  assert(p >= 0 && p < NVEL);
  assert(n >= 0 && n < ndist_);

  return f_[p*ndist_*nsite_ + n*nsite_ + index];
}

/*****************************************************************************
 *
 *  distribution_f_set
 *
 *  Set the distribution for site index, velocity p, distribution n.
 *
 *****************************************************************************/

void distribution_f_set(const int index, const int p, const int n,
			const double fvalue) {

  assert(initialised_);
  assert(index >= 0 && index < nsite_);
  assert(p >= 0 && p < NVEL);
  assert(n >= 0 && n < ndist_);

  f_[p*ndist_*nsite_ + n*nsite_ + index] = fvalue;

  return;
}

/*****************************************************************************
 *
 *  distribution_zeroth_moment
 *
 *  Return the zeroth moment of the distribution (rho for n = 0).
 *
 *****************************************************************************/

double distribution_zeroth_moment(const int index, const int n) {

  int p;
  double rho;

  assert(initialised_);
  assert(index >= 0 && index < nsite_);
  assert(n >= 0 && n < ndist_);

  rho = 0.0;

  for (p = 0; p < NVEL; p++) {
    rho += f_[p*ndist_*nsite_ + n*nsite_ + index];
  }

  return rho;
}

/*****************************************************************************
 *
 *  distribution_first_moment
 *
 *  Return the first moment of the distribution p.
 *
 *****************************************************************************/

void distribution_first_moment(const int index, const int n, double g[3]) {

  int p;
  int ia;

  assert(initialised_);
  assert(index >= 0 && index < nsite_);
  assert(n >= 0 && n < ndist_);

  for (ia = 0; ia < 3; ia++) {
    g[ia] = 0.0;
  }

  for (p = 0; p < NVEL; p++) {
    for (ia = 0; ia < NDIM; ia++) {
      g[ia] += cv[p][ia]*f_[p*ndist_*nsite_ + n*nsite_ + index];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_zeroth_moment_set_equilibrium
 *
 *  Project the given value of rho onto the equilibrium distribution
 *  via
 *
 *    f_i = w_i rho
 *
 *****************************************************************************/

void distribution_zeroth_moment_set_equilibrium(const int index, const int n,
						const double rho) {
  int p;

  assert(initialised_);
  assert (n >= 0 && n < ndist_);
  assert(index >= 0 && index < nsite_);

  for (p = 0; p < NVEL; p++) {
    f_[p*ndist_*nsite_ + n*nsite_ + index] = wv[p]*rho;
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_index
 *
 *  Return the distribution n at index.
 *
 *****************************************************************************/

void distribution_index(const int index, const int n, double f[NVEL]) {

  int p;

  assert(initialised_);
  assert(n >= 0 && n < ndist_);
  assert(index >= 0 && index < nsite_);

  for (p = 0; p < NVEL; p++) {
    f[p] = f_[p*ndist_*nsite_ + n*nsite_ + index];
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_index_set
 *
 *  Set distribution n and index.
 *
 *****************************************************************************/

void distribution_index_set(const int index, const int n,
			    const double f[NVEL]) {
  int p;

  assert(initialised_);
  assert(n >= 0 && n < ndist_);
  assert(index >= 0 && index < nsite_);

  for (p = 0; p < NVEL; p++) {
    f_[p*ndist_*nsite_ + n*nsite_ + index] = f[p];
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_order
 *
 *****************************************************************************/

int distribution_order(void) {

  return MODEL_R;
}
