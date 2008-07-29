/*****************************************************************************
 *
 *  stats_turbulent.c
 *
 *  Statistics to probe turbulent flow.
 *
 *  $Id: stats_turbulent.c,v 1.1.2.1 2008-07-29 14:23:08 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "lattice.h"
#include "leesedwards.h"
#include "stats_turbulent.h"

static double * ubar_;
static int initialised_ = 0;
static int time_counter_ = 0;
static MPI_Comm comm_y_;
static MPI_Comm comm_z_;

static void stats_turbulent_init_mpi(void);

/*****************************************************************************
 *
 *  stats_turbulent_init
 *
 *****************************************************************************/

void stats_turbulent_init() {

  int nlocal[3];

  get_N_local(nlocal);

  ubar_ = (double *) malloc(3*nlocal[X]*nlocal[Z]*sizeof(double));
  if (ubar_ == NULL) fatal("malloc(ubar_) failed\n");

  stats_turbulent_init_mpi();
  initialised_ = 1;

  stats_turbulent_ubar_zero();
  return;
}

/*****************************************************************************
 *
 *  stats_turbulent_init_mpi
 *
 *  I have to check here that the MPI ranks the new 1-d communicators
 *  are those one expects (ie., the cart_coord in that direction). If
 *  not, the output logic will not work later. 
 *
 *****************************************************************************/

static void stats_turbulent_init_mpi() {

  int rank;
  int remainders[3];
  MPI_Comm comm = cart_comm();

  /* Initialise the streamwise (y-direction) communicator */

  remainders[X] = 0;
  remainders[Y] = 1;
  remainders[Z] = 0;

  MPI_Cart_sub(comm, remainders, &comm_y_);

  /* Initialise the z-direction communicator */

  remainders[X] = 0;
  remainders[Y] = 0;
  remainders[Z] = 1;

  MPI_Cart_sub(comm, remainders, &comm_z_);

  /* Are the ranks ok? */

  MPI_Comm_rank(comm_y_, &rank);

  if (rank != cart_coords(Y)) {
    fatal("rank in streamwise_ communicator not cart_coords(Y)\n");
  }

  MPI_Comm_rank(comm_z_, &rank);

  if (rank != cart_coords(Z)) {
    fatal("rank in z communicator not cart_coords(Z)\n");
  }

  return;
}

/*****************************************************************************
 *
 *  stats_turbulent_finish
 *
 *****************************************************************************/

void stats_turbulent_finish() {

  MPI_Comm_free(&comm_y_);
  MPI_Comm_free(&comm_z_);

  free(ubar_);
  initialised_ = 0;

  return;
}

/*****************************************************************************
 *
 *  stats_turbulent_ubar_accumulate
 *
 *  Accumulate the current contribution to ubar_.
 *
 *****************************************************************************/

void stats_turbulent_ubar_accumulate() {

  int nlocal[3];
  int ic, jc, kc;
  int index, ia;
  double u[3];

  assert(initialised_);
  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);
	hydrodynamics_get_velocity(index, u);

	for (ia = 0; ia < 3; ia++) {
	  ubar_[3*(nlocal[Z]*(ic - 1) + kc - 1) + ia] += u[ia];
	}
      }
    }
  }

  time_counter_ += 1;

  return;
}

/*****************************************************************************
 *
 *  stats_turbulent_ubar_zero
 *
 *  Zero the accumulated mean ubar_ and time step counter.
 *
 *****************************************************************************/

void stats_turbulent_ubar_zero() {

  int nlocal[3];
  int ic, kc;
  int ia;

  assert(initialised_);
  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (kc = 1; kc <= nlocal[Z]; kc++) {
      for (ia = 0; ia < 3; ia++) {
	ubar_[3*(nlocal[Z]*(ic - 1) + kc - 1) + ia] = 0.0;
      }
    }
  }

  time_counter_ = 0;

  return;
}

/*****************************************************************************
 *
 *  stats_turbulent_output_ubar
 *
 *  Output the statistics to the given file.
 *  As this is a severely averaged data set, it's always in serial.
 *
 *****************************************************************************/

void stats_turbulent_ubar_output(const char * filename) {

  int ic, kc;
  int nlocal[3];
  int n, is_writing;
  FILE   * fp_output = NULL;
  double * f1;
  double * f1z;
  double raverage;
  double uy;

  MPI_Comm comm = cart_comm();
  MPI_Status status;
  int token = 0;
  int rank;
  const int tag_token = 4129;

  assert(initialised_);
  get_N_local(nlocal);

  f1 = (double *) malloc(3*nlocal[X]*nlocal[Z]*sizeof(double));
  if (f1 == NULL) fatal("malloc(f1) failed\n");

  f1z = (double *) malloc(3*N_total(Z)*sizeof(double));
  if (f1z == NULL) fatal("malloc(f1z) failed\n");

  /* Set the averaging factor (if no data, set to zero) */
  raverage = 0.0;
  if (time_counter_ > 0) raverage = 1.0/(L(Y)*time_counter_); 

  /* Take the sum in the y-direction and store in f1(x,z) */

  MPI_Reduce(ubar_, f1, 3*nlocal[X]*nlocal[Z], MPI_DOUBLE, MPI_SUM,
	     0, comm_y_);

  /* Output now only involves cart_coords(Y) = 0 */

  if (cart_coords(Y) == 0) {

    is_writing = (cart_coords(Z) == 0);

    if (cart_coords(X) == 0) {
      /* Open the file */
      if (is_writing) fp_output = fopen(filename, "w");
      verbose("root x opens file\n");
    }
    else {
      /* Block until we get the token from the previous process and
       * then can reopen the file... */
      rank = cart_neighb(BACKWARD, X);
      MPI_Recv(&token, 1, MPI_INT, rank, tag_token, comm, &status);

      if (is_writing) fp_output = fopen(filename, "a");
    }

    if (is_writing) {
      if (fp_output == NULL) fatal("fopen(%s) failed\n", filename);
    }

    for (ic = 1; ic <= nlocal[X]; ic++) {

      /* Correct f1[Y] for leesedwards planes before output */
      /* Also take the average here. */

      uy = le_get_block_uy(ic);

      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (n = 0; n < 3; n++) {
	  f1[3*((ic-1)*nlocal[Z] + kc - 1) + n] *= raverage;
	}
	f1[3*((ic-1)*nlocal[Z] + kc - 1) + Y] += uy;
      }

      MPI_Gather(f1 + 3*(ic-1)*nlocal[Z], 3*nlocal[Z], MPI_DOUBLE, f1z,
		 3*nlocal[Z], MPI_DOUBLE, 0, comm_z_);

      /* write data */
      if (is_writing) {
	n = fwrite(f1z, sizeof(double), 3*N_total(Z), fp_output);
	if (n != 3*N_total(Z)) fatal("fwrite(f1z) returned %d\n", n);
      }
    }

    /* Close the file and send the token to the next process */

    if (is_writing) {
      if (ferror(fp_output)) {
	perror("perror: ");
	fatal("File error on writing %s\n", filename);
      }
      fclose(fp_output);
    }

    if (cart_coords(X) < cart_size(X) - 1) {
      rank = cart_neighb(FORWARD, X);
      MPI_Ssend(&token, 1, MPI_INT, rank, tag_token, comm);
    }
  }

  free(f1z);
  free(f1);

  return;
}
