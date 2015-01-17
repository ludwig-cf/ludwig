/*****************************************************************************
 *
 *  stats_turbulent.c
 *
 *  Statistics to probe turbulent flow.
 *
 *  $Id: stats_turbulent.c,v 1.3 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008-2015 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "stats_turbulent.h"

struct stats_turb_s {
  coords_t * cs;       /* Reference to coordinate system */
  double * ubar;
  int time_counter;
  MPI_Comm comm_y;
  MPI_Comm comm_z;
};

static int stats_turbulent_init_mpi(stats_turb_t * stat);

/*****************************************************************************
 *
 *  stats_turbulent_init
 *
 *****************************************************************************/

int stats_turbulent_create(coords_t * cs, stats_turb_t ** pstat) {

  stats_turb_t * stat = NULL;
  int nlocal[3];

  assert(cs);
  assert(pstat);

  stat = (stats_turb_t *) calloc(1, sizeof(stats_turb_t));
  if (stat == NULL) fatal ("calloc(stats_turb_t) failed\n");

  coords_nlocal(cs, nlocal);

  stat->ubar = (double *) malloc(3*nlocal[X]*nlocal[Z]*sizeof(double));
  if (stat->ubar == NULL) fatal("calloc(stat->ubar) failed\n");

  stat->cs = cs;
  coords_retain(cs);

  stats_turbulent_init_mpi(stat);

  *pstat = stat;

  return 0;
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

static int stats_turbulent_init_mpi(stats_turb_t * stat) {

  int rank;
  int remainders[3];
  int cartcoords[3];
  MPI_Comm comm;

  assert(stat);

  coords_cart_comm(stat->cs, &comm);
  coords_cart_coords(stat->cs, cartcoords);

  /* Initialise the streamwise (y-direction) communicator */

  remainders[X] = 0;
  remainders[Y] = 1;
  remainders[Z] = 0;

  MPI_Cart_sub(comm, remainders, &stat->comm_y);

  /* Initialise the z-direction communicator */

  remainders[X] = 0;
  remainders[Y] = 0;
  remainders[Z] = 1;

  MPI_Cart_sub(comm, remainders, &stat->comm_z);

  /* Are the ranks ok? */

  MPI_Comm_rank(stat->comm_y, &rank);

  if (rank != cartcoords[Y]) {
    fatal("rank in streamwise_ communicator not cart_coords[Y]\n");
  }

  MPI_Comm_rank(stat->comm_z, &rank);

  if (rank != cartcoords[Z]) {
    fatal("rank in z communicator not cart_coords[Z]\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_turbulent_free
 *
 *****************************************************************************/

int stats_turbulent_free(stats_turb_t * stat) {

  assert(stat);

  MPI_Comm_free(&stat->comm_y);
  MPI_Comm_free(&stat->comm_z);

  free(stat->ubar);
  coords_free(stat->cs);
  free(stat);

  return 0;
}

/*****************************************************************************
 *
 *  stats_turbulent_ubar_accumulate
 *
 *  Accumulate the current contribution to ubar.
 *
 *****************************************************************************/

int stats_turbulent_ubar_accumulate(stats_turb_t * stat, hydro_t * hydro) {

  int nlocal[3];
  int ic, jc, kc;
  int index, ia;
  double u[3];

  assert(stat);
  assert(hydro);
  coords_nlocal(stat->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(stat->cs, ic, jc, kc);
	hydro_u(hydro, index, u);

	for (ia = 0; ia < 3; ia++) {
	  stat->ubar[3*(nlocal[Z]*(ic - 1) + kc - 1) + ia] += u[ia];
	}
      }
    }
  }

  stat->time_counter += 1;

  return 0;
}

/*****************************************************************************
 *
 *  stats_turbulent_ubar_zero
 *
 *  Zero the accumulated mean ubar_ and time step counter.
 *
 *****************************************************************************/

int stats_turbulent_ubar_zero(stats_turb_t * stat) {

  int nlocal[3];
  int ic, kc;
  int ia;

  assert(stat);
  coords_nlocal(stat->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (kc = 1; kc <= nlocal[Z]; kc++) {
      for (ia = 0; ia < 3; ia++) {
	stat->ubar[3*(nlocal[Z]*(ic - 1) + kc - 1) + ia] = 0.0;
      }
    }
  }

  stat->time_counter = 0;

  return 0;
}

/*****************************************************************************
 *
 *  stats_turbulent_output_ubar
 *
 *  Output the statistics to the given file.
 *  As this is a severely averaged data set, it's always in serial.
 *
 *****************************************************************************/

int stats_turbulent_ubar_output(stats_turb_t * stat, const char * filename) {

  int ic, kc;
  int ntotal[3];
  int nlocal[3];
  int cartcoords[3];
  int cartsz[3];
  int n, is_writing;
  FILE   * fp_output = NULL;
  double * f1;
  double * f1z;
  double raverage;
  double uy;
  double ltot[3];

  MPI_Comm comm;
  MPI_Status status;
  int token = 0;
  int rank;
  const int tag_token = 4129;

  assert(stat);

  coords_ltot(stat->cs, ltot);
  coords_cart_comm(stat->cs, &comm);
  coords_cart_coords(stat->cs, cartcoords);
  coords_cartsz(stat->cs, cartsz);
  coords_ntotal(stat->cs, ntotal);
  coords_nlocal(stat->cs, nlocal);

  f1 = (double *) malloc(3*nlocal[X]*nlocal[Z]*sizeof(double));
  if (f1 == NULL) fatal("malloc(f1) failed\n");

  f1z = (double *) malloc(3*ntotal[Z]*sizeof(double));
  if (f1z == NULL) fatal("malloc(f1z) failed\n");

  /* Set the averaging factor (if no data, set to zero) */
  raverage = 0.0;
  if (stat->time_counter > 0) raverage = 1.0/(ltot[Y]*stat->time_counter); 

  /* Take the sum in the y-direction and store in f1(x,z) */

  MPI_Reduce(stat->ubar, f1, 3*nlocal[X]*nlocal[Z], MPI_DOUBLE, MPI_SUM,
	     0, stat->comm_y);

  /* Output now only involves cart_coords[Y] = 0 */

  if (cartcoords[Y] == 0) {

    is_writing = (cartcoords[Z] == 0);

    if (cartcoords[X] == 0) {
      /* Open the file */
      if (is_writing) fp_output = fopen(filename, "w");
    }
    else {
      /* Block until we get the token from the previous process and
       * then can reopen the file... */
      rank = coords_cart_neighb(stat->cs, BACKWARD, X);
      MPI_Recv(&token, 1, MPI_INT, rank, tag_token, comm, &status);

      if (is_writing) fp_output = fopen(filename, "a");
    }

    if (is_writing) {
      if (fp_output == NULL) fatal("fopen(%s) failed\n", filename);
    }

    for (ic = 1; ic <= nlocal[X]; ic++) {

      /* Correct f1[Y] for leesedwards planes before output */
      /* Also take the average here. */

      /* uy = le_get_block_uy(ic);*/
      assert(0); /* Can above be corrected via hydro to avoid dependency
		    on LE ? */

      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (n = 0; n < 3; n++) {
	  f1[3*((ic-1)*nlocal[Z] + kc - 1) + n] *= raverage;
	}
	f1[3*((ic-1)*nlocal[Z] + kc - 1) + Y] += uy;
      }

      MPI_Gather(f1 + 3*(ic-1)*nlocal[Z], 3*nlocal[Z], MPI_DOUBLE, f1z,
		 3*nlocal[Z], MPI_DOUBLE, 0, stat->comm_z);

      /* write data */
      if (is_writing) {
	n = fwrite(f1z, sizeof(double), 3*ntotal[Z], fp_output);
	if (n != 3*ntotal[Z]) fatal("fwrite(f1z) returned %d\n", n);
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

    if (cartcoords[X] < cartsz[X] - 1) {
      rank = coords_cart_neighb(stat->cs, FORWARD, X);
      MPI_Ssend(&token, 1, MPI_INT, rank, tag_token, comm);
    }
  }

  free(f1z);
  free(f1);

  return 0;
}
