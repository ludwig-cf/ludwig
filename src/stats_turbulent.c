/*****************************************************************************
 *
 *  stats_turbulent.c
 *
 *  Statistics to probe turbulent flow.
 *
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "stats_turbulent.h"

struct stats_turb_s {
  pe_t * pe;
  cs_t * cs;
  double * ubar;
  int time_counter;
  MPI_Comm comm_y;
  MPI_Comm comm_z;
};

static int stats_turbulent_init_mpi(stats_turb_t * stat);

/*****************************************************************************
 *
 *  stats_turbulent_create
 *
 *****************************************************************************/

int stats_turbulent_create(pe_t * pe, cs_t * cs, stats_turb_t ** pobj) {

  stats_turb_t * obj = NULL;
  int nlocal[3];

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (stats_turb_t *) calloc(1, sizeof(stats_turb_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(stats_turb_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  cs_nlocal(cs, nlocal);

  obj->ubar = (double *) malloc(3*nlocal[X]*nlocal[Z]*sizeof(double));
  if (obj->ubar == NULL) pe_fatal(pe, "malloc(ubar_) failed\n");

  stats_turbulent_init_mpi(obj);
  stats_turbulent_ubar_zero(obj);

  *pobj = obj;

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
  int mpi_cartcoords[3];
  MPI_Comm comm;

  assert(stat);

  cs_cart_comm(stat->cs, &comm);
  cs_cart_coords(stat->cs, mpi_cartcoords);

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

  if (rank != mpi_cartcoords[Y]) {
    pe_fatal(stat->pe, "rank in streamwise_ communicator not cart_coords(Y)\n");
  }

  MPI_Comm_rank(stat->comm_z, &rank);

  if (rank != mpi_cartcoords[Z]) {
    pe_fatal(stat->pe, "rank in z communicator not cart_coords(Z)\n");
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

  cs_nlocal(stat->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(stat->cs, ic, jc, kc);
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

  cs_nlocal(stat->cs, nlocal);

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
  int nlocal[3];
  int ntotal[3];
  int n, is_writing;
  int token = 0;
  int rank;
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  const int tag_token = 4129;
  FILE   * fp_output = NULL;
  double * f1;
  double * f1z;
  double raverage;
  double uy;
  double ltot[3];

  MPI_Comm comm;
  MPI_Status status;

  assert(stat);

  cs_ltot(stat->cs, ltot);
  cs_nlocal(stat->cs, nlocal);
  cs_ntotal(stat->cs, ntotal);
  cs_cartsz(stat->cs, mpi_cartsz);
  cs_cart_comm(stat->cs, &comm);
  cs_cart_coords(stat->cs, mpi_cartcoords);

  f1 = (double *) malloc(3*nlocal[X]*nlocal[Z]*sizeof(double));
  assert(f1);
  if (f1 == NULL) pe_fatal(stat->pe, "malloc(f1) failed\n");

  f1z = (double *) malloc(3*ntotal[Z]*sizeof(double));
  assert(f1z);
  if (f1z == NULL) pe_fatal(stat->pe, "malloc(f1z) failed\n");

  /* Set the averaging factor (if no data, set to zero) */
  raverage = 0.0;
  if (stat->time_counter > 0) raverage = 1.0/(ltot[Y]*stat->time_counter); 

  /* Take the sum in the y-direction and store in f1(x,z) */

  MPI_Reduce(stat->ubar, f1, 3*nlocal[X]*nlocal[Z], MPI_DOUBLE, MPI_SUM,
	     0, stat->comm_y);

  /* Output now only involves cart_coords(Y) = 0 */

  if (mpi_cartcoords[Y] == 0) {

    is_writing = (mpi_cartcoords[Z] == 0);

    if (mpi_cartcoords[X] == 0) {
      /* Open the file */
      if (is_writing) fp_output = fopen(filename, "w");
    }
    else {
      /* Block until we get the token from the previous process and
       * then can reopen the file... */
      rank = cs_cart_neighb(stat->cs, CS_BACK, X);
      MPI_Recv(&token, 1, MPI_INT, rank, tag_token, comm, &status);

      if (is_writing) fp_output = fopen(filename, "a");
    }

    if (is_writing) {
      if (fp_output == NULL) pe_fatal(stat->pe, "fopen(%s) failed\n", filename);
    }

    for (ic = 1; ic <= nlocal[X]; ic++) {

      /* Correct f1[Y] for leesedwards planes before output */
      /* Also take the average here. */

      /* lees_edw_block_uy(le, ic, &uy); To be replaced */
      uy = 0.0;

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
	if (n != 3*ntotal[Z]) pe_fatal(stat->pe, "fwrite(f1z) returned %d\n", n);
      }
    }

    /* Close the file and send the token to the next process */

    if (is_writing) {
      if (ferror(fp_output)) {
	perror("perror: ");
	pe_fatal(stat->pe, "File error on writing %s\n", filename);
      }
      fclose(fp_output);
    }

    if (mpi_cartcoords[X] < mpi_cartsz[X] - 1) {
      rank = cs_cart_neighb(stat->cs, CS_FORW, X);
      MPI_Ssend(&token, 1, MPI_INT, rank, tag_token, comm);
    }
  }

  free(f1z);
  free(f1);

  return 0;
}
