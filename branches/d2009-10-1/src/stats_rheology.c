/*****************************************************************************
 *
 *  stats_rheology
 *
 *  Profiles of various quantities in the shear direction (for Lees
 *  Edwards planes at x = constant).
 *
 *  In particular we have the free energy density profile (averaged
 *  over y,z), the stress_xy profile (averaged over y,z,t). There is
 *  also an instantaneous stress (averaged over the system).
 *
 *  $Id: stats_rheology.c,v 1.6.4.1 2009-11-04 10:20:43 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "control.h"
#include "lattice.h"
#include "physics.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "stats_rheology.h"

static int initialised_ = 0;
static int counter_sxy_ = 0;
static double * sxy_;
static double * stat_xz_;
static MPI_Comm comm_yz_;
static MPI_Comm comm_y_;
static MPI_Comm comm_z_;

static void stats_rheology_print_s(const char *, double s[3][3]);
static void stats_rheology_print_matrix(FILE *, double s[3][3]);

#define NSTAT1 7  /* Number of data items for stress statistics */
#define NSTAT2 12 /* Number of data items for 3-d stress stats
		   * 3 components of velocity, 3 components of
                   * 3 different contributions to the stress */

/*****************************************************************************
 *
 *  stats_rheology_init
 *
 *  Set up communicator in YZ plane. For the output strategy to work,
 *  rank zero in the new communicator should correspond to
 *  cart_coords(Y) = 0 and cart_coords(Z) = 0.
 *
 *****************************************************************************/

void stats_rheology_init(void) {

  int rank;
  int remainder[3];
  int nlocal[3];

  remainder[X] = 0;
  remainder[Y] = 1;
  remainder[Z] = 1;

  MPI_Cart_sub(cart_comm(), remainder, &comm_yz_);
  MPI_Comm_rank(comm_yz_, &rank);

  if (rank == 0) {
    if (cart_coords(Y) != 0) {
      fatal("rank 0 in stats_rheology comm_yz_ not cart_coords(Y) zero\n");
    }
    if (cart_coords(Z) != 0) {
      fatal("rank 0 in stats_rheology comm_yz_ not cart_coords(Z) zero\n");
    }
  }

  remainder[X] = 0;
  remainder[Y] = 1;
  remainder[Z] = 0;

  MPI_Cart_sub(cart_comm(), remainder, &comm_y_);

  remainder[X] = 0;
  remainder[Y] = 0;
  remainder[Z] = 0;

  MPI_Cart_sub(cart_comm(), remainder, &comm_z_);

  MPI_Comm_rank(comm_y_, &rank);

  if (rank != cart_coords(Y)) {
    fatal("rank not equal to cart_coords(Y) in rheology stats\n");
  }

  MPI_Comm_rank(comm_z_, &rank);

  if (rank != cart_coords(Z)) {
    fatal("rank not equal to cart_coords(Z) in rheology stats\n");
  }

  /* sxy_ */

  get_N_local(nlocal);

  sxy_ = (double *) malloc(NSTAT1*nlocal[X]*sizeof(double));
  if (sxy_ == NULL) fatal("malloc(sxy_) failed\n");

  /* stat_xz_ */

  stat_xz_ = (double *) malloc(NSTAT2*nlocal[X]*nlocal[Z]*sizeof(double));
  if (stat_xz_ == NULL) fatal("malloc(stat_xz_) failed\n");

  initialised_ = 1;

  stats_rheology_stress_profile_zero();

  return;
}

/*****************************************************************************
 *
 *  stats_rheology_finish
 *
 *****************************************************************************/

void stats_rheology_finish(void) {

  assert(initialised_);

  MPI_Comm_free(&comm_yz_);
  MPI_Comm_free(&comm_y_);
  MPI_Comm_free(&comm_z_);
  free(sxy_);
  free(stat_xz_);
  initialised_ = 0;

  return;
}

/*****************************************************************************
 *
 *  stats_rheology_free_energy_density_profile
 *
 *  Compute and write to file a profile of the free energy density
 *    fex(x) = mean_yz free_energy_density(x,y,z)
 *
 *****************************************************************************/

void stats_rheology_free_energy_density_profile(const char * filename) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];
  int rank;
  int token = 0;
  const int tag_token = 28125;

  double * fex;
  double * fexmean;
  double raverage;

  FILE * fp_output;
  MPI_Status status;
  MPI_Comm comm = cart_comm();

  double (* free_energy_density)(const int index);

  assert(initialised_);

  get_N_local(nlocal);
  get_N_offset(noffset);

  fex = (double *) malloc(nlocal[X]*sizeof(double));
  if (fex == NULL) fatal("malloc(fex) failed\n");
  fexmean = (double *) malloc(nlocal[X]*sizeof(double));
  if (fexmean == NULL) fatal("malloc(fexmean failed\n");

  free_energy_density = fe_density_function();

  /* Accumulate the local average over y,z */
  /* Do the reduction in (y,z) to give local f(x) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    fex[ic-1] = 0.0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);
	fex[ic-1] += free_energy_density(index); 
      }
    }
  }

  MPI_Reduce(fex, fexmean, nlocal[X], MPI_DOUBLE, MPI_SUM, 0, comm_yz_);

  /* Write f(x) to file in order */

  raverage = 1.0/(L(Y)*L(Z));

  if (cart_coords(Y) == 0 && cart_coords(Z) == 0) {

    if (cart_coords(X) == 0) {
      /* First to write ... */
      fp_output = fopen(filename, "w");
    }
    else {
      /* Block unitl the token is received from the previous process,
       * then reopen the file */

      rank = cart_neighb(BACKWARD, X);
      MPI_Recv(&token, 1, MPI_INT, rank, tag_token, comm, &status);
      fp_output = fopen(filename, "a");
    }

    if (fp_output == NULL) fatal("fopen(%s) failed\n");

    for (ic = 1; ic <= nlocal[X]; ic++) {
      /* This is an average over (y,z) so don't care about the
       * Lees Edwards planes (but remember to take average!) */

      fexmean[ic-1] *= raverage;
      fprintf(fp_output, "%6d %18.12e\n", noffset[X] + ic, fexmean[ic-1]);
    }

    /* Close the file and send the token to next process */

    if (ferror(fp_output)) {
      perror("perror: ");
      fatal("File error on writing %s\n", filename);
    }
    fclose(fp_output);

    if (cart_coords(X) < cart_size(X) - 1) {
      rank = cart_neighb(FORWARD, X);
      MPI_Ssend(&token, 1, MPI_INT, rank, tag_token, comm);
    }
  }

  free(fex);
  free(fexmean);

  return;
}

/*****************************************************************************
 *
 *  stats_rheology_stress_profile_zero
 *
 *  Set the stress profile to zero.
 *
 *****************************************************************************/

void stats_rheology_stress_profile_zero(void) {

  int ic, kc, n;
  int nlocal[3];

  assert(initialised_);
  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (n = 0; n < NSTAT1; n++) {
      sxy_[NSTAT1*(ic-1) + n] = 0.0;
    }
    for (kc = 1; kc <= nlocal[Z]; kc++) {
      for (n = 0; n < NSTAT2; n++) {
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + n] = 0.0;
      }
    }
  }


  counter_sxy_ = 0;

  return;
}

/*****************************************************************************
 *
 *  stats_rheology_stress_profile_accumulate
 *
 *  Accumulate the contribution to the mean stress profile
 *  for this time step.
 *
 *****************************************************************************/

void stats_rheology_stress_profile_accumulate(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  double rho;
  double u[3];
  double s[3][3];

  void (* chemical_stress)(const int index, double s[3][3]);

  assert(initialised_);
  get_N_local(nlocal);

  chemical_stress = fe_chemical_stress_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = ADDR(ic, jc, kc);
	distribution_get_stress_at_site(index, s);

	sxy_[NSTAT1*(ic-1)    ] += s[X][Y];

	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  0] += s[X][Y];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  1] += s[X][Z];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  2] += s[Y][Z];

        chemical_stress(index, s);

	sxy_[NSTAT1*(ic-1) + 1] += s[X][Y];

	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  3] += s[X][Y];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  4] += s[X][Z];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  5] += s[Y][Z];

	rho = get_rho_at_site(index);
	rho = 1.0/rho;
	get_momentum_at_site(index, u);

	sxy_[NSTAT1*(ic-1) + 2] += rho*u[X]*u[Y];
	sxy_[NSTAT1*(ic-1) + 3] += rho*u[X];
	sxy_[NSTAT1*(ic-1) + 4] += rho*u[Y];
	sxy_[NSTAT1*(ic-1) + 5] += rho*u[Z];

	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  6] += rho*u[X]*u[Y];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  7] += rho*u[X]*u[Z];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  8] += rho*u[Y]*u[Z];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) +  9] += rho*u[X];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + 10] += rho*u[Y];
	stat_xz_[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + 11] += rho*u[Z];

	hydrodynamics_velocity_gradient_tensor(ic, jc,  kc, s);
	sxy_[NSTAT1*(ic-1) + 6] += (s[X][Y] + s[Y][X]);
      }
    }
  }

  counter_sxy_++;

  return;
}

/*****************************************************************************
 *
 *  stats_rheology_stress_profile
 *
 *  Average and output the accumulated mean stress profile. There are
 *  three results which are the xy components of
 *
 *    mean_yzt hydrodymanic stress(x)
 *    mean_yzt chemical stress(x)
 *    mean_yzt rho uu(x)
 *
 *  As it's convenient, we also take the opportunity to get the mean
 *  velocities:
 *
 *    mean_yzt u_x
 *    mean_yzt u_y
 *    mean_yzt u_z
 *
 *  Finally, the viscosus stress via finite difference, which is
 *    mean_yzt eta*(d_x u_y + d_y u_x)
 *
 *****************************************************************************/

void stats_rheology_stress_profile(const char * filename) {

  int ic;
  int nlocal[3];
  int noffset[3];
  double * sxymean;
  double rmean;
  double uy;
  double eta = get_eta_shear();

  const int tag_token = 728;
  int rank;
  int token = 0;
  MPI_Status status;
  MPI_Comm comm = cart_comm();
  FILE * fp_output;

  assert(initialised_);
  get_N_local(nlocal);
  get_N_offset(noffset);

  sxymean = (double *) malloc(NSTAT1*nlocal[X]*sizeof(double));
  if (sxymean == NULL) fatal("malloc(sxymean) failed\n");

  MPI_Reduce(sxy_, sxymean, NSTAT1*nlocal[X], MPI_DOUBLE, MPI_SUM, 0, comm_yz_);

  rmean = 1.0/(L(Y)*L(Z)*counter_sxy_);

  /* Write to file in order of x */

  if (cart_coords(Y) == 0 && cart_coords(Z) == 0) {

    if (cart_coords(X) == 0) {
      /* First to write ... */
      fp_output = fopen(filename, "w");
    }
    else {
      /* Block unitl the token is received from the previous process,
       * then reopen the file */

      rank = cart_neighb(BACKWARD, X);
      MPI_Recv(&token, 1, MPI_INT, rank, tag_token, comm, &status);
      fp_output = fopen(filename, "a");
    }

    if (fp_output == NULL) fatal("fopen(%s) failed\n");

    for (ic = 1; ic <= nlocal[X]; ic++) {
      /* This is an average over (y,z) so don't care about the
       * Lees Edwards places, but correct uy */

      uy = le_get_block_uy(ic);

      fprintf(fp_output,
	      "%6d %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e\n",
	      noffset[X] + ic,
	      rmean*sxymean[NSTAT1*(ic-1)  ],
	      rmean*sxymean[NSTAT1*(ic-1)+1],
	      rmean*sxymean[NSTAT1*(ic-1)+2],
	      rmean*sxymean[NSTAT1*(ic-1)+3],
	      rmean*sxymean[NSTAT1*(ic-1)+4] + uy,
	      rmean*sxymean[NSTAT1*(ic-1)+5],
	      rmean*sxymean[NSTAT1*(ic-1)+6]*eta);

    }

    /* Close the file and send the token to next process */

    if (ferror(fp_output)) {
      perror("perror: ");
      fatal("File error on writing %s\n", filename);
    }
    fclose(fp_output);

    if (cart_coords(X) < cart_size(X) - 1) {
      rank = cart_neighb(FORWARD, X);
      MPI_Ssend(&token, 1, MPI_INT, rank, tag_token, comm);
    }
  }

  free(sxymean);

  return;
}

/*****************************************************************************
 *
 *  stats_rheology_stress_section
 *
 *  Outputs a section in the x-z direction (ie., averaged along the
 *  flow direction of various quantities:
 *
 *  Full stress                  s_xy, s_xz, s_yz from \sum_i f_i Q_iab
 *  Thermodynamic contribution   s_xy, s_xz, s_yz from P^th_ab
 *  Equilibrium stress           s_xy, s_xz, s_xz from rho u_a u_b
 *  Velocity                     u_x, u_y, x_z
 *
 *  In the output, the z-direction runs faster, then the x-direction.
 *
 *****************************************************************************/

void stats_rheology_stress_section(const char * filename) {

  int ic, kc;
  int nlocal[3];
  int n, is_writing;
  FILE   * fp_output = NULL;
  double * stat_2d;
  double * stat_1d;
  double raverage;
  double uy;

  MPI_Comm comm = cart_comm();
  MPI_Status status;
  int token = 0;
  int rank;
  const int tag_token = 1012;

  assert(initialised_);
  get_N_local(nlocal);

  stat_2d = (double *) malloc(NSTAT2*nlocal[X]*nlocal[Z]*sizeof(double));
  if (stat_2d == NULL) fatal("malloc(stat_2d) failed\n");

  stat_1d = (double *) malloc(NSTAT2*N_total(Z)*sizeof(double));
  if (stat_1d == NULL) fatal("malloc(stat_1d) failed\n");

  /* Set the averaging factor (if no data, set to zero) */

  raverage = 0.0;
  if (counter_sxy_ > 0) raverage = 1.0/(L(Y)*counter_sxy_); 

  /* Take the sum in the y-direction and store in stat_2d(x,z) */

  MPI_Reduce(stat_xz_, stat_2d, NSTAT2*nlocal[X]*nlocal[Z], MPI_DOUBLE,
	     MPI_SUM, 0, comm_y_);

  /* Output now only involves cart_coords(Y) = 0 */

  if (cart_coords(Y) == 0) {

    /* The strategy is to gather strip-wise in the z-direction,
     * and only write from the first process in this direction.
     * We then sweep over x to give an xz section. */

    is_writing = (cart_coords(Z) == 0);

    if (cart_coords(X) == 0) {
      /* Open the file */
      if (is_writing) fp_output = fopen(filename, "w");
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
	for (n = 0; n < NSTAT2; n++) {
	  stat_2d[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + n] *= raverage;
	}
	/* u_y must be corrected */
	stat_2d[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + 10] += uy;
      }

      MPI_Gather(stat_2d + NSTAT2*nlocal[Z]*(ic-1), NSTAT2*nlocal[Z],
		 MPI_DOUBLE, stat_1d, NSTAT2*nlocal[Z], MPI_DOUBLE, 0,
		 comm_z_);

      /* write data */
      if (is_writing) {
	for (kc = 1; kc <= N_total(Z); kc++) {
	  for (n = 0; n < NSTAT2; n++) {
	    fprintf(fp_output, " %15.8e", stat_1d[NSTAT2*(kc-1) + n]);
	  }
	  fprintf(fp_output, "\n");
	}
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

  return;
}

/*****************************************************************************
 *
 *  stats_rheology_mean_stress
 *
 *  Provide a summary of the instantaneous mean stress to info().
 *  We have:
 *
 *    Full deviatoric stress:   S_ab    = \sum_i f_i Q_iab
 *    Chemical stress:          P_ab      for current free energy
 *    Equilibrium stress:       S^eq_ab = \rho u_a u_b
 *
 *****************************************************************************/

void stats_rheology_mean_stress(const char * filename) {

#define NCOMP 27

  double stress[3][3];
  double rhouu[3][3];
  double pchem[3][3], plocal[3][3];
  double s[3][3];
  double u[3];
  double send[NCOMP];
  double recv[NCOMP];
  double rho, rrho, rv;
  int nlocal[3];
  int ic, jc, kc, index, ia, ib;
  FILE * fp;

  void (* chemical_stress)(const int index, double s[3][3]);

  rv = 1.0/(L(X)*L(Y)*L(Z));

  get_N_local(nlocal);

  chemical_stress = fe_chemical_stress_function();

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      stress[ia][ib] = 0.0;
      pchem[ia][ib] = 0.0;
      rhouu[ia][ib] = 0.0;
    }
  }

  /* Accumulate contributions to the stress */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = get_site_index(ic, jc, kc);

	rho = get_rho_at_site(index);
	get_momentum_at_site(index, u);
	distribution_get_stress_at_site(index, s);
        chemical_stress(index, plocal);

	rrho = 1.0/rho;
        for (ia = 0; ia < 3; ia++) {
          for (ib = 0; ib < 3; ib++) {
            rhouu[ia][ib] += rrho*u[ia]*u[ib];
            stress[ia][ib] += s[ia][ib];
            pchem[ia][ib] += plocal[ia][ib];
          }
        }

      }
    }
  }

  /* Acculumate the sums */

  kc = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      send[kc++] = stress[ia][ib];
      send[kc++] = pchem[ia][ib];
      send[kc++] = rhouu[ia][ib];
    }
  }

  MPI_Reduce(send, recv, NCOMP, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  kc = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      /* Include the normaliser for the average here */
      stress[ia][ib] = rv*recv[kc++];
      pchem[ia][ib]  = rv*recv[kc++];
      rhouu[ia][ib]  = rv*recv[kc++];
    }
  }

  if (filename == NULL || strcmp(filename, "") == 0) {
    /* Use info() */
    stats_rheology_print_s("stress_hydro", stress);
    stats_rheology_print_s("stress_pchem", pchem);
    stats_rheology_print_s("stress_rhouu", rhouu);
  }
  else {
    /* Use filename supplied */

    if (pe_rank() == 0) {
      fp = fopen(filename, "a");
      if (fp == NULL) fatal("fopen(%s) failed\n", filename);

      fprintf(fp, "%9d ", get_step());
      stats_rheology_print_matrix(fp, stress);
      stats_rheology_print_matrix(fp, pchem);
      stats_rheology_print_matrix(fp, rhouu);
      fprintf(fp, "\n");

      fclose(fp);
    }
  }

  return;
}

/****************************************************************************
 *
 *  stats_rheology_print_matrix
 *
 *  Prints six components of a symmetric stress tensor.
 *
 ****************************************************************************/

static void stats_rheology_print_matrix(FILE * fp, double s[3][3]) {

  assert(fp);

  fprintf(fp, "%15.8e %15.8e %15.8e ", s[X][X], s[X][Y], s[X][Z]);  
  fprintf(fp, "%15.8e %15.8e %15.8e ", s[Y][Y], s[Y][Z], s[Z][Z]);  

  return;
}

/*****************************************************************************
 *
 *  stats_rheology_print_s
 *
 *  Output a matrix with a label.
 *
 *****************************************************************************/

static void stats_rheology_print_s(const char * label, double s[3][3]) {

  info("%s x %15.8e %15.8e %15.8e\n", label, s[X][X], s[X][Y], s[X][Z]);
  info("%s y %15.8e %15.8e %15.8e\n", label, s[Y][X], s[Y][Y], s[Y][Z]);
  info("%s z %15.8e %15.8e %15.8e\n", label, s[Z][X], s[Z][Y], s[Z][Z]);

  return;
}
