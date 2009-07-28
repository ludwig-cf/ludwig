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
 *  $Id: stats_rheology.c,v 1.1 2009-07-28 11:31:57 kevin Exp $
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

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "stats_rheology.h"

static int initialised_ = 0;
static int counter_sxy_ = 0;
static double * sxy_;
static MPI_Comm comm_yz_;

static void stats_rheology_print_s(const char *, double s[3][3]);

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

  /* sxy_ */

  get_N_local(nlocal);

  sxy_ = (double *) malloc(3*nlocal[X]*sizeof(double));
  if (sxy_ == NULL) fatal("malloc(sxy_) failed\n");

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
  free(sxy_);
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

  assert(initialised_);

  get_N_local(nlocal);
  get_N_offset(noffset);

  fex = (double *) malloc(nlocal[X]*sizeof(double));
  if (fex == NULL) fatal("malloc(fex) failed\n");
  fexmean = (double *) malloc(nlocal[X]*sizeof(double));
  if (fexmean == NULL) fatal("malloc(fexmean failed\n");

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

  int ic;
  int nlocal[3];

  assert(initialised_);
  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    sxy_[3*(ic-1)    ] = 0.0;
    sxy_[3*(ic-1) + 1] = 0.0;
    sxy_[3*(ic-1) + 2] = 0.0;
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

  assert(initialised_);
  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = ADDR(ic, jc, kc);
	distribution_get_stress_at_site(index, s);
	sxy_[3*(ic-1)    ] += s[X][Y];
        free_energy_get_chemical_stress(index, s);
	sxy_[3*(ic-1) +  1] += s[X][Y];
	rho = get_rho_at_site(index);
	get_momentum_at_site(index, u);
	sxy_[3*(ic-1) +  2] += rho*u[X]*u[Y];
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
 *    mean_yzt hydrodymanic stess(x)
 *    mean_yzt chemical stess(x)
 *    mean_yzt rho uu(x) 
 *
 *****************************************************************************/

void stats_rheology_stress_profile(const char * filename) {

  int ic;
  int nlocal[3];
  int noffset[3];
  double * sxymean;
  double rmean;

  const int tag_token = 90728;
  int rank;
  int token = 0;
  MPI_Status status;
  MPI_Comm comm = cart_comm();
  FILE * fp_output;

  assert(initialised_);
  get_N_local(nlocal);
  get_N_offset(noffset);

  sxymean = (double *) malloc(3*nlocal[X]*sizeof(double));
  if (sxymean == NULL) fatal("malloc(sxymean) failed\n");

  MPI_Reduce(sxy_, sxymean, 3*nlocal[X], MPI_DOUBLE, MPI_SUM, 0, comm_yz_);

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
       * Lees Edwards planes */

      fprintf(fp_output, "%6d %18.10e %18.10e %18.10e\n", noffset[X] + ic,
	      rmean*sxymean[3*(ic-1)], rmean*sxymean[3*(ic-1)+1],
	      rmean*sxymean[3*(ic-1)+2]);
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
 *  stats_rheology_mean_stress
 *
 *  Provide a summary of the instantaneous mean stress to info().
 *  We have:
 *
 *    Full deviatoric stress:   S_ab    = \sum_i f_i Q_iab
 *    Equilibrium stress:       S^eq_ab = \rho u_a u_b
 *    Chemical stress:          P_ab      for current free energy
 *
 *****************************************************************************/

void stats_rheology_mean_stress(void) {

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

  rv = 1.0/(L(X)*L(Y)*L(Z));

  get_N_local(nlocal);

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
        free_energy_get_chemical_stress(index, plocal);

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

  stats_rheology_print_s("stress_hydro", stress);
  stats_rheology_print_s("stress_pchem", pchem);
  stats_rheology_print_s("stress_rhouu", rhouu);

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
