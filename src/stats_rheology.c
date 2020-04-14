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
 *  TODO:
 *  mean stress function belongs to lb_t (no averages stored)
 *  cs could be replaced by lees_edw?
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "lb_model_s.h"
#include "util.h"
#include "control.h"
#include "physics.h"
#include "free_energy.h"
#include "stats_rheology.h"

struct stats_rheo_s {
  pe_t * pe;
  cs_t * cs;
  int counter_sxy;
  double * sxy;
  double * stat_xz;
  MPI_Comm comm_yz;
  MPI_Comm comm_y;
  MPI_Comm comm_z;
};

static void stats_rheology_print_s(pe_t * pe, const char *, double s[3][3]);
static void stats_rheology_print_matrix(FILE *, double s[3][3]);

#define NSTAT1 7  /* Number of data items for stess statistics */
#define NSTAT2 22 /* Number of data items for 2-d stress stats
		   * 3 components of velocity, 6 components of
                   * 3 different contributions to the stress;
		   * 1 isotropic chemical stress (placeholder) */

/*****************************************************************************
 *
 *  stats_rheology_create
 *
 *  Set up communicator in YZ plane. For the output strategy to work,
 *  rank zero in the new communicator should correspond to
 *  mpi_cart_coords[Y] = 0 and mpi_cart_coords[Z] = 0.
 *
 *****************************************************************************/

int stats_rheology_create(pe_t * pe, cs_t * cs, stats_rheo_t ** pobj) {

  stats_rheo_t * obj = NULL;
  int nelements;
  int rank;
  int remainder[3];
  int nlocal[3];
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  MPI_Comm comm;

  assert(pe);
  assert(cs);

  obj = (stats_rheo_t *) calloc(1, sizeof(stats_rheo_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(stats_rheo_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  cs_cart_comm(cs, &comm);
  cs_cartsz(cs, mpi_cartsz);
  cs_cart_coords(cs, mpi_cartcoords);

  remainder[X] = 0;
  remainder[Y] = 1;
  remainder[Z] = 1;

  MPI_Cart_sub(comm, remainder, &obj->comm_yz);
  MPI_Comm_rank(obj->comm_yz, &rank);

  if (rank == 0) {
    if (mpi_cartcoords[Y] != 0) {
      pe_fatal(pe, "rank 0 in rheology comm_yz_ not cartcoords[Y] zero\n");
    }
    if (mpi_cartcoords[Z] != 0) {
      pe_fatal(pe, "rank 0 in rheology comm_yz_ not cartcoords[Z] zero\n");
    }
  }

  remainder[X] = 0;
  remainder[Y] = 1;
  remainder[Z] = 0;

  MPI_Cart_sub(comm, remainder, &obj->comm_y);

  remainder[X] = 0;
  remainder[Y] = 0;
  remainder[Z] = 1;

  MPI_Cart_sub(comm, remainder, &obj->comm_z);

  MPI_Comm_rank(obj->comm_y, &rank);

  if (rank != mpi_cartcoords[Y]) {
    pe_fatal(pe, "rank not equal to cart_coords(Y) in rheology stats\n");
  }

  MPI_Comm_rank(obj->comm_z, &rank);

  if (rank != mpi_cartcoords[Z]) {
    pe_fatal(pe, "rank not equal to cart_coords(Z) in rheology stats\n");
  }

  /* sxy_ */

  cs_nlocal(cs, nlocal);

  nelements = NSTAT1*nlocal[X];
  obj->sxy = (double *) malloc(nelements*sizeof(double));
  assert(obj->sxy);
  if (obj->sxy == NULL) pe_fatal(pe, "malloc(sxy) failed\n");

  /* stat_xz_ */

  nelements = NSTAT2*nlocal[X]*nlocal[Z];
  obj->stat_xz = (double *) malloc(nelements*sizeof(double));
  assert(obj->stat_xz);
  if (obj->stat_xz == NULL) pe_fatal(pe, "malloc(stat_xz) failed\n");

  stats_rheology_stress_profile_zero(obj);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  stats_rheology_free
 *
 *****************************************************************************/

int stats_rheology_free(stats_rheo_t * stat) {

  MPI_Comm_free(&stat->comm_yz);
  MPI_Comm_free(&stat->comm_y);
  MPI_Comm_free(&stat->comm_z);
  free(stat->sxy);
  free(stat->stat_xz);
  free(stat);

  return 0;
}

/*****************************************************************************
 *
 *  stats_rheology_free_energy_density_profile
 *
 *  Compute and write to file a profile of the free energy density
 *    fex(x) = mean_yz free_energy_density(x,y,z)
 *
 *****************************************************************************/

int stats_rheology_free_energy_density_profile(stats_rheo_t * stat, fe_t * fe,
					       const char * filename) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  int rank;
  int token = 0;
  const int tag_token = 28125;

  double fed;
  double * fex;
  double * fexmean;
  double raverage;
  double ltot[3];

  FILE * fp_output;
  MPI_Status status;
  MPI_Comm comm;

  assert(stat);

  cs_ltot(stat->cs, ltot);
  cs_nlocal(stat->cs, nlocal);
  cs_nlocal_offset(stat->cs, noffset);
  cs_cartsz(stat->cs, mpi_cartsz);
  cs_cart_coords(stat->cs, mpi_cartcoords);
  cs_cart_comm(stat->cs, &comm);

  fex = (double *) malloc(nlocal[X]*sizeof(double));
  assert(fex);
  if (fex == NULL) pe_fatal(stat->pe, "malloc(fex) failed\n");
  fexmean = (double *) malloc(nlocal[X]*sizeof(double));
  assert(fexmean);
  if (fexmean == NULL) pe_fatal(stat->pe, "malloc(fexmean failed\n");

  /* Accumulate the local average over y,z */
  /* Do the reduction in (y,z) to give local f(x) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    fex[ic-1] = 0.0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(stat->cs, ic, jc, kc);
	fe->func->fed(fe, index, &fed);
	fex[ic-1] += fed; 
      }
    }
  }

  MPI_Reduce(fex, fexmean, nlocal[X], MPI_DOUBLE, MPI_SUM, 0, stat->comm_yz);

  /* Write f(x) to file in order */

  raverage = 1.0/(ltot[Y]*ltot[Z]);

  if (mpi_cartcoords[Y] == 0 && mpi_cartcoords[Z] == 0) {

    if (mpi_cartcoords[X] == 0) {
      /* First to write ... */
      fp_output = fopen(filename, "w");
    }
    else {
      /* Block unitl the token is received from the previous process,
       * then reopen the file */

      rank = cs_cart_neighb(stat->cs, CS_BACK, X);
      MPI_Recv(&token, 1, MPI_INT, rank, tag_token, comm, &status);
      fp_output = fopen(filename, "a");
    }

    if (fp_output == NULL) pe_fatal(stat->pe, "fopen(%s) failed\n", filename);

    for (ic = 1; ic <= nlocal[X]; ic++) {
      /* This is an average over (y,z) so don't care about the
       * Lees Edwards planes (but remember to take average!) */

      fexmean[ic-1] *= raverage;
      fprintf(fp_output, "%6d %18.12e\n", noffset[X] + ic, fexmean[ic-1]);
    }

    /* Close the file and send the token to next process */

    if (ferror(fp_output)) {
      perror("perror: ");
      pe_fatal(stat->pe, "File error on writing %s\n", filename);
    }
    fclose(fp_output);

    if (mpi_cartcoords[X] < mpi_cartsz[X] - 1) {
      rank = cs_cart_neighb(stat->cs, CS_FORW, X);
      MPI_Ssend(&token, 1, MPI_INT, rank, tag_token, comm);
    }
  }

  free(fex);
  free(fexmean);

  return 0;
}

/*****************************************************************************
 *
 *  stats_rheology_stress_profile_zero
 *
 *  Set the stress profile to zero.
 *
 *****************************************************************************/

int stats_rheology_stress_profile_zero(stats_rheo_t * stat) {

  int ic, kc, n;
  int nlocal[3];

  assert(stat);

  cs_nlocal(stat->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (n = 0; n < NSTAT1; n++) {
      stat->sxy[NSTAT1*(ic-1) + n] = 0.0;
    }
    for (kc = 1; kc <= nlocal[Z]; kc++) {
      for (n = 0; n < NSTAT2; n++) {
	stat->stat_xz[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + n] = 0.0;
      }
    }
  }


  stat->counter_sxy = 0;

  return 0;
}

/*****************************************************************************
 *
 *  stats_rheology_stress_profile_accumulate
 *
 *  Accumulate the contribution to the mean stress profile
 *  for this time step.
 *
 *****************************************************************************/

int stats_rheology_stress_profile_accumulate(stats_rheo_t * stat, lb_t * lb,
					     fe_t * fe,
					     hydro_t * hydro) {

  int ic, jc, kc, index;
  int nlocal[3];
  int ia, ib, ndata;
  double rho, rrho;
  double u[3];
  double s[3][3];

  assert(stat);
  assert(lb);
  assert(hydro);

  cs_nlocal(stat->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(stat->cs, ic, jc, kc);

	/* Set up the (inverse) density, velocity */

	lb_0th_moment(lb, index, LB_RHO, &rho);
	rrho = 1.0/rho;
	lb_1st_moment(lb, index, LB_RHO, u);

	/* Work out the vicous stress and accumulate. The factor
	 * which relates the distribution \sum_i f_ c_ia c_ib
	 * to the viscous stress is not included until output
	 * is required. (See output routine.) */

	lb_2nd_moment(lb, index, LB_RHO, s);

	stat->sxy[NSTAT1*(ic-1)] += s[X][Y];

	ndata = 0;
	for (ia = 0; ia < 3; ia++) {
	  for (ib = ia; ib < 3; ib++) {
	    s[ia][ib] = s[ia][ib] - rrho*u[ia]*u[ib];
	    stat->stat_xz[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + ndata++] += s[ia][ib];
	  }
	}
	assert(ndata == 6);

	/* Thermodynamic part of stress */

	fe->func->stress(fe, index, s);

	stat->sxy[NSTAT1*(ic-1) + 1] += s[X][Y];

	for (ia = 0; ia < 3; ia++) {
	  for (ib = ia; ib < 3; ib++) {
	    stat->stat_xz[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + ndata++] += s[ia][ib];
	  }
	}

	stat->sxy[NSTAT1*(ic-1) + 2] += rrho*u[X]*u[Y];
	stat->sxy[NSTAT1*(ic-1) + 3] += rrho*u[X];
	stat->sxy[NSTAT1*(ic-1) + 4] += rrho*u[Y];
	stat->sxy[NSTAT1*(ic-1) + 5] += rrho*u[Z];

	/* Trivial part. */

	for (ia = 0; ia < 3; ia++) {
	  for (ib = ia; ib < 3; ib++) {
	    stat->stat_xz[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + ndata++]
	      += rrho*u[ia]*u[ib];
	  }
	}

	/* Finally, three components of velocity */

	stat->stat_xz[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + ndata++] += rrho*u[X];
	stat->stat_xz[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + ndata++] += rrho*u[Y];
	stat->stat_xz[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + ndata++] += rrho*u[Z];

	/* Placeholder for isotropic part of chemical stress (set zero) */
	stat->stat_xz[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + ndata++] = 0.0;

	assert(ndata == NSTAT2);

	hydro_u_gradient_tensor(hydro, ic, jc, kc, s);
	stat->sxy[NSTAT1*(ic-1) + 6] += (s[X][Y] + s[Y][X]);
      }
    }
  }

  stat->counter_sxy += 1;

  return 0;
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

int stats_rheology_stress_profile(stats_rheo_t * stat, const char * filename) {

  int ic;
  int nelements;
  int nlocal[3];
  int noffset[3];
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  double * sxymean;
  double rmean;
  double eta;
  double ltot[3];

  const int tag_token = 728;
  int rank;
  int token = 0;

  physics_t * phys = NULL;

  MPI_Status status;
  MPI_Comm comm;
  FILE * fp_output;

  assert(stat);

  cs_ltot(stat->cs, ltot);
  cs_nlocal(stat->cs, nlocal);
  cs_nlocal_offset(stat->cs, noffset);
  cs_cartsz(stat->cs, mpi_cartsz);
  cs_cart_coords(stat->cs, mpi_cartcoords);
  cs_cart_comm(stat->cs, &comm);

  physics_ref(&phys);
  physics_eta_shear(phys, &eta);

  nelements = NSTAT1*nlocal[X];
  sxymean = (double *) malloc(nelements*sizeof(double));
  assert(sxymean);
  if (sxymean == NULL) pe_fatal(stat->pe, "malloc(sxymean) failed\n");

  MPI_Reduce(stat->sxy, sxymean, NSTAT1*nlocal[X], MPI_DOUBLE, MPI_SUM, 0,
	     stat->comm_yz);

  rmean = 1.0/(ltot[Y]*ltot[Z]*stat->counter_sxy);

  /* Write to file in order of x */

  if (mpi_cartcoords[Y] == 0 && mpi_cartcoords[Z] == 0) {

    if (mpi_cartcoords[X] == 0) {
      /* First to write ... */
      fp_output = fopen(filename, "w");
    }
    else {
      /* Block unitl the token is received from the previous process,
       * then reopen the file */

      rank = cs_cart_neighb(stat->cs, CS_BACK, X);
      MPI_Recv(&token, 1, MPI_INT, rank, tag_token, comm, &status);
      fp_output = fopen(filename, "a");
    }

    if (fp_output == NULL) pe_fatal(stat->pe, "fopen(%s) failed\n", filename);

    for (ic = 1; ic <= nlocal[X]; ic++) {
      /* This is an average over (y,z) so don't care about the
       * Lees Edwards places, but correct uy */

      double uy = 0.0; /* Correct in post-processing if wanted at all */

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
      pe_fatal(stat->pe, "File error on writing %s\n", filename);
    }
    fclose(fp_output);

    if (mpi_cartcoords[X] < mpi_cartsz[X] - 1) {
      rank = cs_cart_neighb(stat->cs, CS_FORW, X);
      MPI_Ssend(&token, 1, MPI_INT, rank, tag_token, comm);
    }
  }

  free(sxymean);

  return 0;
}

/*****************************************************************************
 *
 *  stats_rheology_stress_section
 *
 *  Outputs a section in the x-z direction (ie., averaged along the
 *  flow direction of various quantities:
 *
 *  Viscous stress               + eta (d_a u_b + d_b u_a)
 *  Thermodynamic contribution   + P^th_ab
 *  Equilibrium stress           rho u_a u_b
 *  Velocity                     u_x, u_y, x_z
 *  Isotropic part P^th          PENDING
 *
 *  For the 3 tensors, there are included 6 independent components:
 *  S_xx S_xy S_xz S_yy S_yz S_zz.
 *
 *  There are a number of factors and corrections required:
 *    1. The viscous stress must be recovered from the accumulated
 *       measured second moment of the distribution by multiplying
 *       by the factor -3 eta / tau, where tau is the LB relaxation
 *       time (See Kruger etal PRE 2009).
 *    2. The Lees-Edwards planes must be accounted for in the y
 *       component of the velocity.
 *    3. All the statistics are averaged to take account of L_y,
 *       and the number of accumulated time steps.
 *
 *  In the output, the z-direction runs faster, then the x-direction.
 *
 *****************************************************************************/

int stats_rheology_stress_section(stats_rheo_t * stat, const char * filename) {

  int ic, kc;
  int nelements;
  int nlocal[3];
  int ntotal[3];
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  int n, is_writing;
  FILE   * fp_output = NULL;
  double * stat_2d;
  double * stat_1d;
  double raverage;
  double uy = 0.0;
  double eta, viscous;
  double ltot[3];

  physics_t * phys = NULL;

  MPI_Comm comm;
  MPI_Status status;
  int token = 0;
  int rank;
  const int tag_token = 1012;

  assert(stat);

  cs_ltot(stat->cs, ltot);
  cs_nlocal(stat->cs, nlocal);
  cs_ntotal(stat->cs, ntotal);
  cs_cartsz(stat->cs, mpi_cartsz);
  cs_cart_coords(stat->cs, mpi_cartcoords);
  cs_cart_comm(stat->cs, &comm);

  physics_ref(&phys);
  physics_eta_shear(phys, &eta);
  viscous = -rcs2*eta*2.0/(1.0 + 6.0*eta);

  nelements = NSTAT2*nlocal[X]*nlocal[Z];
  stat_2d = (double *) malloc(nelements*sizeof(double));
  assert(stat_2d);
  if (stat_2d == NULL) pe_fatal(stat->pe, "malloc(stat_2d) failed\n");

  nelements = NSTAT2*ntotal[Z];
  stat_1d = (double *) malloc(nelements*sizeof(double));
  assert(stat_1d);
  if (stat_1d == NULL) pe_fatal(stat->pe, "malloc(stat_1d) failed\n");

  /* Set the averaging factor (if no data, set to zero) */

  raverage = 0.0;
  if (stat->counter_sxy > 0) raverage = 1.0/(ltot[Y]*stat->counter_sxy); 

  /* Take the sum in the y-direction and store in stat_2d(x,z) */

  MPI_Reduce(stat->stat_xz, stat_2d, NSTAT2*nlocal[X]*nlocal[Z], MPI_DOUBLE,
	     MPI_SUM, 0, stat->comm_y);

  /* Output now only involves mpi_cartcoords[Y] = 0 */

  if (mpi_cartcoords[Y] == 0) {

    /* The strategy is to gather strip-wise in the z-direction,
     * and only write from the first process in this direction.
     * We then sweep over x to give an xz section. */

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

      /* Correct f1[Y] for leesedwards planes before output. */
      /* Viscous stress likewise. Also take the average here. */

      /* lees_edw_block_uy(le, ic, &uy); TODO */ 

      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (n = 0; n < NSTAT2; n++) {
	  stat_2d[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + n] *= raverage;
	}

	/* The viscous stress must be corrected (first tensor) */
	for (n = 0; n < 6; n++) {
	  stat_2d[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + n] *= viscous;
	}

	/* u_y must be corrected */
	stat_2d[NSTAT2*(nlocal[Z]*(ic-1) + kc-1) + 19] += uy;
      }

      MPI_Gather(stat_2d + NSTAT2*nlocal[Z]*(ic-1), NSTAT2*nlocal[Z],
		 MPI_DOUBLE, stat_1d, NSTAT2*nlocal[Z], MPI_DOUBLE, 0,
		 stat->comm_z);

      /* write data */
      if (is_writing) {
	for (kc = 1; kc <= ntotal[Z]; kc++) {
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
	pe_fatal(stat->pe, "File error on writing %s\n", filename);
      }
      fclose(fp_output);
    }

    if (mpi_cartcoords[X] < mpi_cartsz[X] - 1) {
      rank = cs_cart_neighb(stat->cs, CS_FORW, X);
      MPI_Ssend(&token, 1, MPI_INT, rank, tag_token, comm);
    }
  }

  free(stat_1d);
  free(stat_2d);

  return 0;
}

/*****************************************************************************
 *
 *  stats_rheology_mean_stress
 *
 *  Provide a summary of the instantaneous mean stress to stdout.
 *  We have:
 *
 *  Viscous stress               + eta (d_a u_b + d_b u_a)
 *  Thermodynamic contribution   + P^th_ab
 *  Equilibrium stress           rho u_a u_b
 *
 *  Note that the viscous stress is constructed from the second
 *  moment of the distribution.
 *
 *****************************************************************************/

int stats_rheology_mean_stress(lb_t * lb, fe_t * fe, const char * filename) {

#define NCOMP 27

  int nlocal[3];
  int ic, jc, kc, index, ia, ib;
  double stress[3][3];
  double rhouu[3][3];
  double pchem[3][3], plocal[3][3];
  double s[3][3];
  double u[3];
  double send[NCOMP];
  double recv[NCOMP];
  double rho, rrho, rv;
  double eta, viscous;
  double ltot[3];
  physics_t * phys = NULL;
  FILE * fp;
  MPI_Comm comm;

  assert(lb);

  pe_mpi_comm(lb->pe, &comm);
  cs_ltot(lb->cs, ltot);
  cs_nlocal(lb->cs, nlocal);

  rv = 1.0/(ltot[X]*ltot[Y]*ltot[Z]);

  physics_ref(&phys);
  physics_eta_shear(phys, &eta);
  viscous = -rcs2*eta*2.0/(1.0 + 6.0*eta);

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

        index = cs_index(lb->cs, ic, jc, kc);

	lb_0th_moment(lb, index, LB_RHO, &rho);
	lb_1st_moment(lb, index, LB_RHO, u);
	lb_2nd_moment(lb, index, LB_RHO, s);
        fe->func->stress(fe, index, plocal);

	rrho = 1.0/rho;
        for (ia = 0; ia < 3; ia++) {
          for (ib = 0; ib < 3; ib++) {
            rhouu[ia][ib] += rrho*u[ia]*u[ib];
            pchem[ia][ib] += plocal[ia][ib];
            stress[ia][ib] += (s[ia][ib] - rrho*u[ia]*u[ib]);
          }
        }

      }
    }
  }

  /* Acculumate the sums */

  kc = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      send[kc++] = viscous*stress[ia][ib];
      send[kc++] = pchem[ia][ib];
      send[kc++] = rhouu[ia][ib];
    }
  }

  MPI_Reduce(send, recv, NCOMP, MPI_DOUBLE, MPI_SUM, 0, comm);

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
    stats_rheology_print_s(lb->pe, "stress_viscs", stress);
    stats_rheology_print_s(lb->pe, "stress_pchem", pchem);
    stats_rheology_print_s(lb->pe, "stress_rhouu", rhouu);
  }
  else {
    /* Use filename supplied */

    if (pe_mpi_rank(lb->pe) == 0) {
      fp = fopen(filename, "a");
      if (fp == NULL) pe_fatal(lb->pe, "fopen(%s) failed\n", filename);

      fprintf(fp, "%9d ", physics_control_timestep(phys));
      stats_rheology_print_matrix(fp, stress);
      stats_rheology_print_matrix(fp, pchem);
      stats_rheology_print_matrix(fp, rhouu);
      fprintf(fp, "\n");

      fclose(fp);
    }
  }

  return 0;
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

static void stats_rheology_print_s(pe_t * pe, const char * label, double s[3][3]) {

  pe_info(pe, "%s x %15.8e %15.8e %15.8e\n", label, s[X][X], s[X][Y], s[X][Z]);
  pe_info(pe, "%s y %15.8e %15.8e %15.8e\n", label, s[Y][X], s[Y][Y], s[Y][Z]);
  pe_info(pe, "%s z %15.8e %15.8e %15.8e\n", label, s[Z][X], s[Z][Y], s[Z][Z]);

  return;
}
