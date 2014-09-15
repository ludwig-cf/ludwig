/*****************************************************************************
 *
 *  psi_stats.c
 *
 *  Statistics for the electrokintic quantities.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdlib.h>


#include "pe.h"
#include "coords.h"
#include "util.h"
#include "psi.h"
#include "psi_stats.h"

/*****************************************************************************
 *
 *  psi_stats_info
 *
 *****************************************************************************/

int psi_stats_info(psi_t * obj) {

  int n, nk, nrho;
  double * rho_min;
  double * rho_max;
  double * rho_tot;

  MPI_Comm comm;

  assert(obj);

  psi_nk(obj, &nk);
  nrho = 2 + nk;
  comm = pe_comm();

  rho_min = calloc(nrho, sizeof(double));
  rho_max = calloc(nrho, sizeof(double));
  rho_tot = calloc(nrho, sizeof(double));
  if (rho_min == NULL) fatal("calloc(rho_min) failed\n");
  if (rho_max == NULL) fatal("calloc(rho_max) failed\n");
  if (rho_tot == NULL) fatal("calloc(rho_tot) failed\n");

  /* Reduce to rank 0 in pe_comm for info */

  psi_stats_reduce(obj, rho_min, rho_max, rho_tot, 0, comm);

  info("[psi] %14.7e %14.7e %14.7e\n", rho_min[0], rho_max[0], rho_tot[0]);
  for (n = 0; n < nk; n++) {
    info("[rho] %14.7e %14.7e %14.7e\n", rho_min[1+n], rho_max[1+n],
	 rho_tot[1+n]);
  }
  info("[elc] %14.7e %14.7e %14.7e\n", rho_min[1+nk], rho_max[1+nk],
       rho_tot[1+nk]);

  free(rho_tot);
  free(rho_max);
  free(rho_min);

  return 0;
}

/*****************************************************************************
 *
 *  psi_stats_reduce
 *
 *  Reduce local contributions to psi totals to rank in comm.
 *  The user must supply rho_min, rho_max, and rho_tot of at
 *  least psi->nk + 2 in length, which are the returned global
 *  quanitities on rank.
 *
 *  The returned values are not meaningful on other ranks.
 *  Collective call in communicator comm.
 *
 *****************************************************************************/

int psi_stats_reduce(psi_t * obj, double * rho_min, double * rho_max,
		     double * rho_tot, int rank, MPI_Comm comm) {
  int nk, nrho;
  double * rho_min_local;
  double * rho_max_local;
  double * rho_tot_local;

  assert(obj);
  assert(rho_min);
  assert(rho_max);
  assert(rho_tot);

  psi_nk(obj, &nk);
  nrho = 2 + nk;

  rho_min_local = calloc(nrho, sizeof(double));
  rho_max_local = calloc(nrho, sizeof(double));
  rho_tot_local = calloc(nrho, sizeof(double));
  if (rho_min_local == NULL) fatal("calloc(rho_min_local) failed\n");
  if (rho_max_local == NULL) fatal("calloc(rho_max_local) failed\n");
  if (rho_tot_local == NULL) fatal("calloc(rho_tot_local) failed\n");

  psi_stats_local(obj, rho_min_local, rho_max_local, rho_tot_local);

  MPI_Reduce(rho_min_local, rho_min, nrho, MPI_DOUBLE, MPI_MIN, rank, comm);
  MPI_Reduce(rho_max_local, rho_max, nrho, MPI_DOUBLE, MPI_MAX, rank, comm);
  MPI_Reduce(rho_tot_local, rho_tot, nrho, MPI_DOUBLE, MPI_SUM, rank, comm);

  free(rho_tot_local);
  free(rho_max_local);
  free(rho_min_local);

  return 0;
}

/*****************************************************************************
 *
 *  psi_stats_local
 *
 *  The min,max etc for the potential, individual charge densities,
 *  and total charge rho_elec are computed locally.
 *
 *  Each supplied array must be at least of size 2 + psi->nk to hold
 *  the relevant quantities.
 *
 *  These values for the potential may not be very meaningful, but
 *  they are included for completeness.
 *
 *  The values for the charge densities do not include the
 *  unit charge or the valency, so should all be positive.
 *
 *****************************************************************************/

int psi_stats_local(psi_t * obj, double * rho_min, double * rho_max,
		    double * rho_tot) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nk;

  int nrho;
  double psi, rho, rho_elec;

  assert(obj);
  assert(rho_min);
  assert(rho_max);
  assert(rho_tot);

  coords_nlocal(nlocal);
  psi_nk(obj, &nk);
  nrho = 2 + nk;

  for (n = 0; n < nrho; n++) {
    rho_min[n] = DBL_MAX;
    rho_max[n] = -DBL_MAX;
    rho_tot[n] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi(obj, index, &psi);
	rho_min[0] = dmin(psi, rho_min[0]);
	rho_max[0] = dmax(psi, rho_max[0]);
	rho_tot[0] += psi;

	for (n = 0; n < nk; n++) {
	  psi_rho(obj, index, n, &rho);
	  rho_min[1+n] = dmin(rho, rho_min[1+n]);
	  rho_max[1+n] = dmax(rho, rho_max[1+n]);
	  rho_tot[1+n] += rho;
	}

	psi_rho_elec(obj, index, &rho_elec);
	rho_min[1+nk] = dmin(rho_elec, rho_min[1+nk]);
	rho_max[1+nk] = dmax(rho_elec, rho_max[1+nk]);
	rho_tot[1+nk] += rho_elec;

      }
    }
  }

  return 0;
}
