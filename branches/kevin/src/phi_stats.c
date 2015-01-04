/*****************************************************************************
 *
 *  phi_stats.c
 *
 *  Order parameter statistics.
 *
 *  $Id: phi_stats.c,v 1.9 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>


#include "util.h"
#include "field_s.h"
#include "phi_stats.h"

/*****************************************************************************
 *
 *  stats_field_info
 *
 *****************************************************************************/

int stats_field_info(field_t * obj, map_t * map) {

  int n, nf;
  MPI_Comm comm;

  double fmin[NQAB];
  double fmax[NQAB];
  double fsum[NQAB];
  double fvar[NQAB];
  double fvol, rvol;
  double fbar, f2;

  assert(obj);
  assert(map);

  field_nf(obj, &nf);
  assert(nf <= NQAB);

  comm = pe_comm();
  stats_field_reduce(obj, map, fmin, fmax, fsum, fvar, &fvol, 0, comm);

  rvol = 1.0 / fvol;

  for (n = 0; n < nf; n++) {

    fbar = rvol*fsum[n];                 /* mean */
    f2   = rvol*fvar[n]  - fbar*fbar;    /* variance */

    info("[phi] %14.7e %14.7e%14.7e %14.7e%14.7e\n", fsum[n], fbar, f2,
	 fmin[n], fmax[n]);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_field_info_bbl
 *
 *  With correction for bbl for conserved order parameters (largely
 *  for binary fluid).
 *
 *****************************************************************************/

int stats_field_info_bbl(field_t * obj, map_t * map, bbl_t * bbl) {

  int n, nf;
  MPI_Comm comm;

  double fmin[NQAB];
  double fmax[NQAB];
  double fsum[NQAB];
  double fvar[NQAB];
  double fbbl[NQAB];
  double fbbl_local[NQAB];
  double fvol, rvol;
  double fbar, f2;

  assert(obj);
  assert(map);
  assert(bbl);

  field_nf(obj, &nf);
  assert(nf <= NQAB);

  comm = pe_comm();
  stats_field_reduce(obj, map, fmin, fmax, fsum, fvar, &fvol, 0, comm);

  /* BBL corrections to be added */
  for (n = 0; n < nf; n++) {
    fbbl_local[n] = 0.0;
  }
  bbl_order_parameter_deficit(bbl, fbbl_local);
  MPI_Reduce(fbbl_local, fbbl, nf, MPI_DOUBLE, MPI_SUM, 0, comm);
  for (n = 0; n < nf; n++) {
    fsum[n] += fbbl[n];
  }

  rvol = 1.0 / fvol;

  for (n = 0; n < nf; n++) {

    fbar = rvol*fsum[n];                 /* mean */
    f2   = rvol*fvar[n]  - fbar*fbar;    /* variance */

    info("[phi] %14.7e %14.7e%14.7e %14.7e%14.7e\n", fsum[n], fbar, f2,
	 fmin[n], fmax[n]);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_field_reduce
 *
 *  This is a global reduction to rank in communicator comm.
 *
 *  We expect and assert NQAB to be the largest number of field elements
 *  to avoid memory allocation and deallocation here.
 *
 *****************************************************************************/

int stats_field_reduce(field_t * obj, map_t * map, double * fmin,
		       double * fmax,  double * fsum, double * fvar,
		       double * fvol, int rank, MPI_Comm comm) {
  int nf;

  double fmin_local[NQAB];
  double fmax_local[NQAB];
  double fsum_local[NQAB];
  double fvar_local[NQAB];
  double fvol_local[1];

  assert(obj);
  assert(map);

  field_nf(obj, &nf);
  assert(nf <= NQAB);

  stats_field_local(obj, map, fmin_local, fmax_local, fsum_local, fvar_local,
		    fvol_local);

  MPI_Reduce(fmin_local, fmin, nf, MPI_DOUBLE, MPI_MIN, rank, comm);
  MPI_Reduce(fmax_local, fmax, nf, MPI_DOUBLE, MPI_MAX, rank, comm);
  MPI_Reduce(fsum_local, fsum, nf, MPI_DOUBLE, MPI_SUM, rank, comm);
  MPI_Reduce(fvar_local, fvar, nf, MPI_DOUBLE, MPI_SUM, rank, comm);
  MPI_Reduce(fvol_local, fvol,  1, MPI_DOUBLE, MPI_SUM, rank, comm);

  return 0;
}

/*****************************************************************************
 *
 *  stats_field_local
 *
 *  Accumulate the local statistics for each field scalar:
 *
 *   fmin[]  minimum
 *   fmax[]  maximum
 *   fsum[]  the sum
 *   fvar[]  the sum of the squares used to compute the variance
 *   fvol    volume of fluid required to get the mean
 *
 *   Each of the arrays must be large enough to hold the value for
 *   a field with nf elements.
 *
 *****************************************************************************/

int stats_field_local(field_t * obj, map_t * map, double * fmin, double * fmax,
		      double * fsum, double * fvar, double * fvol) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nf;
  int status;

  double f0[NQAB];

  assert(obj);
  assert(fmin);
  assert(fmax);
  assert(fsum);
  assert(map);

  coords_nlocal(obj->cs, nlocal);
  field_nf(obj, &nf);
  assert(nf <= NQAB);

  *fvol = 0.0;

  for (n = 0; n < nf; n++) {
    fmin[n] = +DBL_MAX;
    fmax[n] = -DBL_MAX;
    fsum[n] = 0.0;
    fvar[n] = 0.0;
  }

  /* Local sum */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(obj->cs, ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	*fvol += 1.0;
	field_scalar_array(obj, index, f0);

	for (n = 0; n < nf; n++) {
	  fmin[n] = dmin(fmin[n], f0[n]);
	  fmax[n] = dmax(fmax[n], f0[n]);
	  fsum[n] += f0[n];
	  fvar[n] += f0[n]*f0[n];
	}

      }
    }
  }

  return 0;
}
