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

#include "pe.h"
#include "coords.h"
#include "site_map.h"
#include "bbl.h"
#include "field.h"
#include "phi_lb_coupler.h"
#include "util.h"
#include "phi_stats.h"

/*****************************************************************************
 *
 *  stats_field_info
 *
 *****************************************************************************/

int stats_field_info(field_t * obj) {

  int n, nf;
  MPI_Comm comm;

  double fmin[NQAB];
  double fmax[NQAB];
  double fsum[NQAB];
  double fvar[NQAB];
  double fvol, rvol;
  double fbar, f2;

  field_nf(obj, &nf);
  assert(nf <= NQAB);

  comm = pe_comm();

  stats_field_reduce(obj, fmin, fmax, fsum, fvar, &fvol, 0, comm);

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

int stats_field_reduce(field_t * obj, double * fmin, double * fmax,
		       double * fsum, double * fvar, double * fvol,
		       int rank, MPI_Comm comm) {
  int nf;

  double fmin_local[NQAB];
  double fmax_local[NQAB];
  double fsum_local[NQAB];
  double fvar_local[NQAB];
  double fvol_local[1];

  field_nf(obj, &nf);
  assert(nf <= NQAB);

  stats_field_local(obj, fmin_local, fmax_local, fsum_local, fvar_local,
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

int stats_field_local(field_t * obj, double * fmin, double * fmax,
		      double * fsum, double * fvar, double * fvol) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nf;

  double f0[NQAB];

  assert(obj);
  assert(fmin);
  assert(fmax);
  assert(fsum);

  coords_nlocal(nlocal);
  field_nf(obj, &nf);
  assert(nf <= NQAB);

  *fvol = 0.0;

  for (n = 0; n < nf; n++) {
    fmin[n] = +DBL_MAX;
    fmax[n] = -DBL_MAX;
    fsum[n] = 0.0;
    fvar[n] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	if (site_map_get_status_index(index) != FLUID) continue;

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

/*****************************************************************************
 *
 *  phi_init_block
 *
 *  Initialise two blocks with interfaces at z = Lz/4 and z = 3Lz/4.
 *
 *****************************************************************************/

void phi_init_block(const double xi0) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z1, z2;
  double phi;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  z1 = 0.25*L(Z);
  z2 = 0.75*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	z = noffset[Z] + kc;

	if (z > 0.5*L(Z)) {
	  phi = tanh((z-z2)/xi0);
	}
	else {
	  phi = -tanh((z-z1)/xi0);
	}

	assert(0);
	/* change interface and implementation; move to symmtric init */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_init_bath
 *
 *  Initialise one interface at z = Lz/8. This is inended for
 *  capillary rise in systems with z not periodic.
 *
 *****************************************************************************/

void phi_init_bath() {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z0;
  double phi, xi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  z0 = 0.25*L(Z);
  xi0 = 1.13;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	z = noffset[Z] + kc;
	phi = tanh((z-z0)/xi0);

	assert(0);
	/* change interface inplementation; move to symmtric init */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_init_surfactant
 *
 *  Initialise a uniform surfactant concentration.
 *
 *****************************************************************************/

void phi_init_surfactant(double psi) {

  int ic, jc, kc, index;
  int nlocal[3];

  coords_nlocal(nlocal);


  info("Initialising surfactant concentration to %f\n", psi);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = coords_index(ic, jc, kc);

	assert(0);
	/* Change interface / implmentation move to surfactant init */
      }
    }
  }

  return;
}
