/*****************************************************************************
 *
 *  phi_stats.c
 *
 *  Order parameter statistics.
 *
 *  $Id: phi_stats.c,v 1.8.4.4 2010-05-12 18:19:39 kevin Exp $
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
#include "phi.h"
#include "phi_lb_coupler.h"
#include "phi_stats.h"

/*****************************************************************************
 *
 *  phi_stats_print_stats
 *
 *  Return: the total, the mean, the variance, the maximum, the minimum
 *  of the order parameter.
 *
 *****************************************************************************/

void phi_stats_print_stats() {

  int      index, ic, jc, kc, n;
  int      nlocal[3];
  int      nop;
  double   phi0, phi1;
  double * phi_local;
  double * phi_total;

  coords_nlocal(nlocal);
  nop = phi_nop();

  /* This is required to update phi_site[] when full LB is active */

  phi_compute_phi_site();

  /* Stats: */

  phi_local = (double *) malloc(5*nop*sizeof(double));
  phi_total = (double *) malloc(5*nop*sizeof(double));
  if (phi_local == NULL) fatal("malloc(phi_local) failed\n");
  if (phi_total == NULL) fatal("malloc(phi_total) failed\n");


  for (n = 0; n < nop; n++) {
    phi_local[         n] = 0.0;        /* volume */
    phi_local[1*nop + n] = 0.0;        /* phi    */
    phi_local[2*nop + n] = 0.0;        /* phi^2  */
    phi_local[3*nop + n] = +DBL_MAX;   /* min    */
    phi_local[4*nop + n] = -DBL_MAX;   /* max    */
  }

  if (phi_is_finite_difference()) {
    /* There's no correction coming from BBL */
  }
  else {
    phi_local[1*nop + 0] = bbl_order_parameter_deficit();
  }

  /* Compute the mean phi in the domain proper */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = coords_index(ic, jc, kc);

	for (n = 0; n < nop; n++) {
	  phi0 = phi_op_get_phi_site(index, n);

	  phi_local[         n] += 1.0;
	  phi_local[1*nop + n] += phi0;
	  phi_local[2*nop + n] += phi0*phi0;
	  phi_local[3*nop + n] = dmin(phi0, phi_local[3*nop + n]);
	  phi_local[4*nop + n] = dmax(phi0, phi_local[4*nop + n]);
	}
      }
    }
  }

  MPI_Reduce(phi_local, phi_total, 3*nop, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
  MPI_Reduce(phi_local + 3*nop, phi_total + 3*nop, nop, MPI_DOUBLE,
	     MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(phi_local + 4*nop, phi_total + 4*nop, nop, MPI_DOUBLE,
	     MPI_MAX, 0, MPI_COMM_WORLD);

  /* Mean and variance */

  for (n = 0; n < nop; n++) {

    phi0 = phi_total[1*nop + n]/phi_total[n];
    phi1 = (phi_total[2*nop + n]/phi_total[n]) - phi0*phi0;

    info("[phi] %14.7e %14.7e%14.7e %14.7e%14.7e\n", phi_total[1*nop + n],
	 phi0, phi1, phi_total[3*nop + n], phi_total[4*nop + n]);
  }

  free(phi_local);
  free(phi_total);

  return;
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
	phi_set_phi_site(index, phi);
	phi_lb_coupler_phi_set(index, phi);
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
	phi_set_phi_site(index, phi);

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

  if (phi_nop() == 2) {

    info("Initialising surfactant concentration to %f\n", psi);

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  index = coords_index(ic, jc, kc);
	  phi_op_set_phi_site(index, 1, psi);
	}
      }
    }

  }

  return;
}
