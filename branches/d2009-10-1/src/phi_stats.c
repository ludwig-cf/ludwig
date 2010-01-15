/*****************************************************************************
 *
 *  phi_stats.c
 *
 *  Order parameter statistics.
 *
 *  $Id: phi_stats.c,v 1.8.4.1 2010-01-15 16:50:48 kevin Exp $
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
  double   phi0, phi1;
  double * phi_local;
  double * phi_total;

  /* This is required to update phi_site[] when full LB is active */
  phi_compute_phi_site();

  /* Stats: */

  phi_local = (double *) malloc(5*nop_*sizeof(double));
  phi_total = (double *) malloc(5*nop_*sizeof(double));
  if (phi_local == NULL) fatal("malloc(phi_local) failed\n");
  if (phi_total == NULL) fatal("malloc(phi_total) failed\n");

  get_N_local(nlocal);

  for (n = 0; n < nop_; n++) {
    phi_local[         n] = 0.0;        /* volume */
    phi_local[1*nop_ + n] = 0.0;        /* phi    */
    phi_local[2*nop_ + n] = 0.0;        /* phi^2  */
    phi_local[3*nop_ + n] = +DBL_MAX;   /* min    */
    phi_local[4*nop_ + n] = -DBL_MAX;   /* max    */
  }

  if (phi_is_finite_difference()) {
    /* There's no correction coming from BBL */
  }
  else {
    phi_local[1*nop_ + 0] = bbl_order_parameter_deficit();
  }

  /* Compute the mean phi in the domain proper */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = get_site_index(ic, jc, kc);

	for (n = 0; n < nop_; n++) {
	  phi0 = phi_op_get_phi_site(index, n);

	  phi_local[         n] += 1.0;
	  phi_local[1*nop_ + n] += phi0;
	  phi_local[2*nop_ + n] += phi0*phi0;
	  phi_local[3*nop_ + n] = dmin(phi0, phi_local[3*nop_ + n]);
	  phi_local[4*nop_ + n] = dmax(phi0, phi_local[4*nop_ + n]);
	}
      }
    }
  }

  MPI_Reduce(phi_local, phi_total, 3*nop_, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
  MPI_Reduce(phi_local + 3*nop_, phi_total + 3*nop_, nop_, MPI_DOUBLE,
	     MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(phi_local + 4*nop_, phi_total + 4*nop_, nop_, MPI_DOUBLE,
	     MPI_MAX, 0, MPI_COMM_WORLD);

  /* Mean and variance */

  for (n = 0; n < nop_; n++) {

    phi0 = phi_total[1*nop_ + n]/phi_total[n];
    phi1 = (phi_total[2*nop_ + n]/phi_total[n]) - phi0*phi0;

    info("[phi] %14.7e %14.7e%14.7e %14.7e%14.7e\n", phi_total[1*nop_ + n],
	 phi0, phi1, phi_total[3*nop_ + n], phi_total[4*nop_ + n]);
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

void phi_init_block() {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z1, z2;
  double phi, xi0;

  get_N_local(nlocal);
  get_N_offset(noffset);

  z1 = 0.25*L(Z);
  z2 = 0.75*L(Z);
  /* This is currently hardwired as the value that is generally
   * used, but may want to change. */
  xi0 = 1.13;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);
	z = noffset[Z] + kc;

	if (z > 0.5*L(Z)) {
	  phi = tanh((z-z2)/xi0);
	  phi_set_phi_site(index, phi);
	}
	else {
	  phi = -tanh((z-z1)/xi0);
	  phi_set_phi_site(index, phi);
	}
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

  get_N_local(nlocal);
  get_N_offset(noffset);

  z0 = 0.25*L(Z);
  xi0 = 1.13;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);
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

  get_N_local(nlocal);

  if (nop_ == 2) {

    info("Initialising surfactant concentration to %f\n", psi);

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  index = get_site_index(ic, jc, kc);
	  phi_op_set_phi_site(index, 1, psi);
	}
      }
    }

  }

  return;
}
