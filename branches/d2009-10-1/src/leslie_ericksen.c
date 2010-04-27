/*****************************************************************************
 *
 *  leslie_ericksen.c
 *
 *  Updates a vector order parameter according to something looking
 *  like a Leslie-Ericksen equation.
 *
 *  $Id: leslie_ericksen.c,v 1.1.2.5 2010-04-27 13:21:44 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "lattice.h"
#include "free_energy_vector.h"
#include "phi.h"
#include "advection.h"
#include "leslie_ericksen.h"

static double Gamma_;       /* Rotational diffusion constant */
static double swim_ = 0.0;  /* Self-advection parameter */

static double * fluxe;
static double * fluxw;
static double * fluxy;
static double * fluxz;

static void leslie_ericksen_update_fluid(void);
static void leslie_ericksen_add_swimming_velocity(void);

/*****************************************************************************
 *
 *  leslie_ericken_gamma_set
 *
 *****************************************************************************/

void leslie_ericksen_gamma_set(const double gamma) {

  Gamma_ = gamma;
  return;
}

/*****************************************************************************
 *
 *  leslie_ericksen_swim_set
 *
 *****************************************************************************/

void leslie_ericksen_swim_set(const double s) {

  swim_ = s;
  return;
}

/*****************************************************************************
 *
 *  leslie_ericksen_update
 *
 *  Note there is a side effect on the velocity field here if the
 *  self-advection term is not zero.
 *
 *****************************************************************************/

void leslie_ericksen_update(void) {

  int nsites;

  assert(phi_nop() == 3); /* Vector order parameters only */

  nsites = coords_nsites();

  fluxe = (double *) malloc(3*nsites*sizeof(double));
  fluxw = (double *) malloc(3*nsites*sizeof(double));
  fluxy = (double *) malloc(3*nsites*sizeof(double));
  fluxz = (double *) malloc(3*nsites*sizeof(double));
  if (fluxe == NULL) fatal("malloc(fluxe) failed\n");
  if (fluxw == NULL) fatal("malloc(fluxw) failed\n");
  if (fluxy == NULL) fatal("malloc(fluxy) failed\n");
  if (fluxz == NULL) fatal("malloc(fluxz) failed\n");

  if (swim_ != 0.0) leslie_ericksen_add_swimming_velocity();
  hydrodynamics_halo_u();
  advection_upwind_third_order(fluxe, fluxw, fluxy, fluxz);
  leslie_ericksen_update_fluid();

  free(fluxz);
  free(fluxy);
  free(fluxw);
  free(fluxe);

  return;
}

/*****************************************************************************
 *
 * leslie_ericksen_update_fluid
 *
 *****************************************************************************/

static void leslie_ericksen_update_fluid(void) {

  int ic, jc, kc, index;
  int indexj, indexk;
  int ia, ib;
  int nlocal[3];

  double lambda;
  double p[3];
  double h[3];
  double d[3][3];
  double omega[3][3];
  double w[3][3];
  double sum;

  const double dt = 1.0;

  void (* fe_molecular_field_function)(int index, double h[3]);

  coords_nlocal(nlocal);
  fe_molecular_field_function = fe_v_molecular_field();
  lambda = fe_v_lambda();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	phi_vector(index, p);
	fe_molecular_field_function(index, h);
	hydrodynamics_velocity_gradient_tensor(ic, jc, kc, w);

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    d[ia][ib]     = 0.5*(w[ia][ib] + w[ib][ia]);
	    omega[ia][ib] = 0.5*(w[ia][ib] - w[ib][ia]);
	  }
	}

	/* update */

	indexj = coords_index(ic, jc-1, kc);
	indexk = coords_index(ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {

	  sum = 0.0;
	  for (ib = 0; ib < 3; ib++) {
	    sum += lambda*d[ia][ib]*p[ib] - omega[ia][ib]*p[ib];
	  }

	  p[ia] += dt*(-fluxe[3*index + ia] + fluxw[3*index  + ia]
		       -fluxy[3*index + ia] + fluxy[3*indexj + ia]
		       -fluxz[3*index + ia] + fluxz[3*indexk + ia]
		       + sum + Gamma_*h[ia]);
	}

	phi_vector_set(index, p);

	/* Next lattice site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  leslie_ericksen_add_swimming_velocity
 *
 *****************************************************************************/

static void leslie_ericksen_add_swimming_velocity(void) {

  int ic, jc, kc, index;
  int ia;
  int nlocal[3];

  double p[3];
  double u[3];

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	phi_vector(index, p);
	hydrodynamics_get_velocity(index, u);

	for (ia = 0; ia < 3; ia++) {
	  u[ia] += swim_*p[ia];
	}
	hydrodynamics_set_velocity(index, u);
      }
    }
  }

  return;
}
