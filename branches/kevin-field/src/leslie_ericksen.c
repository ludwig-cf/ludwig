/*****************************************************************************
 *
 *  leslie_ericksen.c
 *
 *  Updates a vector order parameter according to something looking
 *  like a Leslie-Ericksen equation.
 *
 *  $Id: leslie_ericksen.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
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
#include "free_energy_vector.h"

#ifdef OLD_PHI
#include "phi.h"
#else
#include "field.h"
#endif

#include "advection.h"
#include "leslie_ericksen.h"

static double Gamma_;       /* Rotational diffusion constant */
static double swim_ = 0.0;  /* Self-advection parameter */

static double * fluxe;
static double * fluxw;
static double * fluxy;
static double * fluxz;

static int leslie_ericksen_update_fluid(hydro_t * hydro);
static int leslie_ericksen_add_swimming_velocity(hydro_t * hydro);

/*****************************************************************************
 *
 *  leslie_ericken_gamma_set
 *
 *****************************************************************************/

int leslie_ericksen_gamma_set(const double gamma) {

  Gamma_ = gamma;
  return 0;
}

/*****************************************************************************
 *
 *  leslie_ericksen_swim_set
 *
 *****************************************************************************/

int leslie_ericksen_swim_set(const double s) {

  swim_ = s;
  return 0;
}

/*****************************************************************************
 *
 *  leslie_ericksen_update
 *
 *  The hydro is allowed to be NULL, in which case the dynamics is
 *  purely relaxational.
 *
 *  Note there is a side effect on the velocity field here if the
 *  self-advection term is not zero.
 *
 *****************************************************************************/

int leslie_ericksen_update(hydro_t * hydro) {

  int nsites;

#ifdef OLD_PHI
  assert(phi_nop() == 3); /* Vector order parameters only */
#endif

  nsites = coords_nsites();

  fluxe = calloc(3*nsites, sizeof(double));
  fluxw = calloc(3*nsites, sizeof(double));
  fluxy = calloc(3*nsites, sizeof(double));
  fluxz = calloc(3*nsites, sizeof(double));
  if (fluxe == NULL) fatal("calloc(fluxe) failed\n");
  if (fluxw == NULL) fatal("calloc(fluxw) failed\n");
  if (fluxy == NULL) fatal("calloc(fluxy) failed\n");
  if (fluxz == NULL) fatal("calloc(fluxz) failed\n");

  if (hydro) {
    if (swim_ != 0.0) leslie_ericksen_add_swimming_velocity(hydro);
    hydro_u_halo(hydro);
#ifdef OLD_PHI
    advection_order_n(hydro, fluxe, fluxw, fluxy, fluxz);
#else
    assert(0);
    /* Sort fluxes */
#endif
  }

  leslie_ericksen_update_fluid(hydro);

  free(fluxz);
  free(fluxy);
  free(fluxw);
  free(fluxe);

  return 0;
}

/*****************************************************************************
 *
 * leslie_ericksen_update_fluid
 *
 *****************************************************************************/

static int leslie_ericksen_update_fluid(hydro_t * hydro) {

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

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      w[ia][ib] = 0.0;
    }
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

#ifdef OLD_PHI
	phi_vector(index, p);
#else
	assert(0);
	field_t * test_object = NULL;
	field_vector(test_object, index, p);
#endif
	fe_molecular_field_function(index, h);
	if (hydro) hydro_u_gradient_tensor(hydro, ic, jc, kc, w);

	/* Note that the convection for Leslie Ericksen is that
	 * w_ab = d_a u_b, which is the transpose of what the
	 * above returns. Hence an extra minus sign in the
	 * omega term in the following. */

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    d[ia][ib]     = 0.5*(w[ia][ib] + w[ib][ia]);
	    omega[ia][ib] = -0.5*(w[ia][ib] - w[ib][ia]);
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

#ifdef OLD_PHI
	phi_vector_set(index, p);
#else
	assert(0);
	field_vector_set(test_object, index, p);
#endif

	/* Next lattice site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  leslie_ericksen_add_swimming_velocity
 *
 *****************************************************************************/

static int leslie_ericksen_add_swimming_velocity(hydro_t * hydro) {

  int ic, jc, kc, index;
  int ia;
  int nlocal[3];

  double p[3];
  double u[3];

  assert(hydro);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

#ifdef OLD_PHI
	phi_vector(index, p);
#else
	assert(0);
	/* Update interface */
	field_t * test_object = NULL;
	field_vector(test_object, index, p);
#endif
	hydro_u(hydro, index, u);

	for (ia = 0; ia < 3; ia++) {
	  u[ia] += swim_*p[ia];
	}
	hydro_u_set(hydro, index, u);
      }
    }
  }

  return 0;
}
