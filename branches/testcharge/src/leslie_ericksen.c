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
#include "field.h"
#include "advection_s.h"
#include "leslie_ericksen.h"

static double Gamma_;       /* Rotational diffusion constant */
static double swim_ = 0.0;  /* Self-advection parameter */

static int leslie_ericksen_update_fluid(field_t * p, hydro_t * hydro,
					advflux_t * flux);
static int leslie_ericksen_add_swimming_velocity(field_t * p,
						 hydro_t * hydro);

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

int leslie_ericksen_update(field_t * p, hydro_t * hydro) {

  int nf;
  advflux_t * flux = NULL;

  assert(p);

  field_nf(p, &nf);
  assert(nf == NVECTOR);
  advflux_create(nf, &flux);

  if (hydro) {
    if (swim_ != 0.0) leslie_ericksen_add_swimming_velocity(p, hydro);
    hydro_u_halo(hydro);
    advection_x(flux, hydro, p);
  }

  leslie_ericksen_update_fluid(p, hydro, flux);

  advflux_free(flux);

  return 0;
}

/*****************************************************************************
 *
 * leslie_ericksen_update_fluid
 *
 *  hydro is allowed to be NULL, in which case there is relaxational
 *  dynmaics only.
 *
 *****************************************************************************/

static int leslie_ericksen_update_fluid(field_t * fp, hydro_t * hydro,
					advflux_t * flux) {
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

  assert(fp);
  assert(flux);

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
	field_vector(fp, index, p);
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

	  p[ia] += dt*(-flux->fe[3*index + ia] + flux->fw[3*index  + ia]
		       -flux->fy[3*index + ia] + flux->fy[3*indexj + ia]
		       -flux->fz[3*index + ia] + flux->fz[3*indexk + ia]
		       + sum + Gamma_*h[ia]);
	}

	field_vector_set(fp, index, p);

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

static int leslie_ericksen_add_swimming_velocity(field_t * fp,
						 hydro_t * hydro) {
  int ic, jc, kc, index;
  int ia;
  int nlocal[3];

  double p[3];
  double u[3];

  assert(fp);
  assert(hydro);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	field_vector(fp, index, p);
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
