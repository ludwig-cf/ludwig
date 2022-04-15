/*****************************************************************************
 *
 *  leslie_ericksen.c
 *
 *  Updates a vector order parameter according to something looking
 *  like a Leslie-Ericksen equation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "advection_s.h"
#include "leslie_ericksen.h"

static double Gamma_;       /* Rotational diffusion constant */
static double swim_ = 0.0;  /* Self-advection parameter */

static int leslie_ericksen_update_fluid(fe_polar_t *fe,
					field_t * p, hydro_t * hydro,
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

int leslie_ericksen_update(cs_t * cs, fe_polar_t * fe, field_t * p,
			   hydro_t * hydro) {

  int nf;
  advflux_t * flux = NULL;

  assert(cs);
  assert(p);

  field_nf(p, &nf);
  assert(nf == NVECTOR);
  advflux_cs_create(p->pe, cs, nf, &flux);

  if (hydro) {
    if (swim_ != 0.0) leslie_ericksen_add_swimming_velocity(p, hydro);
    hydro_u_halo(hydro);
    advflux_cs_compute(flux, hydro, p);
  }

  leslie_ericksen_update_fluid(fe, p, hydro, flux);

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

static int leslie_ericksen_update_fluid(fe_polar_t * fe,
					field_t * fp,
					hydro_t * hydro,
					advflux_t * flux) {
  int ic, jc, kc, index;
  int im1, jm1, km1;
  int ia, ib;
  int nlocal[3];

  double p[3];
  double h[3];
  double d[3][3];
  double omega[3][3];
  double w[3][3];
  double sum;
  fe_polar_param_t param;
  const double dt = 1.0;

  assert(fe);
  assert(fp);
  assert(flux);

  fe_polar_param(fe, &param);
  cs_nlocal(flux->cs, nlocal);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      w[ia][ib] = 0.0;
    }
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(flux->cs, ic, jc, kc);
	field_vector(fp, index, p);
	fe_polar_mol_field(fe, index, h);
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

	/* updates involve the following fluxes */

        im1 = cs_index(flux->cs, ic-1, jc, kc);
	jm1 = cs_index(flux->cs, ic, jc-1, kc);
	km1 = cs_index(flux->cs, ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {

	  sum = 0.0;
	  for (ib = 0; ib < 3; ib++) {
	    sum += param.lambda*d[ia][ib]*p[ib] - omega[ia][ib]*p[ib];
	  }

	  p[ia] += dt*(- flux->fx[addr_rank1(flux->nsite, 3, index, ia)]
		       + flux->fx[addr_rank1(flux->nsite, 3, im1,   ia)]
		       - flux->fy[addr_rank1(flux->nsite, 3, index, ia)]
		       + flux->fy[addr_rank1(flux->nsite, 3, jm1,   ia)]
		       - flux->fz[addr_rank1(flux->nsite, 3, index, ia)]
		       + flux->fz[addr_rank1(flux->nsite, 3, km1,   ia)]
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

  cs_nlocal(fp->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(fp->cs, ic, jc, kc);
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
