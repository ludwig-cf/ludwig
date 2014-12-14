/*****************************************************************************
 *
 *  gradient_3d_27pt_solid.c
 *
 *  Gradient routines when solid objects are present (colloids and/or
 *  general porous media). If there are no solid sites nearby it
 *  reduces to the fluid 27pt stencil.
 *
 *  For scalar order parameters with or without wetting.
 *
 *  This is the 'predictor corrector' method described by Desplat et al.
 *  Comp. Phys. Comm. 134, 273--290 (2000).
 *
 *  Note that fluid free energy and surface free energy parameters
 *  are required for wetting. Fluid parameters are via free_energy.h
 *  and surface paramter via site_map.h.
 *
 *  Explicitly, Desplat et al. assume
 *
 *    -kappa f_s = (1/2) C phi_s^2 + H phi_s
 *
 *  where kappa is the fluid parameter and C and H are surface parameters.
 *  If one only needs a set contact angle, can have C = 0. C only comes
 *  into play when consdiering wetting phase transitions.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "free_energy.h"
#include "gradient_3d_27pt_solid.h"

static map_t * map = NULL;

/* These are the 'links' used to form the gradients at boundaries. */

#define NGRAD_ 27
static const int bs_cv[NGRAD_][3] = {{ 0, 0, 0},
				 {-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},
                                 {-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},
                                 {-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},
                                 { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},
                                 { 0, 0,-1},             { 0, 0, 1},
				 { 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},
				 { 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},
				 { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},
				 { 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}};

static void gradient_3d_27pt_solid_op(const int nop,
				      const double * field,
				      double * gradient,
				      double * delsq,
				      const int nextra);

/*****************************************************************************
 *
 *  gradient_3d_27pt_solid_map_set
 *
 *****************************************************************************/

int gradient_3d_27pt_solid_map_set(map_t * map_in) {

  int ndata;
  assert(map_in);

  map = map_in;

  /* We expect at most two wetting parameters; if present
   * first should be C, second H. Default to zero. */

  map_ndata(map, &ndata);
  if (ndata > 2) fatal("Two many wetting parameters for gradient %d\n", ndata);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_3d_27pt_solid_d2
 *
 *****************************************************************************/

int gradient_3d_27pt_solid_d2(const int nop, const double * field,double * t_field,
				double * grad,double * t_grad, double * delsq, double * t_delsq) {

  int nextra;

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  gradient_3d_27pt_solid_op(nop, field, grad, delsq, nextra);

  return 0;
}

/****************************************************************************
 *
 *  gradient_3d_27pt_solid_op
 *
 *  This calculation can be extended into the halo region by
 *  nextra points in each direction.
 *
 ****************************************************************************/

static void gradient_3d_27pt_solid_op(const int nop, const double * field,
				      double * grad, double * delsq,
				      int nextra) {
  int nlocal[3];
  int ic, jc, kc, ic1, jc1, kc1;
  int ia, index, p;
  int n;

  int isite[NGRAD_];

  double count[NGRAD_];
  double gradt[NGRAD_];
  double gradn[3];
  double dphi;
  double rk;
  double c, h, phi_b;

  int status;
  double wet[2] = {0.0, 0.0};

  const double r9 = (1.0/9.0);     /* normaliser for grad */
  const double r18 = (1.0/18.0);   /* normaliser for delsq */

  coords_nlocal(nlocal);

  rk = 1.0/fe_kappa();

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	/* Set solid/fluid flag to index neighbours */

	for (p = 1; p < NGRAD_; p++) {
	  ic1 = ic + bs_cv[p][X];
	  jc1 = jc + bs_cv[p][Y];
	  kc1 = kc + bs_cv[p][Z];

	  isite[p] = coords_index(ic1, jc1, kc1);
	  map_status(map, isite[p], &status);
	  if (status != MAP_FLUID) isite[p] = -1;
	}

	for (n = 0; n < nop; n++) {

	  for (ia = 0; ia < 3; ia++) {
	    count[ia] = 0.0;
	    gradn[ia] = 0.0;
	  }
	  
	  for (p = 1; p < NGRAD_; p++) {

	    if (isite[p] == -1) continue;
	    dphi = field[nop*isite[p] + n] - field[nop*index + n];
	    gradt[p] = dphi;

	    for (ia = 0; ia < 3; ia++) {
	      gradn[ia] += bs_cv[p][ia]*dphi;
	      count[ia] += bs_cv[p][ia]*bs_cv[p][ia];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    if (count[ia] > 0.0) gradn[ia] /= count[ia];
	  }

	  /* Estimate gradient at boundaries */

	  for (p = 1; p < NGRAD_; p++) {

	    if (isite[p] == -1) {
	      phi_b = field[nop*index + n]
		+ 0.5*(bs_cv[p][X]*gradn[X] + bs_cv[p][Y]*gradn[Y]
		       + bs_cv[p][Z]*gradn[Z]);

	      /* Set gradient phi at boundary following wetting properties */

	      ia = coords_index(ic + bs_cv[p][X], jc + bs_cv[p][Y],
				 kc + bs_cv[p][Z]);
	      map_data(map, ia, wet);
	      c = wet[0];
	      h = wet[1];

	      /* kludge: if nop is 2, set h[1] = 0 */
	      /* This is for Langmuir Hinshelwood */
	      c = (1 - n)*c;
	      h = (1 - n)*h;

	      gradt[p] = -(c*phi_b + h)*rk;
	    }
	  }
 
	  /* Accumulate the final gradients */

	  dphi = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    gradn[ia] = 0.0;
	  }

	  for (p = 1; p < NGRAD_; p++) {
	    dphi += gradt[p];
	    for (ia = 0; ia < 3; ia++) {
	      gradn[ia] += gradt[p]*bs_cv[p][ia];
	    }
	  }

	  delsq[nop*index + n] = r9*dphi;
	  for (ia = 0; ia < 3; ia++) {
	    grad[3*(nop*index + n) + ia]  = r18*gradn[ia];
	  }
	}

	/* Next site */
      }
    }
  }

  return;
}
