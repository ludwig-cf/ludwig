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
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "gradient_3d_27pt_solid.h"

struct solid_s {
  map_t * map;
  fe_symm_t * fe_symm;
};

static struct solid_s static_solid = {0};

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

__host__ int grad_3d_27pt_solid_op(field_grad_t * fg, map_t * map,
				   double kappa, int nextra);

/*****************************************************************************
 *
 *  grad_3d_27pt_solid_map_set
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_solid_map_set(map_t * map) {

  int ndata;
  assert(map);

  static_solid.map = map;

  /* We expect at most two wetting parameters; if present
   * first should be C, second H. Default to zero. */

  map_ndata(map, &ndata);
  if (ndata > 2) fatal("Two many wetting parameters for gradient %d\n", ndata);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_solid_fe_set
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_solid_fe_set(fe_symm_t * fe) {

  assert(fe);

  static_solid.fe_symm = fe;

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_solid_d2
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_solid_d2(field_grad_t * fgrad) {

  int nextra;
  fe_symm_param_t param;

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);
  assert(static_solid.map);
  assert(static_solid.fe_symm);

  fe_symm_param(static_solid.fe_symm, &param);

  grad_3d_27pt_solid_op(fgrad, static_solid.map, param.kappa, nextra);

  return 0;
}

/****************************************************************************
 *
 *  grad_3d_27pt_solid_op
 *
 *  This calculation can be extended into the halo region by
 *  nextra points in each direction.
 *
 *  kappa is the interfacial energy penalty in the symmetric picture.
 *
 ****************************************************************************/

__host__ int grad_3d_27pt_solid_op(field_grad_t * fg, map_t * map,
				   double kappa,
				   int nextra) {

  int nop;
  int nlocal[3];
  int ic, jc, kc, ic1, jc1, kc1;
  int ia, index, p;
  int n;
  int nsites;

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

  double * __restrict__ field;

  coords_nlocal(nlocal);

  nop = fg->field->nf;
  nsites = fg->field->nsites;
  field = fg->field->data;
  rk = 1.0/kappa;

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

	    dphi
	      = field[addr_rank1(nsites, nop, isite[p], n)]
	      - field[addr_rank1(nsites, nop, index,    n)];
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
	      phi_b = field[addr_rank1(nsites, nop, index, n)]
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

	  fg->delsq[addr_rank1(nsites, nop, index, n)] = r9*dphi;
	  for (ia = 0; ia < 3; ia++) {
	    fg->grad[addr_rank2(nsites,nop,3,index,n,ia)] = r18*gradn[ia];
	  }
	}

	/* Next site */
      }
    }
  }

  return 0;
}
