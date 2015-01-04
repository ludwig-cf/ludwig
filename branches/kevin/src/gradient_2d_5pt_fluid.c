/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid.c
 *
 *  Gradient operations for 2D five point stencil.
 *
 *                    (ic, jc+1)
 *         (ic-1, jc) (ic, jc  ) (ic+1, jc)
 *                    (ic, jc-1)
 *
 *  d_x phi = [phi(ic+1,jc) - phi(ic-1,jc)] / 2
 *  d_y phi = [phi(ic,jc+1) - phi(ic,jc-1)] / 2
 *
 *  nabla^2 phi = phi(ic+1,jc) + phi(ic-1,jc) + phi(ic,jc+1) + phi(ic,jc-1)
 *              - 4 phi(ic,jc)
 *
 *  Corrections for Lees-Edwards planes and plane wall in X are included.
 *
 *  $Id: gradient_2d_5pt_fluid.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2010)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards_s.h"
#include "gradient_2d_5pt_fluid.h"

static void gradient_2d_5pt_fluid_operator(const int nop,
					   const double * field,
					   double * grad,
					   double * delsq,
					   const int nextra);
static void gradient_2d_5pt_fluid_le_correction(const int nop,
						const double * field,
						double * grad,
						double * delsq,
						int nextra);

/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid_d2
 *
 *****************************************************************************/

int gradient_2d_5pt_fluid_d2(const int nop, const double * field, double * t_field,
			     double * grad, double * t_grad, double * delsq,double * t_delsq, char * siteMask,char * t_siteMask) {

  int nhalo;
  int nextra;

  /* PENDING */
  coords_nhalo(le_stat->cs, &nhalo);
  nextra = nhalo - 1;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_2d_5pt_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_2d_5pt_fluid_le_correction(nop, field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid_d4
 *
 *****************************************************************************/

int gradient_2d_5pt_fluid_d4(const int nop, const double * field,double * t_field,
			     double * grad,double * t_grad, double * delsq, double * t_delsq, char * siteMask,char * t_siteMask) {

  int nhalo;
  int nextra;

  coords_nhalo(le_stat->cs, &nhalo);
  nextra = nhalo - 2;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_2d_5pt_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_2d_5pt_fluid_le_correction(nop, field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid_operator
 *
 *****************************************************************************/

static void gradient_2d_5pt_fluid_operator(const int nop,
					   const double * field,
					   double * grad,
					   double * del2,
					   const int nextra) {
  int nlocal[3];
  int nhalo;
  int n;
  int ic, jc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  /* PENDING */
  coords_t * cs;
  cs = le_stat->cs;

  coords_nlocal(cs, nlocal);
  coords_nhalo(cs, &nhalo);

  ys = nlocal[Z] + 2*nhalo;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = le_site_index(ic, jc, 1);
      indexm1 = le_site_index(icm1, jc, 1);
      indexp1 = le_site_index(icp1, jc, 1);

      for (n = 0; n < nop; n++) {
	grad[3*(nop*index + n) + X]
	  = 0.5*(field[nop*indexp1 + n] - field[nop*indexm1 + n]);
	grad[3*(nop*index + n) + Y]
	  = 0.5*(field[nop*(index + ys) + n] - field[nop*(index - ys) + n]);
	grad[3*(nop*index + n) + Z] = 0.0;
	del2[nop*index + n]
	  = field[nop*indexp1 + n] + field[nop*indexm1 + n]
	  + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	  - 4.0*field[nop*index + n];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_2d_5pt_le_correction
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

static void gradient_2d_5pt_fluid_le_correction(const int nop,
						const double * field,
						double * grad,
						double * del2,
						const int nextra) {
  int nlocal[3];
  int nhalo;
  int nh;                                 /* counter over halo extent */
  int n;
  int nplane;                             /* Number LE planes */
  int ic, jc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */

  /* PENDING */
  coords_t * cs;
  cs = le_stat->cs;

  coords_nlocal(cs, nlocal);
  coords_nhalo(cs, &nhalo);

  assert(nlocal[Z] == 1);

  ys = (nlocal[Z] + 2*nhalo);

  for (nplane = 0; nplane < le_get_nplane_local(); nplane++) {

    ic = le_plane_location(nplane);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(ic, nh-1);
      ic1 = le_index_real_to_buffer(ic, nh  );
      ic2 = le_index_real_to_buffer(ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = le_site_index(ic0, jc, 1);
	index   = le_site_index(ic1, jc, 1);
	indexp1 = le_site_index(ic2, jc, 1);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + X]
	    = 0.5*(field[nop*indexp1 + n] - field[nop*indexm1 + n]);
	  grad[3*(nop*index + n) + Y]
	    = 0.5*(field[nop*(index + ys) + n] - field[nop*(index - ys) + n]);
	  grad[3*(nop*index + n) + Z] = 0.0;
	  del2[nop*index + n]
	    = field[nop*indexp1 + n] + field[nop*indexm1 + n]
	    + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	    - 4.0*field[nop*index + n];
	}
      }
    }

    /* Looking across the plane in the -ve x-direction. */
    ic += 1;

    for (nh = 1; nh <= nextra; nh++) {
      ic2 = le_index_real_to_buffer(ic, -nh+1);
      ic1 = le_index_real_to_buffer(ic, -nh  );
      ic0 = le_index_real_to_buffer(ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = le_site_index(ic0, jc, 1);
	index   = le_site_index(ic1, jc, 1);
	indexp1 = le_site_index(ic2, jc, 1);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + X]
	    = 0.5*(field[nop*indexp1 + n] - field[nop*indexm1 + n]);
	  grad[3*(nop*index + n) + Y]
	    = 0.5*(field[nop*(index + ys) + n] - field[nop*(index - ys) + n]);
	  grad[3*(nop*index + n) + Z] = 0.0;
	  del2[nop*index + n]
	    = field[nop*indexp1 + n] + field[nop*indexm1 + n]
	    + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	    - 4.0*field[nop*index + n];
	}
      }
    }
    /* Next plane */
  }

  return;
}
