/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid.c
 *
 *  Gradient operations for 2D stencil following Tomita Prog. Theor. Phys.
 *  {\bf 85} 47 (1991).
 *
 *  del^2 phi(r) = (1 / 1 + 2epsilon) [
 *                 \sum phi(r + a) + epsilon \sum phi(r + b)
 *                 - 4(1 + epsilon) phi(r) ]
 *
 *  where a represent 4 nearest neighbours, b represent 4 next-nearest
 *  neighbours, and epsilon is a parameter. If epsilon = 0, this
 *  collapses to a five-point stencil. The optimum value epsilon is
 *  reckoned to be epsilon = 0.5.
 *
 *  In the same spirit, the gradient is approximated as:
 *
 *  d_x phi = 0.5*(1 / 1 + 2epsilon) * [
 *            epsilon * ( phi(ic+1, jc-1) - phi(ic-1, jc-1) )
 *                    + ( phi(ic+1, jc  ) - phi(ic-1, jc  ) )
 *          + epsilon * ( phi(ic+1, jc+1) - phi(ic-1, jc+1) ) ]
 *
 *  and likewise for d_y phi. Again, this gives 5-point stencil for
 *  epsilon = 0. For the gradient, the optimum value is epsilon = 0.25,
 *  which gives weights consistent with Wolfram J. Stat. Phys {\bf 45},
 *  471 (1986). See also Shan PRE {\bf 73} 047701 (2006).
 *
 *  Corrections for Lees-Edwards planes and plane wall in X are included.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "wall.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "gradient_2d_tomita_fluid.h"

static const double epsilon_ = 0.5;
static const double epsilon1_ = 0.25;

__host__ int grad_2d_tomita_fluid_operator(lees_edw_t * le, field_grad_t * fg,
					   int nextra);
__host__ int grad_2d_tomita_fluid_le(lees_edw_t * le, field_grad_t * fg,
				     int nextra);

/*****************************************************************************
 *
 *  grad_2d_tomita_fluid_d2
 *
 *  TODO:
 *  The assert(0) indicates no regression test.
 *
 *****************************************************************************/

__host__ int grad_2d_tomita_fluid_d2(field_grad_t * fg) {

  int nextra;
  lees_edw_t * le = NULL;

  assert(fg);
  assert(fg->field);

  le = fg->field->le;
  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  assert(0); /* NO TEST */

  grad_2d_tomita_fluid_operator(le, fg, nextra);
  grad_2d_tomita_fluid_le(le, fg, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_tomita_fluid_d4
 *
 *  TODO:
 *  The assert(0) indicates no test available.
 *
 *****************************************************************************/

__host__ int grad_2d_tomita_fluid_d4(field_grad_t * fg) {

  int nextra;
  lees_edw_t * le = NULL;

  nextra = coords_nhalo() - 2;
  assert(nextra >= 0);

  assert(0); /* NO TEST */

  grad_2d_tomita_fluid_operator(le, fg, nextra);
  grad_2d_tomita_fluid_le(le, fg, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_tomita_fluid_operator
 *
 *****************************************************************************/

__host__ int grad_2d_tomita_fluid_operator(lees_edw_t * le, field_grad_t * fg,
					   int nextra) {

  int nop;
  int nlocal[3];
  int nhalo;
  int n;
  int ic, jc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  const double rfactor = 1.0 / (1.0 + 2.0*epsilon_);
  const double rfactor1 = 1.0 / (1.0 + 2.0*epsilon1_);

  double * __restrict__ field;
  double * __restrict__ grad;
  double * __restrict__ del2;

  assert(le);
  assert(fg);

  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);

  ys = nlocal[Z] + 2*nhalo;

  nop = fg->field->nf;
  field = fg->field->data;
  grad = fg->grad;
  del2 = fg->delsq;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = lees_edw_index(le, ic, jc, 1);
      indexm1 = lees_edw_index(le, icm1, jc, 1);
      indexp1 = lees_edw_index(le, icp1, jc, 1);

      for (n = 0; n < nop; n++) {
	grad[3*(nop*index + n) + X] = 0.5*rfactor1*
	  (field[nop*indexp1 + n] - field[nop*indexm1 + n]
	   + epsilon1_*
	   (field[nop*(indexp1 + ys) + n] - field[nop*(indexm1 + ys) + n] +
	    field[nop*(indexp1 - ys) + n] - field[nop*(indexm1 - ys) + n]));
	grad[3*(nop*index + n) + Y] = 0.5*rfactor1*
	  (field[nop*(index + ys) + n] - field[nop*(index - ys) + n]
	   + epsilon1_*
	   (field[nop*(indexp1 + ys) + n] - field[nop*(indexp1 - ys) + n] +
	    field[nop*(indexm1 + ys) + n] - field[nop*(indexm1 - ys) + n]));
	grad[3*(nop*index + n) + Z] = 0.0;
	del2[nop*index + n] =
	  rfactor*(field[nop*indexp1 + n] +
		   field[nop*indexm1 + n] +
		   field[nop*(index + ys) + n] +
		   field[nop*(index - ys) + n] +
		   epsilon_*(field[nop*(indexp1 + ys) + n] +
			     field[nop*(indexp1 - ys) + n] +
			     field[nop*(indexm1 + ys) + n] +
			     field[nop*(indexm1 - ys) + n])
		   - 4.0*(1.0 + epsilon_)*field[nop*index + n]);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_tomita_le
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

__host__ int grad_2d_tomita_fluid_le(lees_edw_t * le, field_grad_t * fg,
				     int nextra) {
  int nop;
  int nlocal[3];
  int nhalo;
  int nh;                                 /* counter over halo extent */
  int n;
  int nplane;                             /* Number LE planes */
  int ic, jc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */

  const double rfactor = 1.0 / (1.0 + 2.0*epsilon_);

  double * __restrict__ field;
  double * __restrict__ grad;
  double * __restrict__ del2;

  assert(le);
  assert(fg);

  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);

  assert(nlocal[Z] == 1);

  ys = (nlocal[Z] + 2*nhalo);

  nop = fg->field->nf;
  field = fg->field->data;
  grad = fg->grad;
  del2 = fg->delsq;

  for (nplane = 0; nplane < lees_edw_nplane_local(le); nplane++) {

    ic = lees_edw_plane_location(le, nplane);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = lees_edw_ic_to_buff(le, ic, nh-1);
      ic1 = lees_edw_ic_to_buff(le, ic, nh  );
      ic2 = lees_edw_ic_to_buff(le, ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = lees_edw_index(le, ic0, jc, 1);
	index   = lees_edw_index(le, ic1, jc, 1);
	indexp1 = lees_edw_index(le, ic2, jc, 1);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + X] = 0.5*rfactor*
	    (field[nop*indexp1 + n] - field[nop*indexm1 + n]
	     + epsilon_*
	     (field[nop*(indexp1 + ys) + n] - field[nop*(indexm1 + ys) + n] +
	      field[nop*(indexp1 - ys) + n] - field[nop*(indexm1 - ys) + n]));
	  grad[3*(nop*index + n) + Y] = 0.5*rfactor*
	    (field[nop*(index + ys) + n] - field[nop*(index - ys) + n]
	     + epsilon_*
	     (field[nop*(indexp1 + ys) + n] - field[nop*(indexp1 - ys) + n] +
	      field[nop*(indexm1 + ys) + n] - field[nop*(indexm1 - ys) + n]));
	  grad[3*(nop*index + n) + Z] = 0.0;

	  del2[nop*index + n] =
	    rfactor*(field[nop*indexp1 + n] +
		     field[nop*indexm1 + n] +
		     field[nop*(index + ys) + n] +
		     field[nop*(index - ys) + n] +
		     epsilon_*(field[nop*(indexp1 + ys) + n] +
			       field[nop*(indexp1 - ys) + n] +
			       field[nop*(indexm1 + ys) + n] +
			       field[nop*(indexm1 - ys) + n])
		     - 4.0*(1.0 + epsilon_)*field[nop*index + n]);
	}
      }
    }

    /* Looking across the plane in the -ve x-direction. */
    ic += 1;

    for (nh = 1; nh <= nextra; nh++) {
      ic2 = lees_edw_ic_to_buff(le, ic, -nh+1);
      ic1 = lees_edw_ic_to_buff(le, ic, -nh  );
      ic0 = lees_edw_ic_to_buff(le, ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = lees_edw_index(le, ic0, jc, 1);
	index   = lees_edw_index(le, ic1, jc, 1);
	indexp1 = lees_edw_index(le, ic2, jc, 1);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + X] = 0.5*rfactor*
	    (field[nop*indexp1 + n] - field[nop*indexm1 + n]
	     + epsilon_*
	     (field[nop*(indexp1 + ys) + n] - field[nop*(indexm1 + ys) + n] +
	      field[nop*(indexp1 - ys) + n] - field[nop*(indexm1 - ys) + n]));
	  grad[3*(nop*index + n) + Y] = 0.5*rfactor*
	    (field[nop*(index + ys) + n] - field[nop*(index - ys) + n]
	     + epsilon_*
	     (field[nop*(indexp1 + ys) + n] - field[nop*(indexp1 - ys) + n] +
	      field[nop*(indexm1 + ys) + n] - field[nop*(indexm1 - ys) + n]));
	  grad[3*(nop*index + n) + Z] = 0.0;

	  del2[nop*index + n] =
	    rfactor*(field[nop*indexp1 + n] +
		     field[nop*indexm1 + n] +
		     field[nop*(index + ys) + n] +
		     field[nop*(index - ys) + n] +
		     epsilon_*(field[nop*(indexp1 + ys) + n] +
			       field[nop*(indexp1 - ys) + n] +
			       field[nop*(indexm1 + ys) + n] +
			       field[nop*(indexm1 - ys) + n])
		     - 4.0*(1.0 + epsilon_)*field[nop*index + n]);

	}
      }
    }
    /* Next plane */
  }

  return 0;
}
