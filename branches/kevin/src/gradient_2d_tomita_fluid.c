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
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "leesedwards_s.h"
#include "gradient_2d_tomita_fluid.h"

static const double epsilon_ = 0.5;
static const double epsilon1_ = 0.25;

static void gradient_2d_tomita_fluid_operator(const int nop,
					      const double * field,
					      double * grad,
					      double * delsq,
					      const int nextra);
static void gradient_2d_tomita_fluid_le_correction(const int nop,
						   const double * field,
						   double * grad,
						   double * delsq,
						   int nextra);

/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid_d2
 *
 *****************************************************************************/

int gradient_2d_tomita_fluid_d2(const int nop, const double * field,double * t_field,
				double * grad,double * t_grad, double * delsq, double * t_delsq, char * siteMask,char * t_siteMask) {

  int nhalo;
  int nextra;

  coords_nhalo(le_stat->cs, &nhalo);
  nextra = nhalo - 1;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_2d_tomita_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_2d_tomita_fluid_le_correction(nop, field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid_d4
 *
 *****************************************************************************/

int gradient_2d_tomita_fluid_d4(const int nop, const double * field,double * t_field,
				double * grad,double * t_grad, double * delsq, double * t_delsq, char * siteMask,char * t_siteMask){

  int nhalo;
  int nextra;

  /* PENDING */
  le_t * le = le_stat;

  le_nhalo(le, &nhalo);
  nextra = nhalo - 2;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_2d_tomita_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_2d_tomita_fluid_le_correction(nop, field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid_operator
 *
 *****************************************************************************/

static void gradient_2d_tomita_fluid_operator(const int nop,
					      const double * field,
					      double * grad,
					      double * del2,
					      const int nextra) {
  int nlocal[3];
  int n;
  int ic, jc;
  int zs, ys, xs;
  int icm1, icp1;
  int index, indexm1, indexp1;

  const double rfactor = 1.0 / (1.0 + 2.0*epsilon_);
  const double rfactor1 = 1.0 / (1.0 + 2.0*epsilon1_);

  /* PENDING */
  le_t * le;
  le = le_stat;

  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(le, ic, -1);
    icp1 = le_index_real_to_buffer(le, ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index   = le_site_index(le, ic, jc, 1);
      indexm1 = le_site_index(le, icm1, jc, 1);
      indexp1 = le_site_index(le, icp1, jc, 1);

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

  return;
}

/*****************************************************************************
 *
 *  gradient_2d_tomita_le_correction
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

static void gradient_2d_tomita_fluid_le_correction(const int nop,
						const double * field,
						double * grad,
						double * del2,
						const int nextra) {

  int nplane;                             /* Number LE planes */
  int nlocal[3];
  int nh;                                 /* counter over halo extent */
  int n, np;
  int ic, jc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int zs, ys, xs;                         /* strides for 1d address */

  const double rfactor = 1.0 / (1.0 + 2.0*epsilon_);

  /* PENDING */
  le_t * le;
  le = le_stat;

  le_nplane_local(le, &nplane);
  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);

  assert(nlocal[Z] == 1);

  for (np = 0; np < nplane; np++) {

    ic = le_plane_location(le, np);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(le, ic, nh-1);
      ic1 = le_index_real_to_buffer(le, ic, nh  );
      ic2 = le_index_real_to_buffer(le, ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = le_site_index(le, ic0, jc, 1);
	index   = le_site_index(le, ic1, jc, 1);
	indexp1 = le_site_index(le, ic2, jc, 1);

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
      ic2 = le_index_real_to_buffer(le, ic, -nh+1);
      ic1 = le_index_real_to_buffer(le, ic, -nh  );
      ic0 = le_index_real_to_buffer(le, ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = le_site_index(le, ic0, jc, 1);
	index   = le_site_index(le, ic1, jc, 1);
	indexp1 = le_site_index(le, ic2, jc, 1);

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

  return;
}
