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

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "wall.h"
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
static void gradient_2d_tomita_fluid_wall_correction(const int nop,
						     const double * field,
						     double * grad,
						     double * delsq,
						     const int nextra);

/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid_d2
 *
 *****************************************************************************/

int gradient_2d_tomita_fluid_d2(const int nop, const double * field,double * t_field,
				double * grad,double * t_grad, double * delsq, double * t_delsq) {

  int nextra;

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_2d_tomita_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_2d_tomita_fluid_le_correction(nop, field, grad, delsq, nextra);
  gradient_2d_tomita_fluid_wall_correction(nop, field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid_d4
 *
 *****************************************************************************/

int gradient_2d_tomita_fluid_d4(const int nop, const double * field,double * t_field,
				double * grad,double * t_grad, double * delsq, double * t_delsq){

  int nextra;

  nextra = coords_nhalo() - 2;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_2d_tomita_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_2d_tomita_fluid_le_correction(nop, field, grad, delsq, nextra);
  gradient_2d_tomita_fluid_wall_correction(nop, field, grad, delsq, nextra);

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
  int nhalo;
  int n;
  int ic, jc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  const double rfactor = 1.0 / (1.0 + 2.0*epsilon_);
  const double rfactor1 = 1.0 / (1.0 + 2.0*epsilon1_);

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();

  ys = nlocal[Z] + 2*nhalo;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = le_site_index(ic, jc, 1);
      indexm1 = le_site_index(icm1, jc, 1);
      indexp1 = le_site_index(icp1, jc, 1);

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

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();

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
      ic2 = le_index_real_to_buffer(ic, -nh+1);
      ic1 = le_index_real_to_buffer(ic, -nh  );
      ic0 = le_index_real_to_buffer(ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = le_site_index(ic0, jc, 1);
	index   = le_site_index(ic1, jc, 1);
	indexp1 = le_site_index(ic2, jc, 1);

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

/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid_wall_correction
 *
 *  Correct the gradients near the X boundary wall, if necessary.
 *
 *****************************************************************************/

static void gradient_2d_tomita_fluid_wall_correction(const int nop,
						     const double * field,
						     double * grad,
						     double * del2,
						     const int nextra) {
  int nlocal[3];
  int nhalo;
  int n;
  int jc;
  int index;
  int xs, ys;

  double fb;                    /* Extrapolated value of field at boundary */
  double gradm1, gradp1;        /* gradient terms */
  double rk;                    /* Fluid free energy parameter (reciprocal) */
  double * c;                   /* Solid free energy parameters C */
  double * h;                   /* Solid free energy parameters H */

  assert(wall_at_edge(X) == 0); /* NOT IMPLEMENTED */
  coords_nlocal(nlocal);
  nhalo = coords_nhalo();

  assert(nlocal[Z] == 1);

  ys = (nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  assert(wall_at_edge(Y) == 0);
  assert(wall_at_edge(Z) == 0);

  /* This enforces C = 0 and H = 0, ie., neutral wetting, as there
   * is currently no mechanism to obtain the free energy parameters. */

  c = (double *) malloc(nop*sizeof(double));
  h = (double *) malloc(nop*sizeof(double));

  if (c == NULL) fatal("malloc(c) failed\n");
  if (h == NULL) fatal("malloc(h) failed\n");

  for (n = 0; n < nop; n++) {
    c[n] = 0.0;
    h[n] = 0.0;
  }
  rk = 0.0;

  if (wall_at_edge(X) && cart_coords(X) == 0) {

    /* Correct the lower wall */

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = le_site_index(1, jc, 1);

      for (n = 0; n < nop; n++) {
	gradp1 = field[nop*(index + xs) + n] - field[nop*index + n];
	fb = field[nop*index + n] - 0.5*gradp1;
	gradm1 = -(c[n]*fb + h[n])*rk;
	grad[3*(nop*index + n) + X] = 0.5*(gradp1 - gradm1);
	del2[nop*index + n]
	  = gradp1 - gradm1
	  + field[nop*(index + ys) + n] + field[nop*(index  - ys) + n]
	  - 2.0*field[nop*index + n];
      }

      /* Next site */
    }
  }

  if (wall_at_edge(X) && cart_coords(X) == cart_size(X) - 1) {

    /* Correct the upper wall */

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = le_site_index(nlocal[X], jc, 1);

      for (n = 0; n < nop; n++) {
	gradm1 = field[nop*index + n] - field[nop*(index - xs) + n];
	fb = field[nop*index + n] + 0.5*gradm1;
	gradp1 = -(c[n]*fb + h[n])*rk;
	grad[3*(nop*index + n) + X] = 0.5*(gradp1 - gradm1);
	del2[nop*index + n]
	  = gradp1 - gradm1
	  + field[nop*(index + ys) + n] + field[nop*(index  - ys) + n]
	  - 2.0*field[nop*index + n];
      }
      /* Next site */
    }
  }

  free(c);
  free(h);

  return;
}
