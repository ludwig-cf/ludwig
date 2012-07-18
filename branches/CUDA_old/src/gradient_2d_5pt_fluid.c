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
#include "leesedwards.h"
#include "wall.h"
#include "gradient.h"
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
static void gradient_2d_5pt_fluid_wall_correction(const int nop,
						  const double * field,
						  double * grad,
						  double * delsq,
						  const int nextra);

/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid_init
 *
 *****************************************************************************/

void gradient_2d_5pt_fluid_init(void) {

  gradient_d2_set(gradient_2d_5pt_fluid_d2);
  gradient_d4_set(gradient_2d_5pt_fluid_d4);
  gradient_d2_dyadic_set(gradient_2d_5pt_fluid_dyadic);

  return;
}

/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid_d2
 *
 *****************************************************************************/

void gradient_2d_5pt_fluid_d2(const int nop, const double * field,
			      double * grad, double * delsq) {

  int nextra;

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_2d_5pt_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_2d_5pt_fluid_le_correction(nop, field, grad, delsq, nextra);
  gradient_2d_5pt_fluid_wall_correction(nop, field, grad, delsq, nextra);

  return;
}

/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid_d4
 *
 *****************************************************************************/

void gradient_2d_5pt_fluid_d4(const int nop, const double * field,
			      double * grad, double * delsq) {

  int nextra;

  nextra = coords_nhalo() - 2;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_2d_5pt_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_2d_5pt_fluid_le_correction(nop, field, grad, delsq, nextra);
  gradient_2d_5pt_fluid_wall_correction(nop, field, grad, delsq, nextra);

  return;
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

/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid_wall_correction
 *
 *  Correct the gradients near the X boundary wall, if necessary.
 *
 *****************************************************************************/

static void gradient_2d_5pt_fluid_wall_correction(const int nop,
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

/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid_dyadic
 *
 *  To compute d_c P_a P_b for vector order parameters P_a
 *
 *****************************************************************************/

void gradient_2d_5pt_fluid_dyadic(const int nop,
				  const double * field,
				  double * grad,
				  double * delsq) {
  int nlocal[3];
  int nhalo;
  int nextra;
  int ic, jc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  nhalo = coords_nhalo();
  nextra = nhalo - 1;

  coords_nlocal(nlocal);

  assert(nlocal[Z] == 1);   /* 2d */
  assert(nop == 3);         /* vector order parameter */
  assert(field);            /* order parameter field */
  assert(grad);             /* gradient of dyadic tensor */
  assert(delsq);            /* delsq of dyadic tensor */

  ys = nlocal[Z] + 2*nhalo;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = le_site_index(ic, jc, 1);
      indexm1 = le_site_index(icm1, jc, 1);
      indexp1 = le_site_index(icp1, jc, 1);

      grad[18*index + 6*X + XX] = 0.5*
	(+ field[nop*indexp1 + X]*field[nop*indexp1 + X]
	 - field[nop*indexm1 + X]*field[nop*indexm1 + X]);
      grad[18*index + 6*X + XY] = 0.5*
	(+ field[nop*indexp1 + X]*field[nop*indexp1 + Y]
	 - field[nop*indexm1 + X]*field[nop*indexm1 + Y]);
      grad[18*index + 6*X + XZ] = 0.0;
      grad[18*index + 6*X + YY] = 0.5*
	(+ field[nop*indexp1 + Y]*field[nop*indexp1 + Y]
	 - field[nop*indexm1 + Y]*field[nop*indexm1 + Y]);
      grad[18*index + 6*X + YZ] = 0.0;
      grad[18*index + 6*X + ZZ] = 0.0;

      grad[18*index + 6*Y + XX] = 0.5*
	(+ field[nop*(index+ys) + X]*field[nop*(index+ys) + X]
	 - field[nop*(index-ys) + X]*field[nop*(index-ys) + X]);
      grad[18*index + 6*Y + XY] = 0.5*
	(+ field[nop*(index+ys) + X]*field[nop*(index+ys) + Y]
	 - field[nop*(index-ys) + X]*field[nop*(index-ys) + Y]);
      grad[18*index + 6*Y + XZ] = 0.0;
      grad[18*index + 6*Y + YY] = 0.5*
	(+ field[nop*(index+ys) + Y]*field[nop*(index+ys) + Y]
	 - field[nop*(index-ys) + Y]*field[nop*(index-ys) + Y]);
      grad[18*index + 6*Y + YZ] = 0.0;
      grad[18*index + 6*Y + ZZ] = 0.0;

      grad[18*index + 6*Z + XX] = 0.0;
      grad[18*index + 6*Z + XY] = 0.0;
      grad[18*index + 6*Z + XZ] = 0.0;
      grad[18*index + 6*Z + YY] = 0.0;
      grad[18*index + 6*Z + YZ] = 0.0;
      grad[18*index + 6*Z + ZZ] = 0.0;

      delsq[6*index + XX] =
	+ field[nop*indexm1      + X]*field[nop*indexm1      + X]
	+ field[nop*indexp1      + X]*field[nop*indexp1      + X]
	+ field[nop*(index - ys) + X]*field[nop*(index - ys) + X]
	+ field[nop*(index + ys) + X]*field[nop*(index + ys) + X]
	- 4.0*field[nop*index + X]*field[nop*index + X];
      delsq[6*index + XY] =
	+ field[nop*indexm1      + X]*field[nop*indexm1      + Y]
	+ field[nop*indexp1      + X]*field[nop*indexp1      + Y]
	+ field[nop*(index - ys) + X]*field[nop*(index - ys) + Y]
	+ field[nop*(index + ys) + X]*field[nop*(index + ys) + Y]
	- 4.0*field[nop*index + X]*field[nop*index + Y];
      delsq[6*index + XZ] = 0.0;
      delsq[6*index + YY] =
	+ field[nop*indexm1      + Y]*field[nop*indexm1      + Y]
	+ field[nop*indexp1      + Y]*field[nop*indexp1      + Y]
	+ field[nop*(index - ys) + Y]*field[nop*(index - ys) + Y]
	+ field[nop*(index + ys) + Y]*field[nop*(index + ys) + Y]
	- 4.0*field[nop*index + Y]*field[nop*index + Y];
      delsq[6*index + YZ] = 0.0;
      delsq[6*index + ZZ] = 0.0;

      /* Next site */
    }
  }
  
  return;
}
