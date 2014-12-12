/*****************************************************************************
 *
 *  gradient_3d_27pt_fluid.c
 *
 *  Gradient operations for equally-weighted 27-point stencil
 *  in three dimensions.
 *
 *        (ic-1, jc+1, kc) (ic, jc+1, kc) (ic+1, jc+1, kc)
 *        (ic-1, jc,   kc) (ic, jc,   kc) (ic+1, jc,   kc)
 *        (ic-1, jc-1, kc) (ic, jc-1, kc) (ic+1, jc,   kc)
 *
 *  ...and so in z-direction
 *
 *  d_x phi = [phi(ic+1, j, k) - phi(ic-1, j, k)] / 2*9
 *  for all j = jc-1,jc,jc+1, and k = kc-1,kc,kc+1
 * 
 *  d_y phi = [phi(i ,jc+1,k ) - phi(i ,jc-1,k )] / 2*9
 *  for all i = ic-1,ic,ic+1 and k = kc-1,kc,kc+1
 *
 *  d_z phi = [phi(i ,j ,kc+1) - phi(i ,j ,kc-1)] / 2*9
 *  for all i = ic-1,ic,ic+1 and j = jc-1,jc,jc+1
 *
 *  nabla^2 phi = phi(ic+1,jc,kc) + phi(ic-1,jc,kc)
 *              + phi(ic,jc+1,kc) + phi(ic,jc-1,kc)
 *              + phi(ic,jc,kc+1) + phi(ic,jc,kc-1)
 *              etc
 *              - 26 phi(ic,jc,kc)
 *
 *  Corrections for Lees-Edwards planes are included.
 *
 *  This scheme was fist instituted for the work of Kendon et al.
 *  JFM (2001).
 *
 *  $Id: gradient_3d_27pt_fluid.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
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
#include "leesedwards.h"
#include "wall.h"
#include "gradient_3d_27pt_fluid.h"

static void gradient_3d_27pt_fluid_operator(const int nop,
					    const double * field,
					    double * grad, double * delsq,
					    const int nextra);
static void gradient_3d_27pt_fluid_le_correction(const int nop,
						 const double * field,
						 double * grad, double * delsq,
						 const int nextra);
static void gradient_3d_27pt_fluid_wall_correction(const int nop,
						   const double * field,
						   double * grad,
						   double * delsq,
						   const int nextra);

/*****************************************************************************
 *
 *  gradient_3d_27pt_fluid_d2
 *
 *****************************************************************************/

int gradient_3d_27pt_fluid_d2(const int nop, const double * field,double * t_field,
				double * grad,double * t_grad, double * delsq, double * t_delsq, char * siteMask,char * t_siteMask) {

  int nextra;

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  gradient_3d_27pt_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_3d_27pt_fluid_le_correction(nop, field, grad, delsq, nextra);
  gradient_3d_27pt_fluid_wall_correction(nop, field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_3d_27pt_fluid_d4
 *
 *  Higher derivatives are obtained by using the same operation
 *  on appropriate field.
 *
 *****************************************************************************/

int gradient_3d_27pt_fluid_d4(const int nop, const double * field,double * t_field,
				double * grad,double * t_grad, double * delsq, double * t_delsq, char * siteMask,char * t_siteMask) {

  int nextra;

  nextra = coords_nhalo() - 2;
  assert(nextra >= 0);

  gradient_3d_27pt_fluid_operator(nop, field, grad, delsq, nextra);
  gradient_3d_27pt_fluid_le_correction(nop, field, grad, delsq, nextra);
  gradient_3d_27pt_fluid_wall_correction(nop, field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_3d_27pt_fluid_operator
 *
 *****************************************************************************/

static void gradient_3d_27pt_fluid_operator(const int nop,
					    const double * field,
					   double * grad,
					   double * del2,
					   const int nextra) {
  int nlocal[3];
  int nhalo;
  int n;
  int ic, jc, kc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  const double r9 = (1.0/9.0);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  ys = nlocal[Z] + 2*nhalo;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = le_site_index(ic, jc, kc);
	indexm1 = le_site_index(icm1, jc, kc);
	indexp1 = le_site_index(icp1, jc, kc);

	for (n = 0; n < nop; n++) {
	    grad[3*(nop*index + n) + X] = 0.5*r9*
	      (+ field[nop*(indexp1-ys-1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexp1-ys  ) + n] - field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexp1-ys+1) + n] - field[nop*(indexm1-ys+1) + n]
	       + field[nop*(indexp1   -1) + n] - field[nop*(indexm1   -1) + n]
	       + field[nop*(indexp1     ) + n] - field[nop*(indexm1     ) + n]
	       + field[nop*(indexp1   +1) + n] - field[nop*(indexm1   +1) + n]
	       + field[nop*(indexp1+ys-1) + n] - field[nop*(indexm1+ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n] - field[nop*(indexm1+ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexm1+ys+1) + n]
	       );
	    grad[3*(nop*index + n) + Y] = 0.5*r9*
	      (+ field[nop*(indexm1+ys-1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1+ys  ) + n] - field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexm1+ys+1) + n] - field[nop*(indexm1-ys+1) + n]
	       + field[nop*(index  +ys-1) + n] - field[nop*(index  -ys-1) + n]
	       + field[nop*(index  +ys  ) + n] - field[nop*(index  -ys  ) + n]
	       + field[nop*(index  +ys+1) + n] - field[nop*(index  -ys+1) + n]
	       + field[nop*(indexp1+ys-1) + n] - field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n] - field[nop*(indexp1-ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexp1-ys+1) + n]
	       );
	    grad[3*(nop*index + n) + Z] = 0.5*r9*
	      (+ field[nop*(indexm1-ys+1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1   +1) + n] - field[nop*(indexm1   -1) + n]
	       + field[nop*(indexm1+ys+1) + n] - field[nop*(indexm1+ys-1) + n]
	       + field[nop*(index  -ys+1) + n] - field[nop*(index  -ys-1) + n]
	       + field[nop*(index     +1) + n] - field[nop*(index     -1) + n]
	       + field[nop*(index  +ys+1) + n] - field[nop*(index  +ys-1) + n]
	       + field[nop*(indexp1-ys+1) + n] - field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1   +1) + n] - field[nop*(indexp1   -1) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexp1+ys-1) + n]
	       );
	    del2[nop*index + n] = r9*
	      (+ field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexm1-ys+1) + n]
	       + field[nop*(indexm1   -1) + n]
	       + field[nop*(indexm1     ) + n]
	       + field[nop*(indexm1   +1) + n]
	       + field[nop*(indexm1+ys-1) + n]
	       + field[nop*(indexm1+ys  ) + n]
	       + field[nop*(indexm1+ys+1) + n]
	       + field[nop*(index  -ys-1) + n]
	       + field[nop*(index  -ys  ) + n]
	       + field[nop*(index  -ys+1) + n]
	       + field[nop*(index     -1) + n]
	       + field[nop*(index     +1) + n]
	       + field[nop*(index  +ys-1) + n]
	       + field[nop*(index  +ys  ) + n]
	       + field[nop*(index  +ys+1) + n]
	       + field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1-ys  ) + n]
	       + field[nop*(indexp1-ys+1) + n]
	       + field[nop*(indexp1   -1) + n]
	       + field[nop*(indexp1     ) + n]
	       + field[nop*(indexp1   +1) + n]
	       + field[nop*(indexp1+ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n]
	       - 26.0*field[nop*index + n]);
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_27pt_le_correction
 *
 *  The gradients of the order parameter need to be computed in the
 *  buffer region (nextra points). This is so that gradients at all
 *  neighbouring points across a plane can be accessed safely.
 *
 *****************************************************************************/

static void gradient_3d_27pt_fluid_le_correction(const int nop,
						 const double * field,
						 double * grad,
						 double * del2,
						 int nextra) {
  int nlocal[3];
  int nhalo;
  int nh;                                 /* counter over halo extent */
  int n;
  int nplane;                             /* Number LE planes */
  int ic, jc, kc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */

  const double r9 = (1.0/9.0);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  ys = (nlocal[Z] + 2*nhalo);

  for (nplane = 0; nplane < le_get_nplane_local(); nplane++) {

    ic = le_plane_location(nplane);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(ic, nh-1);
      ic1 = le_index_real_to_buffer(ic, nh  );
      ic2 = le_index_real_to_buffer(ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(ic0, jc, kc);
	  index   = le_site_index(ic1, jc, kc);
	  indexp1 = le_site_index(ic2, jc, kc);

	  for (n = 0; n < nop; n++) {
	    grad[3*(nop*index + n) + X] = 0.5*r9*
	      (+ field[nop*(indexp1-ys-1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexp1-ys  ) + n] - field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexp1-ys+1) + n] - field[nop*(indexm1-ys+1) + n]
	       + field[nop*(indexp1   -1) + n] - field[nop*(indexm1   -1) + n]
	       + field[nop*(indexp1     ) + n] - field[nop*(indexm1     ) + n]
	       + field[nop*(indexp1   +1) + n] - field[nop*(indexm1   +1) + n]
	       + field[nop*(indexp1+ys-1) + n] - field[nop*(indexm1+ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n] - field[nop*(indexm1+ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexm1+ys+1) + n]
	       );
	    grad[3*(nop*index + n) + Y] = 0.5*r9*
	      (+ field[nop*(indexm1+ys-1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1+ys  ) + n] - field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexm1+ys+1) + n] - field[nop*(indexm1-ys+1) + n]
	       + field[nop*(index  +ys-1) + n] - field[nop*(index  -ys-1) + n]
	       + field[nop*(index  +ys  ) + n] - field[nop*(index  -ys  ) + n]
	       + field[nop*(index  +ys+1) + n] - field[nop*(index  -ys+1) + n]
	       + field[nop*(indexp1+ys-1) + n] - field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n] - field[nop*(indexp1-ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexp1-ys+1) + n]
	       );
	    grad[3*(nop*index + n) + Z] = 0.5*r9*
	      (+ field[nop*(indexm1-ys+1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1   +1) + n] - field[nop*(indexm1   -1) + n]
	       + field[nop*(indexm1+ys+1) + n] - field[nop*(indexm1+ys-1) + n]
	       + field[nop*(index  -ys+1) + n] - field[nop*(index  -ys-1) + n]
	       + field[nop*(index     +1) + n] - field[nop*(index     -1) + n]
	       + field[nop*(index  +ys+1) + n] - field[nop*(index  +ys-1) + n]
	       + field[nop*(indexp1-ys+1) + n] - field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1   +1) + n] - field[nop*(indexp1   -1) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexp1+ys-1) + n]
	       );
	    del2[nop*index + n] = r9*
	      (+ field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexm1-ys+1) + n]
	       + field[nop*(indexm1   -1) + n]
	       + field[nop*(indexm1     ) + n]
	       + field[nop*(indexm1   +1) + n]
	       + field[nop*(indexm1+ys-1) + n]
	       + field[nop*(indexm1+ys  ) + n]
	       + field[nop*(indexm1+ys+1) + n]
	       + field[nop*(index  -ys-1) + n]
	       + field[nop*(index  -ys  ) + n]
	       + field[nop*(index  -ys+1) + n]
	       + field[nop*(index     -1) + n]
	       + field[nop*(index     +1) + n]
	       + field[nop*(index  +ys-1) + n]
	       + field[nop*(index  +ys  ) + n]
	       + field[nop*(index  +ys+1) + n]
	       + field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1-ys  ) + n]
	       + field[nop*(indexp1-ys+1) + n]
	       + field[nop*(indexp1   -1) + n]
	       + field[nop*(indexp1     ) + n]
	       + field[nop*(indexp1   +1) + n]
	       + field[nop*(indexp1+ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n]
	       - 26.0*field[nop*index + n]);
	  }
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
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(ic0, jc, kc);
	  index   = le_site_index(ic1, jc, kc);
	  indexp1 = le_site_index(ic2, jc, kc);

	  for (n = 0; n < nop; n++) {
	    grad[3*(nop*index + n) + X] = 0.5*r9*
	      (+ field[nop*(indexp1-ys-1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexp1-ys  ) + n] - field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexp1-ys+1) + n] - field[nop*(indexm1-ys+1) + n]
	       + field[nop*(indexp1   -1) + n] - field[nop*(indexm1   -1) + n]
	       + field[nop*(indexp1     ) + n] - field[nop*(indexm1     ) + n]
	       + field[nop*(indexp1   +1) + n] - field[nop*(indexm1   +1) + n]
	       + field[nop*(indexp1+ys-1) + n] - field[nop*(indexm1+ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n] - field[nop*(indexm1+ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexm1+ys+1) + n]
	       );
	    grad[3*(nop*index + n) + Y] = 0.5*r9*
	      (+ field[nop*(indexm1+ys-1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1+ys  ) + n] - field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexm1+ys+1) + n] - field[nop*(indexm1-ys+1) + n]
	       + field[nop*(index  +ys-1) + n] - field[nop*(index  -ys-1) + n]
	       + field[nop*(index  +ys  ) + n] - field[nop*(index  -ys  ) + n]
	       + field[nop*(index  +ys+1) + n] - field[nop*(index  -ys+1) + n]
	       + field[nop*(indexp1+ys-1) + n] - field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n] - field[nop*(indexp1-ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexp1-ys+1) + n]
	       );
	    grad[3*(nop*index + n) + Z] = 0.5*r9*
	      (+ field[nop*(indexm1-ys+1) + n] - field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1   +1) + n] - field[nop*(indexm1   -1) + n]
	       + field[nop*(indexm1+ys+1) + n] - field[nop*(indexm1+ys-1) + n]
	       + field[nop*(index  -ys+1) + n] - field[nop*(index  -ys-1) + n]
	       + field[nop*(index     +1) + n] - field[nop*(index     -1) + n]
	       + field[nop*(index  +ys+1) + n] - field[nop*(index  +ys-1) + n]
	       + field[nop*(indexp1-ys+1) + n] - field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1   +1) + n] - field[nop*(indexp1   -1) + n]
	       + field[nop*(indexp1+ys+1) + n] - field[nop*(indexp1+ys-1) + n]
	       );
	    del2[nop*index + n] = r9*
	      (+ field[nop*(indexm1-ys-1) + n]
	       + field[nop*(indexm1-ys  ) + n]
	       + field[nop*(indexm1-ys+1) + n]
	       + field[nop*(indexm1   -1) + n]
	       + field[nop*(indexm1     ) + n]
	       + field[nop*(indexm1   +1) + n]
	       + field[nop*(indexm1+ys-1) + n]
	       + field[nop*(indexm1+ys  ) + n]
	       + field[nop*(indexm1+ys+1) + n]
	       + field[nop*(index  -ys-1) + n]
	       + field[nop*(index  -ys  ) + n]
	       + field[nop*(index  -ys+1) + n]
	       + field[nop*(index     -1) + n]
	       + field[nop*(index     +1) + n]
	       + field[nop*(index  +ys-1) + n]
	       + field[nop*(index  +ys  ) + n]
	       + field[nop*(index  +ys+1) + n]
	       + field[nop*(indexp1-ys-1) + n]
	       + field[nop*(indexp1-ys  ) + n]
	       + field[nop*(indexp1-ys+1) + n]
	       + field[nop*(indexp1   -1) + n]
	       + field[nop*(indexp1     ) + n]
	       + field[nop*(indexp1   +1) + n]
	       + field[nop*(indexp1+ys-1) + n]
	       + field[nop*(indexp1+ys  ) + n]
	       + field[nop*(indexp1+ys+1) + n]
	       - 26.0*field[nop*index + n]);
	  }
	}
      }
    }
    /* Next plane */
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_27pt_fluid_wall_correction
 *
 *  Correct the gradients near the X boundary wall, if necessary.
 *
 *****************************************************************************/

static void gradient_3d_27pt_fluid_wall_correction(const int nop,
						   const double * field,
						   double * grad,
						   double * del2,
						   const int nextra) {

  if (wall_present()) {
    fatal("Wall not implemented in 3d 27pt gradients yet (use 7pt)\n");
  }

  return;
}
