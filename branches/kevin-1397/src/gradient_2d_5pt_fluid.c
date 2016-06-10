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
 *  (c) 2010-2016 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "field_s.h"
#include "leesedwards.h"
#include "wall.h"
#include "field_grad_s.h"
#include "gradient_2d_5pt_fluid.h"

__host__ int grad_2d_5pt_fluid_operator(field_grad_t * fg, int nextra);
__host__ int grad_2d_5pt_fluid_le_correction(field_grad_t * fg, int nextra);
__host__ int grad_2d_5pt_fluid_wall_correction(field_grad_t * fg, int nextra);

/*****************************************************************************
 *
 *  grad_2d_5pt_fluid_d2
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_d2(field_grad_t * fg) {

  int nextra;

  assert(fg);
  assert(fg->field);

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  grad_2d_5pt_fluid_operator(fg, nextra);
  grad_2d_5pt_fluid_le_correction(fg, nextra);
  grad_2d_5pt_fluid_wall_correction(fg, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_5pt_fluid_d4
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_d4(field_grad_t * fg) {

  int nextra;

  assert(fg);
  assert(fg->field);

  nextra = coords_nhalo() - 2;
  assert(nextra >= 0);

  grad_2d_5pt_fluid_operator(fg, nextra);
  grad_2d_5pt_fluid_le_correction(fg, nextra);
  grad_2d_5pt_fluid_wall_correction(fg, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_5pt_fluid_operator
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_operator(field_grad_t * fg,  int nextra) {

  int nop;
  int nlocal[3];
  int nhalo;
  int nsites;
  int n;
  int ic, jc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  double * __restrict__ field;
  double * __restrict__ grad;
  double * __restrict__ del2;

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nsites = le_nsites();

  ys = nlocal[Z] + 2*nhalo;

  nop = fg->field->nf;
  field = fg->field->data;
  grad = fg->grad;
  del2 = fg->delsq;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = le_site_index(ic, jc, 1);
      indexm1 = le_site_index(icm1, jc, 1);
      indexp1 = le_site_index(icp1, jc, 1);

      for (n = 0; n < nop; n++) {
	grad[addr_rank2(nsites, nop, NVECTOR, index, n, X)]
	  = 0.5*(field[addr_rank1(nsites, nop, indexp1, n)]
	       - field[addr_rank1(nsites, nop, indexm1, n)]);
	grad[addr_rank2(nsites, nop, NVECTOR, index, n, Y)]
	  = 0.5*(field[addr_rank1(nsites, nop, index + ys, n)]
	       - field[addr_rank1(nsites, nop, index - ys, n)]);
	grad[addr_rank2(nsites, nop, NVECTOR, index, n, Z)] = 0.0;

	del2[addr_rank1(nsites, nop, index, n)]
	  = field[addr_rank1(nsites, nop, indexp1,    n)]
	  + field[addr_rank1(nsites, nop, indexm1,    n)]
	  + field[addr_rank1(nsites, nop, index + ys, n)]
	  + field[addr_rank1(nsites, nop, index - ys, n)]
	  - 4.0*field[addr_rank1(nsites, nop, index,  n)];
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_5pt_le_correction
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_le_correction(field_grad_t * fg, int nextra) {

  int nop;
  int nlocal[3];
  int nhalo;
  int nsites;
  int nh;                                 /* counter over halo extent */
  int n;
  int nplane;                             /* Number LE planes */
  int ic, jc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */

  double * __restrict__ field;
  double * __restrict__ grad;
  double * __restrict__ del2;

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nsites = le_nsites();

  assert(nlocal[Z] == 1);

  ys = (nlocal[Z] + 2*nhalo);

  nop = fg->field->nf;
  field = fg->field->data;
  grad = fg->grad;
  del2 = fg->delsq;

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
	  grad[addr_rank2(nsites, nop, NVECTOR, index, n, X)]
	    = 0.5*(field[addr_rank1(nsites, nop, indexp1, n)]
		 - field[addr_rank1(nsites, nop, indexm1, n)]);
	  grad[addr_rank2(nsites, nop, NVECTOR, index, n, Y)]
	    = 0.5*(field[addr_rank1(nsites, nop, index + ys, n)]
		 - field[addr_rank1(nsites, nop, index - ys, n)]);
	  grad[addr_rank2(nsites, nop, NVECTOR, index, n, Z)] = 0.0;
	  del2[addr_rank1(nsites, nop, index, n)]
	    = field[addr_rank1(nsites, nop, indexp1, n)]
	    + field[addr_rank1(nsites, nop, indexm1, n)]
	    + field[addr_rank1(nsites, nop, index + ys, n)]
	    + field[addr_rank1(nsites, nop, index - ys, n)]
	    - 4.0*field[addr_rank1(nsites, nop, index, n)];
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
	  grad[addr_rank2(nsites, nop, NVECTOR, index, n, X)]
	    = 0.5*(field[addr_rank1(nsites, nop, indexp1, n)]
		 - field[addr_rank1(nsites, nop, indexm1, n)]);
	  grad[addr_rank2(nsites, nop, NVECTOR, index, n, Y)]
	    = 0.5*(field[addr_rank1(nsites, nop, (index + ys), n)]
		 - field[addr_rank1(nsites, nop, (index - ys), n)]);
	  grad[addr_rank2(nsites, nop, NVECTOR, index, n, Z)] = 0.0;
	  del2[addr_rank1(nsites, nop, index, n)]
	    = field[addr_rank1(nsites, nop, indexp1, n)]
	    + field[addr_rank1(nsites, nop, indexm1, n)]
	    + field[addr_rank1(nsites, nop, (index + ys), n)]
	    + field[addr_rank1(nsites, nop, (index - ys), n)]
	    - 4.0*field[addr_rank1(nsites, nop, index, n)];
	}
      }
    }
    /* Next plane */
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_5pt_fluid_wall_correction
 *
 *  Correct the gradients near the X boundary wall, if necessary.
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_wall_correction(field_grad_t * fg, int nextra) {

  int nop;
  int nlocal[3];
  int nhalo;
  int nsites;
  int n;
  int jc;
  int index;
  int xs, ys;

  double fb;                    /* Extrapolated value of field at boundary */
  double gradm1, gradp1;        /* gradient terms */
  double rk;                    /* Fluid free energy parameter (reciprocal) */
  double * c;                   /* Solid free energy parameters C */
  double * h;                   /* Solid free energy parameters H */

  double * __restrict__ field;
  double * __restrict__ grad;
  double * __restrict__ del2;

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nsites = le_nsites();

  assert(nlocal[Z] == 1);

  ys = (nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  assert(wall_at_edge(Y) == 0);
  assert(wall_at_edge(Z) == 0);

  nop = fg->field->nf;
  field = fg->field->data;
  grad = fg->grad;
  del2 = fg->delsq;

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
	gradp1 = field[addr_rank1(nsites, nop, index + xs, n)]
	       - field[addr_rank1(nsites, nop, index, n)];
	fb = field[addr_rank1(nsites, nop, index, n)] - 0.5*gradp1;
	gradm1 = -(c[n]*fb + h[n])*rk;
	grad[addr_rank2(nsites, nop, NVECTOR, index, n, X)] = 0.5*(gradp1 - gradm1);
	del2[addr_rank1(nsites, nop, index, n)]
	  = gradp1 - gradm1
	  + field[addr_rank1(nsites, nop, (index + ys), n)]
	  + field[addr_rank1(nsites, nop, (index - ys), n)]
	  - 2.0*field[addr_rank1(nsites, nop, index, n)];
      }
      /* Next site */
    }
  }

  if (wall_at_edge(X) && cart_coords(X) == cart_size(X) - 1) {

    /* Correct the upper wall */

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = le_site_index(nlocal[X], jc, 1);

      for (n = 0; n < nop; n++) {
	gradm1 = field[addr_rank1(nsites, nop, index, n)]
	       - field[addr_rank1(nsites, nop, (index - xs), n)];
	fb = field[addr_rank1(nsites, nop, index, n)] + 0.5*gradm1;
	gradp1 = -(c[n]*fb + h[n])*rk;
	grad[addr_rank2(nsites, nop, NVECTOR, index, n, X)] = 0.5*(gradp1 - gradm1);
	del2[addr_rank1(nsites, nop, index, n)]
	  = gradp1 - gradm1
	  + field[addr_rank1(nsites, nop, (index + ys), n)]
	  + field[addr_rank1(nsites, nop, (index - ys), n)]
	  - 2.0*field[addr_rank1(nsites, nop, index, n)];
      }
      /* Next site */
    }
  }

  free(c);
  free(h);

  return 0;
}
