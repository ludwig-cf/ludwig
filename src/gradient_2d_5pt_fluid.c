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

enum grad_type {GRAD_DEL2, GRAD_DEL4};

__host__ int grad_2d_5pt_fluid_operator(lees_edw_t * le, field_grad_t * fg,
					int nextra, int type);
__host__ int grad_2d_5pt_fluid_le(lees_edw_t * le, field_grad_t * fg,
				  int nextra, int type);
__host__ int grad_2d_5pt_fluid_wall(lees_edw_t * le, field_grad_t * fg,
				    int nextra, int type);

/*****************************************************************************
 *
 *  grad_2d_5pt_fluid_d2
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_d2(field_grad_t * fg) {

  int nhalo, nextra;
  lees_edw_t * le = NULL;

  assert(fg);
  assert(fg->field);

  le = fg->field->le;
  lees_edw_nhalo(le, &nhalo);
  nextra = nhalo - 1;
  assert(nextra >= 0);

  grad_2d_5pt_fluid_operator(le, fg, nextra, GRAD_DEL2);
  grad_2d_5pt_fluid_le(le, fg, nextra, GRAD_DEL2);

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_5pt_fluid_d4
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_d4(field_grad_t * fg) {

  int nhalo, nextra;
  lees_edw_t * le = NULL;

  assert(fg);
  assert(fg->field);

  le = fg->field->le;
  lees_edw_nhalo(le, &nhalo);
  nextra = nhalo - 2;
  assert(nextra >= 0);

  grad_2d_5pt_fluid_operator(le, fg, nextra, GRAD_DEL4);
  grad_2d_5pt_fluid_le(le, fg, nextra, GRAD_DEL4);

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_5pt_fluid_operator
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_operator(lees_edw_t * le, field_grad_t * fg, 
					int nextra,
					int type) {

  int nop;
  int nlocal[3];
  int nhalo;
  int nsites;
  int n;
  int ic, jc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  double * __restrict__ field = NULL;
  double * __restrict__ grad = NULL;
  double * __restrict__ del2 = NULL;

  assert(le);
  assert(fg);

  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);
  lees_edw_nsites(le, &nsites);

  ys = nlocal[Z] + 2*nhalo;

  nop = fg->field->nf;
  if (type == GRAD_DEL2) {
    field = fg->field->data;
    grad  = fg->grad;
    del2  = fg->delsq;
  }
  if (type == GRAD_DEL4) {
    field = fg->delsq;
    grad  = fg->grad_delsq;
    del2 =  fg->delsq_delsq;
  }
  assert(type == GRAD_DEL2 || type == GRAD_DEL4);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = lees_edw_index(le, ic, jc, 1);
      indexm1 = lees_edw_index(le, icm1, jc, 1);
      indexp1 = lees_edw_index(le, icp1, jc, 1);

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
 *  grad_2d_5pt_fluid_le
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

__host__ int grad_2d_5pt_fluid_le(lees_edw_t * le, field_grad_t * fg,
				  int nextra, int type) {

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

  double * __restrict__ field = NULL;
  double * __restrict__ grad = NULL;
  double * __restrict__ del2 = NULL;

  assert(le);
  assert(fg);

  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);
  lees_edw_nsites(le, &nsites);

  assert(nlocal[Z] == 1);

  ys = (nlocal[Z] + 2*nhalo);

  nop = fg->field->nf;
  if (type == GRAD_DEL2) {
    field = fg->field->data;
    grad  = fg->grad;
    del2  = fg->delsq;
  }
  if (type == GRAD_DEL4) {
    field = fg->delsq;
    grad  = fg->grad_delsq;
    del2 =  fg->delsq_delsq;
  }
  assert(type == GRAD_DEL2 || type == GRAD_DEL4);

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
      ic2 = lees_edw_ic_to_buff(le, ic, -nh+1);
      ic1 = lees_edw_ic_to_buff(le, ic, -nh  );
      ic0 = lees_edw_ic_to_buff(le, ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = lees_edw_index(le, ic0, jc, 1);
	index   = lees_edw_index(le, ic1, jc, 1);
	indexp1 = lees_edw_index(le, ic2, jc, 1);

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
