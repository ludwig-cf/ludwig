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
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "memory.h"
#include "leesedwards.h"
#include "wall.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "gradient_3d_27pt_fluid.h"

__host__ int grad_3d_27pt_fluid_operator(lees_edw_t * le, field_grad_t * fg,
					 int nextra);
__host__ int grad_3d_27pt_fluid_le(lees_edw_t * le, field_grad_t * fg,
				   int nextra);
__host__ int grad_3d_27pt_fluid_wall(field_grad_t * fg,  int nextra);

/*****************************************************************************
 *
 *  grad_3d_27pt_fluid_d2
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_d2(field_grad_t * fgrad) {

  int nextra;
  lees_edw_t * le = NULL;

  assert(fgrad);
  assert(fgrad->field);
  assert(fgrad->field->le);

  le = fgrad->field->le;
  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  grad_3d_27pt_fluid_operator(le, fgrad, nextra);
  grad_3d_27pt_fluid_le(le, fgrad, nextra);
  grad_3d_27pt_fluid_wall(fgrad, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_fluid_d4
 *
 *  Higher derivatives are obtained by using the same operation
 *  on appropriate field.
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_d4(field_grad_t * fgrad) {

  int nextra;
  lees_edw_t * le = NULL;

  nextra = coords_nhalo() - 2;
  assert(nextra >= 0);

  assert(0); /* We need this to work for d4. See 2d_5pt. */
  grad_3d_27pt_fluid_operator(le, fgrad, nextra);
  grad_3d_27pt_fluid_le(le, fgrad, nextra);
  grad_3d_27pt_fluid_wall(fgrad, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_fluid_operator
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_operator(lees_edw_t * le, field_grad_t * fg,
					 int nextra) {

  int nop;
  int nlocal[3];
  int nsites;
  int nhalo;
  int n;
  int ic, jc, kc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  const double r9 = (1.0/9.0);

  double * __restrict__ field;
  double * __restrict__ grad;
  double * __restrict__ del2;

  assert(le);
  assert(fg);

  lees_edw_nhalo(le, &nhalo);
  lees_edw_nlocal(le, nlocal);
  lees_edw_nsites(le, &nsites);

  ys = nlocal[Z] + 2*nhalo;

  nop = fg->field->nf;
  field = fg->field->data;
  grad = fg->grad;
  del2 = fg->delsq;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = lees_edw_index_real_to_buffer(le, ic, -1);
    icp1 = lees_edw_index_real_to_buffer(le, ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = lees_edw_index(le, ic, jc, kc);
	indexm1 = lees_edw_index(le, icm1, jc, kc);
	indexp1 = lees_edw_index(le, icp1, jc, kc);

	for (n = 0; n < nop; n++) {
	  grad[addr_rank2(nsites, nop, 3, index, n, X)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1     ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     );
	  grad[addr_rank2(nsites, nop, 3, index, n, Y)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys  ), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     );
	  grad[addr_rank2(nsites, nop, 3, index, n, Z)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index     +1), n)]
	     - field[addr_rank1(nsites, nop, (index     -1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     );
	  del2[addr_rank1(nsites, nop, index, n)] = r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index     -1), n)]
	     + field[addr_rank1(nsites, nop, (index     +1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - 26.0*field[addr_rank1(nsites, nop, index, n)]);
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_le
 *
 *  The gradients of the order parameter need to be computed in the
 *  buffer region (nextra points). This is so that gradients at all
 *  neighbouring points across a plane can be accessed safely.
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_le(lees_edw_t * le, field_grad_t * fg,
				   int nextra) {

  int nop;
  int nlocal[3];
  int nsites;
  int nhalo;
  int nh;                                 /* counter over halo extent */
  int n;
  int nplane;                             /* Number LE planes */
  int ic, jc, kc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */

  const double r9 = (1.0/9.0);

  double * __restrict__ field;
  double * __restrict__ grad;
  double * __restrict__ del2;

  assert(le);
  assert(fg);

  lees_edw_nhalo(le, &nhalo);
  lees_edw_nsites(le, &nsites);
  lees_edw_nlocal(le, nlocal);

  ys = (nlocal[Z] + 2*nhalo);

  nop = fg->field->nf;
  field = fg->field->data;
  grad = fg->grad;
  del2 = fg->delsq;

  for (nplane = 0; nplane < lees_edw_nplane_local(le); nplane++) {

    ic = lees_edw_plane_location(le, nplane);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = lees_edw_index_real_to_buffer(le, ic, nh-1);
      ic1 = lees_edw_index_real_to_buffer(le, ic, nh  );
      ic2 = lees_edw_index_real_to_buffer(le, ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = lees_edw_index(le, ic0, jc, kc);
	  index   = lees_edw_index(le, ic1, jc, kc);
	  indexp1 = lees_edw_index(le, ic2, jc, kc);

	for (n = 0; n < nop; n++) {
	  grad[addr_rank2(nsites, nop, 3, index, n, X)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1     ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     );
	  grad[addr_rank2(nsites, nop, 3, index, n, Y)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys  ), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     );
	  grad[addr_rank2(nsites, nop, 3, index, n, Z)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index     +1), n)]
	     - field[addr_rank1(nsites, nop, (index     -1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     );
	  del2[addr_rank1(nsites, nop, index, n)] = r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index     -1), n)]
	     + field[addr_rank1(nsites, nop, (index     +1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - 26.0*field[addr_rank1(nsites, nop, index, n)]);
	}
	}
      }
    }

    /* Looking across the plane in the -ve x-direction. */
    ic += 1;

    for (nh = 1; nh <= nextra; nh++) {
      ic2 = lees_edw_index_real_to_buffer(le, ic, -nh+1);
      ic1 = lees_edw_index_real_to_buffer(le, ic, -nh  );
      ic0 = lees_edw_index_real_to_buffer(le, ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = lees_edw_index(le, ic0, jc, kc);
	  index   = lees_edw_index(le, ic1, jc, kc);
	  indexp1 = lees_edw_index(le, ic2, jc, kc);

	for (n = 0; n < nop; n++) {
	  grad[addr_rank2(nsites, nop, 3, index, n, X)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1     ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     );
	  grad[addr_rank2(nsites, nop, 3, index, n, Y)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys  ), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     );
	  grad[addr_rank2(nsites, nop, 3, index, n, Z)] = 0.5*r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index     +1), n)]
	     - field[addr_rank1(nsites, nop, (index     -1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     - field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     );
	  del2[addr_rank1(nsites, nop, index, n)] = r9*
	    (+ field[addr_rank1(nsites, nop, (indexm1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexm1+ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  -ys+1), n)]
	     + field[addr_rank1(nsites, nop, (index     -1), n)]
	     + field[addr_rank1(nsites, nop, (index     +1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys-1), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys  ), n)]
	     + field[addr_rank1(nsites, nop, (index  +ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1-ys+1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   -1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1     ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1   +1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys-1), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys  ), n)]
	     + field[addr_rank1(nsites, nop, (indexp1+ys+1), n)]
	     - 26.0*field[addr_rank1(nsites, nop, index, n)]);
	}

	}
      }
    }
    /* Next plane */
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_fluid_wall
 *
 *  Correct the gradients near the X boundary wall, if necessary.
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_wall(field_grad_t * fg,
				     int nextra) {

  if (wall_present()) {
    fatal("Wall not implemented in 3d 27pt gradients yet (use 7pt)\n");
  }

  return 0;
}
