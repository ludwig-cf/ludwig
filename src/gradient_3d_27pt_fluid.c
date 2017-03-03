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
#include "kernel.h"
#include "leesedwards.h"
#include "wall.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "gradient_3d_27pt_fluid.h"

typedef enum grad_enum_type {GRAD_DEL2, GRAD_DEL4} grad_enum_t;

__host__ int grad_3d_27pt_fluid_operator(cs_t * cs, lees_edw_t * le,
					 field_grad_t * fg,
					 int nextra, grad_enum_t type);
__host__ int grad_3d_27pt_fluid_le(lees_edw_t * le, field_grad_t * fg,
				   int nextra, grad_enum_t type);

__host__ int grad_3d_27pt_dab_le_correct(lees_edw_t * le, field_grad_t * df);
__host__ int grad_3d_27pt_dab_compute(lees_edw_t * le, field_grad_t * df);

__global__ void grad_3d_27pt_kernel(kernel_ctxt_t * ktx, int nf, int ys,
				    lees_edw_t * le,
				    grad_enum_t type,
				    field_t * f,
				    field_grad_t * fgrad);

/*****************************************************************************
 *
 *  grad_3d_27pt_fluid_d2
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_d2(field_grad_t * fgrad) {

  int nhalo;
  int nextra;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  assert(fgrad);
  assert(fgrad->field);
  assert(fgrad->field->cs);
  assert(fgrad->field->le);

  cs = fgrad->field->cs;
  le = fgrad->field->le;

  cs_nhalo(cs, &nhalo);
  nextra = nhalo - 1;
  assert(nextra >= 0);

  grad_3d_27pt_fluid_operator(cs, le, fgrad, nextra, GRAD_DEL2);
  grad_3d_27pt_fluid_le(le, fgrad, nextra, GRAD_DEL2);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_fluid_d4
 *
 *  Higher derivatives are obtained by using the same operation
 *  on appropriate field.
 *
 *  TODO: There's no test for this at the moment.
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_d4(field_grad_t * fgrad) {

  int nextra;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  assert(fgrad);
  assert(fgrad->field);
  assert(fgrad->field->cs);
  assert(fgrad->field->le);

  cs = fgrad->field->cs;
  le = fgrad->field->le;

  cs_nhalo(cs, &nextra);
  nextra -= 2;
  assert(nextra >= 0);

  grad_3d_27pt_fluid_operator(cs, le, fgrad, nextra, GRAD_DEL4);
  grad_3d_27pt_fluid_le(le, fgrad, nextra, GRAD_DEL4);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_fluid_dab
 *
 *  This is the full gradient tensor.
 *
 *  d_x d_x phi = phi(ic+1,jc,kc) - 2phi(ic,jc,kc) + phi(ic-1,jc,kc)
 *  d_x d_y phi = 0.25*[ phi(ic+1,jc+1,kc) - phi(ic+1,jc-1,kc)
 *                     - phi(ic-1,jc+1,kc) + phi(ic-1,jc-1,kc) ]
 *  d_x d_z phi = 0.25*[ phi(ic+1,jc,kc+1) - phi(ic+1,jc,kc-1)
 *                     - phi(ic-1,jc,kc+1) + phi(ic-1,jc,kc-1) ]
 *  and so on.
 *
 *  The tensor is symmetric. The 1-d compressed storage is
 *      dab[NSYMM*index + XX] etc.
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_dab(field_grad_t * fgrad) {

  lees_edw_t * le = NULL;

  assert(fgrad);
  assert(fgrad->field);
  assert(fgrad->field->nf == 1); /* Scalars only; host only */

  le = fgrad->field->le;
  grad_3d_27pt_dab_compute(le, fgrad);
  grad_3d_27pt_dab_le_correct(le, fgrad);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_fluid_operator
 *
 *  Kernel driver.
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_fluid_operator(cs_t * cs, lees_edw_t * le,
					 field_grad_t * fg,
					 int nextra, grad_enum_t type) {
  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;
  lees_edw_t * letarget = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  lees_edw_nlocal(le, nlocal);
  lees_edw_strides(le, &xs, &ys, &zs);
  lees_edw_target(le, &letarget);

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(grad_3d_27pt_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, fg->field->nf, ys, letarget, type,
		  fg->field->target, fg->target);
  tdpDeviceSynchronize();

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_kernel
 *
 *  Unvectorised kernel to compute grad, delsq.
 *
 *****************************************************************************/

__global__ void grad_3d_27pt_kernel(kernel_ctxt_t * ktx, int nf, int ys,
				    lees_edw_t * le,
				    grad_enum_t type,
				    field_t * f,
				    field_grad_t * fgrad) {

  int kindex;
  int kiterations;
  const double r9 = (1.0/9.0);

  assert(ktx);
  assert(le);
  assert(type == GRAD_DEL2 || type == GRAD_DEL4);
  assert(f);
  assert(fgrad);

  kiterations = kernel_iterations(ktx);

  __target_simt_for(kindex, kiterations, 1) {

    int n;
    int ic, jc, kc;
    int icm1, icp1;
    int index, indexm1, indexp1;

    double * __restrict__ field;
    double * __restrict__ grad;
    double * __restrict__ del2;

    if (type == GRAD_DEL2) {
      field = f->data;
      grad = fgrad->grad;
      del2 = fgrad->delsq;
    }
    else {
      field = fgrad->delsq;
      grad = fgrad->grad_delsq;
      del2 = fgrad->delsq_delsq;
    }

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = kernel_coords_index(ktx, ic, jc, kc);

    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);
    indexm1 = lees_edw_index(le, icm1, jc, kc);
    indexp1 = lees_edw_index(le, icp1, jc, kc);

    for (n = 0; n < nf; n++) {
      grad[addr_rank2(f->nsites, nf, 3, index, n, X)] = 0.5*r9*
	(+ field[addr_rank1(f->nsites, nf, (indexp1-ys-1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1-ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1-ys  ), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1-ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1-ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1-ys+1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1   -1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1   -1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1     ), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1     ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1   +1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1   +1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys-1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1+ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys  ), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1+ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1+ys+1), n)]
	 );
      grad[addr_rank2(f->nsites, nf, 3, index, n, Y)] = 0.5*r9*
	(+ field[addr_rank1(f->nsites, nf, (indexm1+ys-1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1-ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1+ys  ), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1-ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1+ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1-ys+1), n)]
	 + field[addr_rank1(f->nsites, nf, (index  +ys-1), n)]
	 - field[addr_rank1(f->nsites, nf, (index  -ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (index  +ys  ), n)]
	 - field[addr_rank1(f->nsites, nf, (index  -ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (index  +ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (index  -ys+1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys-1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexp1-ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys  ), n)]
	 - field[addr_rank1(f->nsites, nf, (indexp1-ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexp1-ys+1), n)]
	 );
      grad[addr_rank2(f->nsites, nf, 3, index, n, Z)] = 0.5*r9*
	(+ field[addr_rank1(f->nsites, nf, (indexm1-ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1-ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1   +1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1   -1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1+ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexm1+ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (index  -ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (index  -ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (index     +1), n)]
	 - field[addr_rank1(f->nsites, nf, (index     -1), n)]
	 + field[addr_rank1(f->nsites, nf, (index  +ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (index  +ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1-ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexp1-ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1   +1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexp1   -1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys+1), n)]
	 - field[addr_rank1(f->nsites, nf, (indexp1+ys-1), n)]
	 );
      del2[addr_rank1(f->nsites, nf, index, n)] = r9*
	(+ field[addr_rank1(f->nsites, nf, (indexm1-ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1-ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1-ys+1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1   -1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1     ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1   +1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1+ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1+ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexm1+ys+1), n)]
	 + field[addr_rank1(f->nsites, nf, (index  -ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (index  -ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (index  -ys+1), n)]
	 + field[addr_rank1(f->nsites, nf, (index     -1), n)]
	 + field[addr_rank1(f->nsites, nf, (index     +1), n)]
	 + field[addr_rank1(f->nsites, nf, (index  +ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (index  +ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (index  +ys+1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1-ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1-ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1-ys+1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1   -1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1     ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1   +1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys-1), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys  ), n)]
	 + field[addr_rank1(f->nsites, nf, (indexp1+ys+1), n)]
	 - 26.0*field[addr_rank1(f->nsites, nf, index, n)]);
    }
  }

  return;
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
				   int nextra, grad_enum_t type) {

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
  assert(type == GRAD_DEL2 || type == GRAD_DEL4);

  lees_edw_nhalo(le, &nhalo);
  lees_edw_nsites(le, &nsites);
  lees_edw_nlocal(le, nlocal);

  ys = (nlocal[Z] + 2*nhalo);

  nop = fg->field->nf;
  if (type == GRAD_DEL2) {
    field = fg->field->data;
    grad = fg->grad;
    del2 = fg->delsq;
  }
  else {
    field = fg->delsq;
    grad = fg->grad_delsq;
    del2 = fg->delsq_delsq;
  }

  for (nplane = 0; nplane < lees_edw_nplane_local(le); nplane++) {

    ic = lees_edw_plane_location(le, nplane);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = lees_edw_ic_to_buff(le, ic, nh-1);
      ic1 = lees_edw_ic_to_buff(le, ic, nh  );
      ic2 = lees_edw_ic_to_buff(le, ic, nh+1);

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
      ic2 = lees_edw_ic_to_buff(le, ic, -nh+1);
      ic1 = lees_edw_ic_to_buff(le, ic, -nh  );
      ic0 = lees_edw_ic_to_buff(le, ic, -nh-1);

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
 *  grad_dab_compute
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_dab_compute(lees_edw_t * le, field_grad_t * df) {

  int nlocal[3];
  int nhalo;
  int nextra;
  int nsites;
  int ic, jc, kc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;
  double * __restrict__ dab;
  double * __restrict__ field;

  const double r12 = (1.0/12.0);

  assert(le);
  assert(df);

  lees_edw_nhalo(le, &nhalo);
  nextra = nhalo - 1;
  assert(nextra >= 0);

  lees_edw_nlocal(le, nlocal);
  lees_edw_nsites(le, &nsites);

  ys = nlocal[Z] + 2*nhalo;

  field = df->field->data;
  dab = df->d_ab;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = lees_edw_index(le, ic, jc, kc);
	indexm1 = lees_edw_index(le, icm1, jc, kc);
	indexp1 = lees_edw_index(le, icp1, jc, kc);

	dab[addr_rank1(nsites, NSYMM, index, XX)] = 0.2*
	 (+ 1.0*field[addr_rank0(nsites, indexp1)]
	  + 1.0*field[addr_rank0(nsites, indexm1)]
	  - 2.0*field[addr_rank0(nsites, index)]
	  + 1.0*field[addr_rank0(nsites, indexp1 + ys)]
	  + 1.0*field[addr_rank0(nsites, indexm1 + ys)]
	  - 2.0*field[addr_rank0(nsites, index + ys)]
	  + 1.0*field[addr_rank0(nsites, indexp1 - ys)]
	  + 1.0*field[addr_rank0(nsites, indexm1 - ys)]
	  - 2.0*field[addr_rank0(nsites, index - ys)]
	  + 1.0*field[addr_rank0(nsites, indexp1 + 1)]
	  + 1.0*field[addr_rank0(nsites, indexm1 + 1)]
	  - 2.0*field[addr_rank0(nsites, index + 1)]
	  + 1.0*field[addr_rank0(nsites, indexp1 - 1)]
	  + 1.0*field[addr_rank0(nsites, indexm1 - 1)]
	  - 2.0*field[addr_rank0(nsites, index - 1)]);
/*
	  (+ 1.0*field[addr_rank0(nsites, indexp1)]
	   + 1.0*field[addr_rank0(nsites, indexm1)]
	   - 2.0*field[addr_rank0(nsites, index)]);
*/
	dab[addr_rank1(nsites, NSYMM, index, XY)] = 
/*
-0.5* 
	 (+ 1.0*field[addr_rank0(nsites, indexp1)]
	  + 1.0*field[addr_rank0(nsites, indexm1)]
	  + 1.0*field[addr_rank0(nsites, index + ys)]
	  + 1.0*field[addr_rank0(nsites, index - ys)]
	  - 2.0*field[addr_rank0(nsites, index)]
	  - 1.0*field[addr_rank0(nsites, indexp1 + ys)]
	  - 1.0*field[addr_rank0(nsites, indexm1 - ys)]

);
*/

///*
	  r12*
	  (+ field[addr_rank0(nsites, indexp1 + ys)]
	   - field[addr_rank0(nsites, indexp1 - ys)]
	   - field[addr_rank0(nsites, indexm1 + ys)]
	   + field[addr_rank0(nsites, indexm1 - ys)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 - ys + 1)]
	   - field[addr_rank0(nsites, indexm1 + ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys + 1)]

	   + field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexp1 - ys - 1)]
	   - field[addr_rank0(nsites, indexm1 + ys - 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);
//*/

	dab[addr_rank1(nsites, NSYMM, index, XZ)] = 
/*
-0.5* 
	 (+ 1.0*field[addr_rank0(nsites, indexp1)]
	  + 1.0*field[addr_rank0(nsites, indexm1)]
	  + 1.0*field[addr_rank0(nsites, index + 1)]
	  + 1.0*field[addr_rank0(nsites, index - 1)]
	  - 2.0*field[addr_rank0(nsites, index)]
	  - 1.0*field[addr_rank0(nsites, indexp1 + 1)]
	  - 1.0*field[addr_rank0(nsites, indexm1 - 1)]);
*/
///*
	  r12*
	  (+ field[addr_rank0(nsites, indexp1 + 1)]
	   - field[addr_rank0(nsites, indexp1 - 1)]
	   - field[addr_rank0(nsites, indexm1 + 1)]
	   + field[addr_rank0(nsites, indexm1 - 1)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexm1 + ys + 1)]
	   + field[addr_rank0(nsites, indexm1 + ys - 1)]

	   + field[addr_rank0(nsites, indexp1 - ys + 1)]
	   - field[addr_rank0(nsites, indexp1 - ys - 1)]
	   - field[addr_rank0(nsites, indexm1 - ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);
//*/

	dab[addr_rank1(nsites, NSYMM, index, YY)] = 0.2*
	 (+ 1.0*field[addr_rank0(nsites, index + ys)]
	  + 1.0*field[addr_rank0(nsites, index - ys)]
	  - 2.0*field[addr_rank0(nsites, index)]
	  + 1.0*field[addr_rank0(nsites, indexp1 + ys)]
	  + 1.0*field[addr_rank0(nsites, indexp1 - ys)]
	  - 2.0*field[addr_rank0(nsites, indexp1)]
	  + 1.0*field[addr_rank0(nsites, indexm1 + ys)]
	  + 1.0*field[addr_rank0(nsites, indexm1 - ys)]
	  - 2.0*field[addr_rank0(nsites, indexm1)]
	  + 1.0*field[addr_rank0(nsites, index + 1 + ys)]
	  + 1.0*field[addr_rank0(nsites, index + 1 - ys)]
	  - 2.0*field[addr_rank0(nsites, index + 1 )]
	  + 1.0*field[addr_rank0(nsites, index - 1 + ys)]
	  + 1.0*field[addr_rank0(nsites, index - 1 - ys)]
	  - 2.0*field[addr_rank0(nsites, index - 1 )]);
/*
	  (+ 1.0*field[addr_rank0(nsites, index + ys)]
	   + 1.0*field[addr_rank0(nsites, index - ys)]
	   - 2.0*field[addr_rank0(nsites, index)]);
*/
	dab[addr_rank1(nsites, NSYMM, index, YZ)] = 
/*
-0.5*
	 (+ 1.0*field[addr_rank0(nsites, index + ys)]
	  + 1.0*field[addr_rank0(nsites, index - ys)]
	  + 1.0*field[addr_rank0(nsites, index + 1)]
	  + 1.0*field[addr_rank0(nsites, index - 1)]
	  - 2.0*field[addr_rank0(nsites, index)]
	  - 1.0*field[addr_rank0(nsites, index + ys + 1)]
	  - 1.0*field[addr_rank0(nsites, index - ys - 1)]);
*/
///*
	  r12*
	  (+ field[addr_rank0(nsites, index + ys + 1)]
	   - field[addr_rank0(nsites, index + ys - 1)]
	   - field[addr_rank0(nsites, index - ys + 1)]
	   + field[addr_rank0(nsites, index - ys - 1)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexp1 - ys + 1)]
	   + field[addr_rank0(nsites, indexp1 - ys - 1)]

	   + field[addr_rank0(nsites, indexm1 + ys + 1)]
	   - field[addr_rank0(nsites, indexm1 + ys - 1)]
	   - field[addr_rank0(nsites, indexm1 - ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);
//*/

	dab[addr_rank1(nsites, NSYMM, index, ZZ)] = 0.2*
	 (+ 1.0*field[addr_rank0(nsites, index + 1)]
	  + 1.0*field[addr_rank0(nsites, index - 1)]
	  - 2.0*field[addr_rank0(nsites, index)]
	  + 1.0*field[addr_rank0(nsites, indexp1 + 1)]
	  + 1.0*field[addr_rank0(nsites, indexp1 - 1)]
	  - 2.0*field[addr_rank0(nsites, indexp1)]
	  + 1.0*field[addr_rank0(nsites, indexm1 + 1)]
	  + 1.0*field[addr_rank0(nsites, indexm1 - 1)]
	  - 2.0*field[addr_rank0(nsites, indexm1)]
	  + 1.0*field[addr_rank0(nsites, index + ys + 1)]
	  + 1.0*field[addr_rank0(nsites, index + ys - 1)]
	  - 2.0*field[addr_rank0(nsites, index + ys)]
	  + 1.0*field[addr_rank0(nsites, index - ys + 1)]
	  + 1.0*field[addr_rank0(nsites, index - ys - 1)]
	  - 2.0*field[addr_rank0(nsites, index - ys)]);
/*
	  (+ 1.0*field[addr_rank0(nsites, index + 1)]
	   + 1.0*field[addr_rank0(nsites, index - 1)]
	   - 2.0*field[addr_rank0(nsites, index)]);
*/

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_dab_le_correct
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_dab_le_correct(lees_edw_t * le, field_grad_t * df) {

  int nlocal[3];
  int nhalo;
  int nextra;
  int nsites;
  int nh;                                 /* counter over halo extent */
  int nplane;                             /* Number LE planes */
  int ic, jc, kc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */
  double * __restrict__ dab;
  double * __restrict__ field;

  const double r12 = (1.0/12.0);

  assert(le);
  assert(df);

  lees_edw_nhalo(le, &nhalo);
  lees_edw_nlocal(le, nlocal);
  lees_edw_nsites(le, &nsites);
  ys = (nlocal[Z] + 2*nhalo);

  nextra = nhalo - 1;
  assert(nextra >= 0);

  dab = df->d_ab;
  field = df->field->data;

  for (nplane = 0; nplane < lees_edw_nplane_local(le); nplane++) {

    ic = lees_edw_plane_location(le, nplane);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = lees_edw_ic_to_buff(le, ic, nh-1);
      ic1 = lees_edw_ic_to_buff(le, ic, nh  );
      ic2 = lees_edw_ic_to_buff(le, ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = lees_edw_index(le, ic0, jc, kc);
	  index   = lees_edw_index(le, ic1, jc, kc);
	  indexp1 = lees_edw_index(le, ic2, jc, kc);

	  dab[addr_rank1(nsites, NSYMM, index, XX)] = 0.2*
	   (+ 1.0*field[addr_rank0(nsites, indexp1)]
	    + 1.0*field[addr_rank0(nsites, indexm1)]
	    - 2.0*field[addr_rank0(nsites, index)]
	    + 1.0*field[addr_rank0(nsites, indexp1 + ys)]
	    + 1.0*field[addr_rank0(nsites, indexm1 + ys)]
	    - 2.0*field[addr_rank0(nsites, index + ys)]
	    + 1.0*field[addr_rank0(nsites, indexp1 - ys)]
	    + 1.0*field[addr_rank0(nsites, indexm1 - ys)]
	    - 2.0*field[addr_rank0(nsites, index - ys)]
	    + 1.0*field[addr_rank0(nsites, indexp1 + 1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 + 1)]
	    - 2.0*field[addr_rank0(nsites, index + 1)]
	    + 1.0*field[addr_rank0(nsites, indexp1 - 1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 - 1)]
	    - 2.0*field[addr_rank0(nsites, index - 1)]);
/*
	    (+ 1.0*field[addr_rank0(nsites, indexp1)]
	     + 1.0*field[addr_rank0(nsites, indexm1)]
	     - 2.0*field[addr_rank0(nsites, index)]);
*/

	  dab[addr_rank1(nsites, NSYMM, index, XY)] = 

	  r12*
	  (+ field[addr_rank0(nsites, indexp1 + ys)]
	   - field[addr_rank0(nsites, indexp1 - ys)]
	   - field[addr_rank0(nsites, indexm1 + ys)]
	   + field[addr_rank0(nsites, indexm1 - ys)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 - ys + 1)]
	   - field[addr_rank0(nsites, indexm1 + ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys + 1)]

	   + field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexp1 - ys - 1)]
	   - field[addr_rank0(nsites, indexm1 + ys - 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);
/*
-0.5*

         (+ 1.0*field[addr_rank0(nsites, indexp1)]
          + 1.0*field[addr_rank0(nsites, indexm1)]
          + 1.0*field[addr_rank0(nsites, index + ys)]
          + 1.0*field[addr_rank0(nsites, index - ys)]
          - 2.0*field[addr_rank0(nsites, index)]
          - 1.0*field[addr_rank0(nsites, indexp1 + ys)]
          - 1.0*field[addr_rank0(nsites, indexm1 - ys)]);
*/
/*
0.25*
	    (+ field[addr_rank0(nsites, indexp1 + ys)]
	     - field[addr_rank0(nsites, indexp1 - ys)]
	     - field[addr_rank0(nsites, indexm1 + ys)]
	     + field[addr_rank0(nsites, indexm1 - ys)]);
*/

	  dab[addr_rank1(nsites, NSYMM, index, XZ)] = 

	  r12*
	  (+ field[addr_rank0(nsites, indexp1 + 1)]
	   - field[addr_rank0(nsites, indexp1 - 1)]
	   - field[addr_rank0(nsites, indexm1 + 1)]
	   + field[addr_rank0(nsites, indexm1 - 1)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexm1 + ys + 1)]
	   + field[addr_rank0(nsites, indexm1 + ys - 1)]

	   + field[addr_rank0(nsites, indexp1 - ys + 1)]
	   - field[addr_rank0(nsites, indexp1 - ys - 1)]
	   - field[addr_rank0(nsites, indexm1 - ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);

/*
-0.5*

         (+ 1.0*field[addr_rank0(nsites, indexp1)]
          + 1.0*field[addr_rank0(nsites, indexm1)]
          + 1.0*field[addr_rank0(nsites, index + 1)]
          + 1.0*field[addr_rank0(nsites, index - 1)]
          - 2.0*field[addr_rank0(nsites, index)]
          - 1.0*field[addr_rank0(nsites, indexp1 + 1)]
          - 1.0*field[addr_rank0(nsites, indexm1 - 1)]);
*/
/*
0.25*
	    (+ field[addr_rank0(nsites, indexp1 + 1)]
	     - field[addr_rank0(nsites, indexp1 - 1)]
	     - field[addr_rank0(nsites, indexm1 + 1)]
	     + field[addr_rank0(nsites, indexm1 - 1)]);
*/

	  dab[addr_rank1(nsites, NSYMM, index, YY)] = 0.2*
	   (+ 1.0*field[addr_rank0(nsites, index + ys)]
	    + 1.0*field[addr_rank0(nsites, index - ys)]
	    - 2.0*field[addr_rank0(nsites, index)]
	    + 1.0*field[addr_rank0(nsites, indexp1 + ys)]
	    + 1.0*field[addr_rank0(nsites, indexp1 - ys)]
	    - 2.0*field[addr_rank0(nsites, indexp1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 + ys)]
	    + 1.0*field[addr_rank0(nsites, indexm1 - ys)]
	    - 2.0*field[addr_rank0(nsites, indexm1)]
	    + 1.0*field[addr_rank0(nsites, index + 1 + ys)]
	    + 1.0*field[addr_rank0(nsites, index + 1 - ys)]
	    - 2.0*field[addr_rank0(nsites, index + 1 )]
	    + 1.0*field[addr_rank0(nsites, index - 1 + ys)]
	    + 1.0*field[addr_rank0(nsites, index - 1 - ys)]
	    - 2.0*field[addr_rank0(nsites, index - 1 )]);

/*	    (+ 1.0*field[addr_rank0(nsites, index + ys)]
	     + 1.0*field[addr_rank0(nsites, index - ys)]
	     - 2.0*field[addr_rank0(nsites, index)]);
*/
	  dab[addr_rank1(nsites, NSYMM, index, YZ)] = 

	  r12*
	  (+ field[addr_rank0(nsites, index + ys + 1)]
	   - field[addr_rank0(nsites, index + ys - 1)]
	   - field[addr_rank0(nsites, index - ys + 1)]
	   + field[addr_rank0(nsites, index - ys - 1)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexp1 - ys + 1)]
	   + field[addr_rank0(nsites, indexp1 - ys - 1)]

	   + field[addr_rank0(nsites, indexm1 + ys + 1)]
	   - field[addr_rank0(nsites, indexm1 + ys - 1)]
	   - field[addr_rank0(nsites, indexm1 - ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);

/*
-0.5*

         (+ 1.0*field[addr_rank0(nsites, index + ys)]
          + 1.0*field[addr_rank0(nsites, index - ys)]
          + 1.0*field[addr_rank0(nsites, index + 1)]
          + 1.0*field[addr_rank0(nsites, index - 1)]
          - 2.0*field[addr_rank0(nsites, index)]
          - 1.0*field[addr_rank0(nsites, index + ys + 1)]
          - 1.0*field[addr_rank0(nsites, index - ys - 1)]);
*/
/*
0.25*
	    (+ field[addr_rank0(nsites, index + ys + 1)]
	     - field[addr_rank0(nsites, index + ys - 1)]
	     - field[addr_rank0(nsites, index - ys + 1)]
	     + field[addr_rank0(nsites, index - ys - 1)]);
*/
	  dab[addr_rank1(nsites, NSYMM, index, ZZ)] = 0.2*
	   (+ 1.0*field[addr_rank0(nsites, index + 1)]
	    + 1.0*field[addr_rank0(nsites, index - 1)]
	    - 2.0*field[addr_rank0(nsites, index)]
	    + 1.0*field[addr_rank0(nsites, indexp1 + 1)]
	    + 1.0*field[addr_rank0(nsites, indexp1 - 1)]
	    - 2.0*field[addr_rank0(nsites, indexp1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 + 1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 - 1)]
	    - 2.0*field[addr_rank0(nsites, indexm1)]
	    + 1.0*field[addr_rank0(nsites, index + ys + 1)]
	    + 1.0*field[addr_rank0(nsites, index + ys - 1)]
	    - 2.0*field[addr_rank0(nsites, index + ys)]
	    + 1.0*field[addr_rank0(nsites, index - ys + 1)]
	    + 1.0*field[addr_rank0(nsites, index - ys - 1)]
	    - 2.0*field[addr_rank0(nsites, index - ys)]);

/*
	    (+ 1.0*field[addr_rank0(nsites, index + 1)]
	     + 1.0*field[addr_rank0(nsites, index - 1)]
	     - 2.0*field[addr_rank0(nsites, index)]);
*/

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
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = lees_edw_index(le, ic0, jc, kc);
	  index   = lees_edw_index(le, ic1, jc, kc);
	  indexp1 = lees_edw_index(le, ic2, jc, kc);

	  dab[addr_rank1(nsites, NSYMM, index, XX)] = 0.2*
	   (+ 1.0*field[addr_rank0(nsites, indexp1)]
	    + 1.0*field[addr_rank0(nsites, indexm1)]
	    - 2.0*field[addr_rank0(nsites, index)]
	    + 1.0*field[addr_rank0(nsites, indexp1 + ys)]
	    + 1.0*field[addr_rank0(nsites, indexm1 + ys)]
	    - 2.0*field[addr_rank0(nsites, index + ys)]
	    + 1.0*field[addr_rank0(nsites, indexp1 - ys)]
	    + 1.0*field[addr_rank0(nsites, indexm1 - ys)]
	    - 2.0*field[addr_rank0(nsites, index - ys)]
	    + 1.0*field[addr_rank0(nsites, indexp1 + 1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 + 1)]
	    - 2.0*field[addr_rank0(nsites, index + 1)]
	    + 1.0*field[addr_rank0(nsites, indexp1 - 1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 - 1)]
	    - 2.0*field[addr_rank0(nsites, index - 1)]);
/*
	    (+ 1.0*field[addr_rank0(nsites, indexp1)]
	     + 1.0*field[addr_rank0(nsites, indexm1)]
	     - 2.0*field[addr_rank0(nsites, index)]);
*/

	  dab[addr_rank1(nsites, NSYMM, index, XY)] =
	  r12*
	  (+ field[addr_rank0(nsites, indexp1 + ys)]
	   - field[addr_rank0(nsites, indexp1 - ys)]
	   - field[addr_rank0(nsites, indexm1 + ys)]
	   + field[addr_rank0(nsites, indexm1 - ys)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 - ys + 1)]
	   - field[addr_rank0(nsites, indexm1 + ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys + 1)]

	   + field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexp1 - ys - 1)]
	   - field[addr_rank0(nsites, indexm1 + ys - 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);
/*
 -0.5*
         (+ 1.0*field[addr_rank0(nsites, indexp1)]
          + 1.0*field[addr_rank0(nsites, indexm1)]
          + 1.0*field[addr_rank0(nsites, index + ys)]
          + 1.0*field[addr_rank0(nsites, index - ys)]
          - 2.0*field[addr_rank0(nsites, index)]
          - 1.0*field[addr_rank0(nsites, indexp1 + ys)]
          - 1.0*field[addr_rank0(nsites, indexm1 - ys)]);
*/
/*
0.25*
	    (+ field[addr_rank0(nsites, indexp1 + ys)]
	     - field[addr_rank0(nsites, indexp1 - ys)]
	     - field[addr_rank0(nsites, indexm1 + ys)]
	     + field[addr_rank0(nsites, indexm1 - ys)]);
*/

	  dab[addr_rank1(nsites, NSYMM, index, XZ)] = 

	  r12*
	  (+ field[addr_rank0(nsites, indexp1 + 1)]
	   - field[addr_rank0(nsites, indexp1 - 1)]
	   - field[addr_rank0(nsites, indexm1 + 1)]
	   + field[addr_rank0(nsites, indexm1 - 1)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexm1 + ys + 1)]
	   + field[addr_rank0(nsites, indexm1 + ys - 1)]

	   + field[addr_rank0(nsites, indexp1 - ys + 1)]
	   - field[addr_rank0(nsites, indexp1 - ys - 1)]
	   - field[addr_rank0(nsites, indexm1 - ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);

/*
-0.5*

         (+ 1.0*field[addr_rank0(nsites, indexp1)]
          + 1.0*field[addr_rank0(nsites, indexm1)]
          + 1.0*field[addr_rank0(nsites, index + 1)]
          + 1.0*field[addr_rank0(nsites, index - 1)]
          - 2.0*field[addr_rank0(nsites, index)]
          - 1.0*field[addr_rank0(nsites, indexp1 + 1)]
          - 1.0*field[addr_rank0(nsites, indexm1 - 1)]);
*/
/*
0.25*
	    (+ field[addr_rank0(nsites, indexp1 + 1)]
	     - field[addr_rank0(nsites, indexp1 - 1)]
	     - field[addr_rank0(nsites, indexm1 + 1)]
	     + field[addr_rank0(nsites, indexm1 - 1)]);
*/

	  dab[addr_rank1(nsites, NSYMM, index, YY)] = 0.2*
	   (+ 1.0*field[addr_rank0(nsites, index + ys)]
	    + 1.0*field[addr_rank0(nsites, index - ys)]
	    - 2.0*field[addr_rank0(nsites, index)]
	    + 1.0*field[addr_rank0(nsites, indexp1 + ys)]
	    + 1.0*field[addr_rank0(nsites, indexp1 - ys)]
	    - 2.0*field[addr_rank0(nsites, indexp1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 + ys)]
	    + 1.0*field[addr_rank0(nsites, indexm1 - ys)]
	    - 2.0*field[addr_rank0(nsites, indexm1)]
	    + 1.0*field[addr_rank0(nsites, index + 1 + ys)]
	    + 1.0*field[addr_rank0(nsites, index + 1 - ys)]
	    - 2.0*field[addr_rank0(nsites, index + 1 )]
	    + 1.0*field[addr_rank0(nsites, index - 1 + ys)]
	    + 1.0*field[addr_rank0(nsites, index - 1 - ys)]
	    - 2.0*field[addr_rank0(nsites, index - 1 )]);
/*
	    (+ 1.0*field[addr_rank0(nsites, index + ys)]
	     + 1.0*field[addr_rank0(nsites, index - ys)]
	     - 2.0*field[addr_rank0(nsites, index)]);
*/

	  dab[addr_rank1(nsites, NSYMM, index, YZ)] = 

	  r12*
	  (+ field[addr_rank0(nsites, index + ys + 1)]
	   - field[addr_rank0(nsites, index + ys - 1)]
	   - field[addr_rank0(nsites, index - ys + 1)]
	   + field[addr_rank0(nsites, index - ys - 1)]

	   + field[addr_rank0(nsites, indexp1 + ys + 1)]
	   - field[addr_rank0(nsites, indexp1 + ys - 1)]
	   - field[addr_rank0(nsites, indexp1 - ys + 1)]
	   + field[addr_rank0(nsites, indexp1 - ys - 1)]

	   + field[addr_rank0(nsites, indexm1 + ys + 1)]
	   - field[addr_rank0(nsites, indexm1 + ys - 1)]
	   - field[addr_rank0(nsites, indexm1 - ys + 1)]
	   + field[addr_rank0(nsites, indexm1 - ys - 1)]);

/*
-0.5*

         (+ 1.0*field[addr_rank0(nsites, index + ys)]
          + 1.0*field[addr_rank0(nsites, index - ys)]
          + 1.0*field[addr_rank0(nsites, index + 1)]
          + 1.0*field[addr_rank0(nsites, index - 1)]
          - 2.0*field[addr_rank0(nsites, index)]
          - 1.0*field[addr_rank0(nsites, index + ys + 1)]
          - 1.0*field[addr_rank0(nsites, index - ys - 1)]);
*/
/*
0.25*
	    (+ field[addr_rank0(nsites, index + ys + 1)]
	     - field[addr_rank0(nsites, index + ys - 1)]
	     - field[addr_rank0(nsites, index - ys + 1)]
	     + field[addr_rank0(nsites, index - ys - 1)]);
*/

	  dab[addr_rank1(nsites, NSYMM, index, ZZ)] = 0.2*
	   (+ 1.0*field[addr_rank0(nsites, index + 1)]
	    + 1.0*field[addr_rank0(nsites, index - 1)]
	    - 2.0*field[addr_rank0(nsites, index)]
	    + 1.0*field[addr_rank0(nsites, indexp1 + 1)]
	    + 1.0*field[addr_rank0(nsites, indexp1 - 1)]
	    - 2.0*field[addr_rank0(nsites, indexp1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 + 1)]
	    + 1.0*field[addr_rank0(nsites, indexm1 - 1)]
	    - 2.0*field[addr_rank0(nsites, indexm1)]
	    + 1.0*field[addr_rank0(nsites, index + ys + 1)]
	    + 1.0*field[addr_rank0(nsites, index + ys - 1)]
	    - 2.0*field[addr_rank0(nsites, index + ys)]
	    + 1.0*field[addr_rank0(nsites, index - ys + 1)]
	    + 1.0*field[addr_rank0(nsites, index - ys - 1)]
	    - 2.0*field[addr_rank0(nsites, index - ys)]);
/*
	    (+ 1.0*field[addr_rank0(nsites, index + 1)]
	     + 1.0*field[addr_rank0(nsites, index - 1)]
	     - 2.0*field[addr_rank0(nsites, index)]);
*/

	}
      }
    }
    /* Next plane */
  }

  return 0;
}

