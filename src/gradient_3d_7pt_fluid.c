/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid.c
 *
 *  Gradient operations for 3D seven point stencil.
 *
 *                        (ic, jc+1, kc)
 *         (ic-1, jc, kc) (ic, jc  , kc) (ic+1, jc, kc)
 *                        (ic, jc-1, kc)
 *
 *  ...and so in z-direction
 *
 *  d_x phi = [phi(ic+1,jc,kc) - phi(ic-1,jc,kc)] / 2
 *  d_y phi = [phi(ic,jc+1,kc) - phi(ic,jc-1,kc)] / 2
 *  d_z phi = [phi(ic,jc,kc+1) - phi(ic,jc,kc-1)] / 2
 *
 *  nabla^2 phi = phi(ic+1,jc,kc) + phi(ic-1,jc,kc)
 *              + phi(ic,jc+1,kc) + phi(ic,jc-1,kc)
 *              + phi(ic,jc,kc+1) + phi(ic,jc,kc-1)
 *              - 6 phi(ic,jc,kc)
 *
 *  Corrections for Lees-Edwards planes are included.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "leesedwards.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "timer.h"
#include "gradient_3d_7pt_fluid.h"

#define NSTENCIL 1 /* +/- 1 point in each direction */

__host__ int grad_3d_7pt_fluid_operator(cs_t * cs, lees_edw_t * le,
					field_grad_t * fg,
					int nextra);
__host__ int grad_3d_7pt_fluid_le(lees_edw_t * le, field_grad_t * fg,
				  int nextra);

__host__ int grad_3d_7pt_dab_le_correct(lees_edw_t * le, field_grad_t * df,
					int nextra);
__host__ int grad_3d_7pt_dab_compute(cs_t * cs, lees_edw_t * le,
				     field_grad_t * df,
				     int nextra);


__global__
void grad_3d_7pt_fluid_kernel_v(kernel_ctxt_t * ktx, int nop, int ys,
				lees_edw_t * le,
				field_t * field, 
				field_grad_t * fgrad);
__global__
void grad_3d_7pt_dab_kernel_v(kernel_ctxt_t * ktx, lees_edw_t * le,
			      field_grad_t * df, int nsites,
			      int ys);

/*****************************************************************************
 *
 *  grad_3d_7pt_fluid_d2
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_fluid_d2(field_grad_t * fgrad) {

  int nhalo, nextra;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  assert(fgrad);
  assert(fgrad->field);
  assert(fgrad->field->cs);
  assert(fgrad->field->le);

  cs = fgrad->field->cs;
  le = fgrad->field->le;
  lees_edw_nhalo(le, &nhalo);

  nextra = nhalo - NSTENCIL;
  assert(nextra >= 0);

  grad_3d_7pt_fluid_operator(cs, le, fgrad, nextra);
  grad_3d_7pt_fluid_le(le, fgrad, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_7pt_fluid_d4
 *
 *  TODO:
 *  The assert(0) indicates refactoring is required to make the
 *  extra derivative (cf, 2d_5pt etc). There's no test.
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_fluid_d4(field_grad_t * fgrad) {

  int nhalo, nextra;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  assert(fgrad);
  assert(fgrad->field);
  assert(fgrad->field->cs);
  assert(fgrad->field->le);

  cs = fgrad->field->cs;
  le = fgrad->field->le;
  lees_edw_nhalo(le, &nhalo);

  nextra = nhalo - 2*NSTENCIL;
  assert(nextra >= 0);

  assert(0); /* NO TEST? */
  grad_3d_7pt_fluid_operator(cs, le, fgrad, nextra);
  grad_3d_7pt_fluid_le(le, fgrad, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_7pt_fluid_dab
 *
 *  This is the full gradient tensor, which actually requires more
 *  than the 7-point stencil advertised.
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

__host__ int grad_3d_7pt_fluid_dab(field_grad_t * fgrad) {

  int nhalo;
  int nextra;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  assert(fgrad);
  assert(fgrad->field);
  assert(fgrad->field->le);
  assert(fgrad->field->nf == 1); /* Scalars only */
  
  cs = fgrad->field->cs;
  le = fgrad->field->le;
  lees_edw_nhalo(le, &nhalo);

  nextra = nhalo - NSTENCIL;
  assert(nextra >= 0);

  grad_3d_7pt_dab_compute(cs, le, fgrad, nextra);
  grad_3d_7pt_dab_le_correct(le, fgrad, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_7pt_fluid_operator
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_fluid_operator(cs_t * cs, lees_edw_t * le,
					field_grad_t * fg,
					int nextra) {

  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;
  lees_edw_t * letarget = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(le);

  lees_edw_nlocal(le, nlocal);
  lees_edw_strides(le, &xs, &ys, &zs);
  lees_edw_target(le, &letarget);

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(TIMER_PHI_GRAD_KERNEL);

  tdpLaunchKernel(grad_3d_7pt_fluid_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target,
		  fg->field->nf, ys, letarget, fg->field->target, fg->target);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  TIMER_stop(TIMER_PHI_GRAD_KERNEL);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_7pt_fluid_kernel_v
 *
 *  Compute grad, delsq.
 *
 *****************************************************************************/

__global__
void grad_3d_7pt_fluid_kernel_v(kernel_ctxt_t * ktx, int nf, int ys,
				lees_edw_t * le,
				field_t * field, 
				field_grad_t * fgrad) {

  int kindex;
  int kiterations;

  assert(ktx);
  assert(le);
  assert(field);
  assert(fgrad);

  kiterations = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiterations, NSIMDVL) {

    int n;
    int iv;
    int index;

    int ic[NSIMDVL], jc[NSIMDVL], kc[NSIMDVL];
    int im1[NSIMDVL];
    int ip1[NSIMDVL];
    int indexm1[NSIMDVL];
    int indexp1[NSIMDVL];
    int maskv[NSIMDVL];

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    index = kernel_baseindex(ktx, kindex);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    for_simd_v(iv, NSIMDVL) im1[iv] = lees_edw_ic_to_buff(le, ic[iv], -1);
    for_simd_v(iv, NSIMDVL) ip1[iv] = lees_edw_ic_to_buff(le, ic[iv], +1);

    kernel_coords_index_v(ktx, im1, jc, kc, indexm1);
    kernel_coords_index_v(ktx, ip1, jc, kc, indexp1);

    for (n = 0; n < nf; n++) {
      for_simd_v(iv, NSIMDVL) { 
	if (maskv[iv]) {
	  fgrad->grad[addr_rank2(fgrad->nsite,nf,3,index+iv,n,X)] = 0.5*
	    (field->data[addr_rank1(field->nsites,nf,indexp1[iv],n)] -
	     field->data[addr_rank1(field->nsites,nf,indexm1[iv],n)]); 
	}
      }
      
      for_simd_v(iv, NSIMDVL) { 
	if (maskv[iv]) {
	  fgrad->grad[addr_rank2(fgrad->nsite,nf,3,index+iv,n,Y)] = 0.5*
	    (field->data[addr_rank1(field->nsites,nf,index+iv+ys,n)] -
	     field->data[addr_rank1(field->nsites,nf,index+iv-ys,n)]);
	}
      }
      
      for_simd_v(iv, NSIMDVL) { 
	if (maskv[iv]) {
	  fgrad->grad[addr_rank2(fgrad->nsite,nf,3,index+iv,n,Z)] = 0.5*
	    (field->data[addr_rank1(field->nsites,nf,index+iv+1,n)] -
	     field->data[addr_rank1(field->nsites,nf,index+iv-1,n)]);
	}
      }

      for_simd_v(iv, NSIMDVL) { 
	if (maskv[iv]) {
	  fgrad->delsq[addr_rank1(fgrad->nsite,nf,index+iv,n)]
	    = field->data[addr_rank1(field->nsites,nf,indexp1[iv],n)]
	    + field->data[addr_rank1(field->nsites,nf,indexm1[iv],n)]
	    + field->data[addr_rank1(field->nsites,nf,index+iv+ys,n)]
	    + field->data[addr_rank1(field->nsites,nf,index+iv-ys,n)]
	    + field->data[addr_rank1(field->nsites,nf,index+iv+1,n)]
	    + field->data[addr_rank1(field->nsites,nf,index+iv-1,n)]
	    - 6.0*field->data[addr_rank1(field->nsites,nf,index+iv,n)];
	}
      }
    }
    /* Next sites */
  }

  return;
}


/*****************************************************************************
 *
 *  grad_3d_7pt_le
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_fluid_le(lees_edw_t * le, field_grad_t * fg,
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

  double * __restrict__ grad;
  double * __restrict__ del2;
  double * __restrict__ field;

  assert(le);
  assert(fg);

  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);
  lees_edw_nsites(le, &nsites);

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
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = lees_edw_index(le, ic0, jc, kc);
	  index   = lees_edw_index(le, ic1, jc, kc);
	  indexp1 = lees_edw_index(le, ic2, jc, kc);

	  for (n = 0; n < nop; n++) {
	    grad[addr_rank2(nsites, nop, 3, index, n, X)]
	      = 0.5*(field[addr_rank1(nsites, nop, indexp1, n)]
		   - field[addr_rank1(nsites, nop, indexm1, n)]);
	    grad[addr_rank2(nsites, nop, 3, index, n, Y)]
	      = 0.5*(field[addr_rank1(nsites, nop, (index + ys), n)]
		   - field[addr_rank1(nsites, nop, (index - ys), n)]);
	    grad[addr_rank2(nsites, nop, 3, index, n, Z)]
	      = 0.5*(field[addr_rank1(nsites, nop, (index + 1), n)]
		   - field[addr_rank1(nsites, nop, (index - 1), n)]);
	    del2[addr_rank1(nsites, nop, index, n)]
	      = field[addr_rank1(nsites, nop, indexp1,      n)]
	      + field[addr_rank1(nsites, nop, indexm1,      n)]
	      + field[addr_rank1(nsites, nop, (index + ys), n)]
	      + field[addr_rank1(nsites, nop, (index - ys), n)]
	      + field[addr_rank1(nsites, nop, (index + 1),  n)]
	      + field[addr_rank1(nsites, nop, (index - 1),  n)]
	      - 6.0*field[addr_rank1(nsites, nop, index, n)];
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
	    grad[addr_rank2(nsites, nop, 3, index, n, X)]
	      = 0.5*(field[addr_rank1(nsites, nop, indexp1, n)]
		   - field[addr_rank1(nsites, nop, indexm1, n)]);
	    grad[addr_rank2(nsites, nop, 3, index, n, Y)]
	      = 0.5*(field[addr_rank1(nsites, nop, (index + ys), n)]
		   - field[addr_rank1(nsites, nop, (index - ys), n)]);
	    grad[addr_rank2(nsites, nop, 3, index, n, Z)]
	      = 0.5*(field[addr_rank1(nsites, nop, (index + 1), n)]
		   - field[addr_rank1(nsites, nop, (index - 1), n)]);
	    del2[addr_rank1(nsites, nop, index, n)]
	      = field[addr_rank1(nsites, nop, indexp1,       n)]
	      + field[addr_rank1(nsites, nop, indexm1,       n)]
	      + field[addr_rank1(nsites, nop, (index + ys),  n)]
	      + field[addr_rank1(nsites, nop, (index - ys),  n)]
	      + field[addr_rank1(nsites, nop, (index + 1),   n)]
	      + field[addr_rank1(nsites, nop, (index - 1),   n)]
	      - 6.0*field[addr_rank1(nsites, nop, index, n)];
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
 *  grad_3d_7pt_dab_compute
 *
 *  Kernel driver.
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_dab_compute(cs_t * cs, lees_edw_t * le,
				     field_grad_t * df,
				     int nextra) {
  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;
  lees_edw_t * letarget = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);
  assert(le);
  assert(df);

  lees_edw_nlocal(le, nlocal);
  lees_edw_strides(le, &xs, &ys, &zs);
  lees_edw_target(le, &letarget);

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(grad_3d_7pt_dab_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, letarget, df->target, df->field->nsites, ys);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_7pt_dab_kernel_v
 *
 *  Vectorised kernel for d_a d_b field
 *
 *****************************************************************************/

__global__ void grad_3d_7pt_dab_kernel_v(kernel_ctxt_t * ktx, lees_edw_t * le,
					 field_grad_t * df, int nsites,
					 int ys) {
  int kindex;
  int kiterations;
  int index;
  double * __restrict__ dab;
  double * __restrict__ field;

  assert(ktx);
  assert(le);
  assert(df);

  field = df->field->data;
  dab = df->d_ab;

  kiterations = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiterations, NSIMDVL) {

    int iv;
    int ic[NSIMDVL], jc[NSIMDVL], kc[NSIMDVL];
    int im1[NSIMDVL];
    int ip1[NSIMDVL];
    int indexm1[NSIMDVL];
    int indexp1[NSIMDVL];
    int maskv[NSIMDVL];

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    index = kernel_baseindex(ktx, kindex);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    for_simd_v(iv, NSIMDVL) {
      im1[iv] = lees_edw_ic_to_buff(le, ic[iv], -1);
    }
    for_simd_v(iv, NSIMDVL) {
      ip1[iv] = lees_edw_ic_to_buff(le, ic[iv], +1);
    }

    kernel_coords_index_v(ktx, im1, jc, kc, indexm1);
    kernel_coords_index_v(ktx, ip1, jc, kc, indexp1);

    for_simd_v(iv, NSIMDVL) { 
      if (maskv[iv]) {
	dab[addr_rank1(nsites, NSYMM, index+iv, XX)] =
	  (+ 1.0*field[addr_rank0(nsites, indexp1[iv])]
	   + 1.0*field[addr_rank0(nsites, indexm1[iv])]
	   - 2.0*field[addr_rank0(nsites, index + iv)]);

	dab[addr_rank1(nsites, NSYMM, index+iv, XY)] = 0.25*
	  (+ field[addr_rank0(nsites, indexp1[iv] + ys)]
	   - field[addr_rank0(nsites, indexp1[iv] - ys)]
	   - field[addr_rank0(nsites, indexm1[iv] + ys)]
	   + field[addr_rank0(nsites, indexm1[iv] - ys)]);

	dab[addr_rank1(nsites, NSYMM, index+iv, XZ)] = 0.25*
	  (+ field[addr_rank0(nsites, indexp1[iv] + 1)]
	   - field[addr_rank0(nsites, indexp1[iv] - 1)]
	   - field[addr_rank0(nsites, indexm1[iv] + 1)]
	   + field[addr_rank0(nsites, indexm1[iv] - 1)]);

	dab[addr_rank1(nsites, NSYMM, index+iv, YY)] =
	  (+ 1.0*field[addr_rank0(nsites, index+iv + ys)]
	   + 1.0*field[addr_rank0(nsites, index+iv - ys)]
	   - 2.0*field[addr_rank0(nsites, index+iv     )]);

	dab[addr_rank1(nsites, NSYMM, index+iv, YZ)] = 0.25*
	  (+ field[addr_rank0(nsites, index+iv + ys + 1)]
	   - field[addr_rank0(nsites, index+iv + ys - 1)]
	   - field[addr_rank0(nsites, index+iv - ys + 1)]
	   + field[addr_rank0(nsites, index+iv - ys - 1)]);

	dab[addr_rank1(nsites, NSYMM, index+iv, ZZ)] =
	  (+ 1.0*field[addr_rank0(nsites, index+iv + 1)]
	   + 1.0*field[addr_rank0(nsites, index+iv - 1)]
	   - 2.0*field[addr_rank0(nsites, index+iv    )]);
      }
    }
    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  grad_dab_le_correct
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_dab_le_correct(lees_edw_t * le, field_grad_t * df,
					int nextra) {

  int nlocal[3];
  int nhalo;
  int nsites;
  int nh;                                 /* counter over halo extent */
  int nplane;                             /* Number LE planes */
  int ic, jc, kc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */
  double * __restrict__ dab;
  double * __restrict__ field;

  assert(le);
  assert(df);
  assert(nextra >= 0);

  lees_edw_nhalo(le, &nhalo);
  lees_edw_nlocal(le, nlocal);
  lees_edw_nsites(le, &nsites);
  ys = (nlocal[Z] + 2*nhalo);


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

	  dab[addr_rank1(nsites, NSYMM, index, XX)] =
	    (+ 1.0*field[addr_rank0(nsites, indexp1)]
	     + 1.0*field[addr_rank0(nsites, indexm1)]
	     - 2.0*field[addr_rank0(nsites, index)]);
	  dab[addr_rank1(nsites, NSYMM, index, XY)] = 0.25*
	    (+ field[addr_rank0(nsites, indexp1 + ys)]
	     - field[addr_rank0(nsites, indexp1 - ys)]
	     - field[addr_rank0(nsites, indexm1 + ys)]
	     + field[addr_rank0(nsites, indexm1 - ys)]);
	  dab[addr_rank1(nsites, NSYMM, index, XZ)] = 0.25*
	    (+ field[addr_rank0(nsites, indexp1 + 1)]
	     - field[addr_rank0(nsites, indexp1 - 1)]
	     - field[addr_rank0(nsites, indexm1 + 1)]
	     + field[addr_rank0(nsites, indexm1 - 1)]);
	  dab[addr_rank1(nsites, NSYMM, index, YY)] =
	    (+ 1.0*field[addr_rank0(nsites, index + ys)]
	     + 1.0*field[addr_rank0(nsites, index - ys)]
	     - 2.0*field[addr_rank0(nsites, index)]);
	  dab[addr_rank1(nsites, NSYMM, index, YZ)] = 0.25*
	    (+ field[addr_rank0(nsites, index + ys + 1)]
	     - field[addr_rank0(nsites, index + ys - 1)]
	     - field[addr_rank0(nsites, index - ys + 1)]
	     + field[addr_rank0(nsites, index - ys - 1)]);
	  dab[addr_rank1(nsites, NSYMM, index, ZZ)] =
	    (+ 1.0*field[addr_rank0(nsites, index + 1)]
	     + 1.0*field[addr_rank0(nsites, index - 1)]
	     - 2.0*field[addr_rank0(nsites, index)]);
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

	  dab[addr_rank1(nsites, NSYMM, index, XX)] =
	    (+ 1.0*field[addr_rank0(nsites, indexp1)]
	     + 1.0*field[addr_rank0(nsites, indexm1)]
	     - 2.0*field[addr_rank0(nsites, index)]);
	  dab[addr_rank1(nsites, NSYMM, index, XY)] = 0.25*
	    (+ field[addr_rank0(nsites, indexp1 + ys)]
	     - field[addr_rank0(nsites, indexp1 - ys)]
	     - field[addr_rank0(nsites, indexm1 + ys)]
	     + field[addr_rank0(nsites, indexm1 - ys)]);
	  dab[addr_rank1(nsites, NSYMM, index, XZ)] = 0.25*
	    (+ field[addr_rank0(nsites, indexp1 + 1)]
	     - field[addr_rank0(nsites, indexp1 - 1)]
	     - field[addr_rank0(nsites, indexm1 + 1)]
	     + field[addr_rank0(nsites, indexm1 - 1)]);
	  dab[addr_rank1(nsites, NSYMM, index, YY)] =
	    (+ 1.0*field[addr_rank0(nsites, index + ys)]
	     + 1.0*field[addr_rank0(nsites, index - ys)]
	     - 2.0*field[addr_rank0(nsites, index)]);
	  dab[addr_rank1(nsites, NSYMM, index, YZ)] = 0.25*
	    (+ field[addr_rank0(nsites, index + ys + 1)]
	     - field[addr_rank0(nsites, index + ys - 1)]
	     - field[addr_rank0(nsites, index - ys + 1)]
	     + field[addr_rank0(nsites, index - ys - 1)]);
	  dab[addr_rank1(nsites, NSYMM, index, ZZ)] =
	    (+ 1.0*field[addr_rank0(nsites, index + 1)]
	     + 1.0*field[addr_rank0(nsites, index - 1)]
	     - 2.0*field[addr_rank0(nsites, index)]);
	}
      }
    }
    /* Next plane */
  }

  return 0;
}
