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
 *  Note: weights consistent with the LB D2Q9 weights would have
 *        relative weights of a and b 4:1, ie, epsilon = 0.25...
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
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2024 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "kernel.h"
#include "wall.h"
#include "gradient_2d_tomita_fluid.h"

static const double epsilon_ = 0.5;
static const double epsilon1_ = 0.25;

/* Use weights consistent with the D2Q9 LB model */

#define GRAD_EPSILON 0.25
#define DEL2_EPSILON 0.25

#define E_PARAM(epsilon)          (1.0/(1.0 + 2.0*(epsilon)))
#define GRAD_WEIGHT_W1(epsilon)   (0.5*E_PARAM(epsilon))
#define GRAD_WEIGHT_W2(epsilon)   (0.5*E_PARAM(epsilon)*epsilon)
#define DEL2_WEIGHT_W0(epsilon)   (E_PARAM(epsilon)*4.0*(1.0 + epsilon))
#define DEL2_WEIGHT_W1(epsilon)   (E_PARAM(epsilon))
#define DEL2_WEIGHT_W2(epsilon)   (E_PARAM(epsilon)*epsilon)

__host__ int grad_cs_compute(field_grad_t * fgrad, int nextra);

__host__ int grad_2d_tomita_fluid_operator(lees_edw_t * le, field_grad_t * fg,
					   int nextra);
__host__ int grad_2d_tomita_fluid_le(lees_edw_t * le, field_grad_t * fg,
				     int nextra);

__global__ void grad_cs_kernel(kernel_3d_t k3d, field_grad_t * fgrad,
			       int xs, int ys);

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
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  assert(fg);
  assert(fg->field);

  cs = fg->field->cs;
  le = fg->field->le;

  cs_nhalo(cs, &nextra);
  nextra -= 1;
  assert(nextra >= 0);

  if (le) {
    grad_2d_tomita_fluid_operator(le, fg, nextra);
    grad_2d_tomita_fluid_le(le, fg, nextra);
  }
  else {
    assert(fg->field->cs);
    grad_cs_compute(fg, nextra);
  }

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

  int nhalo;
  int nextra;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  cs = fg->field->cs;

  cs_nhalo(cs, &nhalo);
  nextra = nhalo - 2;
  assert(nextra >= 0);

  assert(0); /* NO TEST */

  grad_2d_tomita_fluid_operator(le, fg, nextra);
  grad_2d_tomita_fluid_le(le, fg, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  grad_cs_compute
 *
 *  Compute grad and delsq (no Lees Edwards SPBC).
 *
 *****************************************************************************/

__host__ int grad_cs_compute(field_grad_t * fgrad, int nextra) {

  int nlocal[3];
  int xs, ys, zs;
  cs_t * cs;

  assert(fgrad);
  assert(fgrad->field);
  assert(fgrad->field->cs);

  cs = fgrad->field->cs;
  cs_nlocal(cs, nlocal);
  cs_strides(cs, &xs, &ys, &zs);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {
      .imin = 1 - nextra, .imax = nlocal[X] + nextra,
      .jmin = 1 - nextra, .jmax = nlocal[Y] + nextra,
      .kmin = 1,          .kmax = 1
    };
    kernel_3d_t k3d = kernel_3d(cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(grad_cs_kernel, nblk, ntpb, 0, 0,
		    k3d, fgrad->target, xs, ys);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_cs_kernel
 *
 *****************************************************************************/

__global__ void grad_cs_kernel(kernel_3d_t k3d, field_grad_t * fgrad,
			       int xs, int ys) {

  int kindex = 0;

  const double r1 = GRAD_WEIGHT_W1(GRAD_EPSILON);
  const double r2 = GRAD_WEIGHT_W2(GRAD_EPSILON);
  const double w0 = DEL2_WEIGHT_W0(DEL2_EPSILON);
  const double w1 = DEL2_WEIGHT_W1(DEL2_EPSILON);
  const double w2 = DEL2_WEIGHT_W2(DEL2_EPSILON);

  assert(fgrad);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    field_t * __restrict__ f = fgrad->field;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = 1;
    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);


    /* for each order parameter */
    for (int n1 = 0; n1 < f->nf; n1++) {

      fgrad->grad[addr_rank2(f->nsites, f->nf, NVECTOR, index, n1, X)] =
	+ r2*f->data[addr_rank1(f->nsites, f->nf, index + xs - ys, n1)]
	- r2*f->data[addr_rank1(f->nsites, f->nf, index - xs - ys, n1)]
	+ r1*f->data[addr_rank1(f->nsites, f->nf, index + xs     , n1)]
	- r1*f->data[addr_rank1(f->nsites, f->nf, index - xs     , n1)]
	+ r2*f->data[addr_rank1(f->nsites, f->nf, index + xs + ys, n1)]
	- r2*f->data[addr_rank1(f->nsites, f->nf, index - xs + ys, n1)];

      fgrad->grad[addr_rank2(f->nsites, f->nf, NVECTOR, index, n1, Y)] =
	+ r2*f->data[addr_rank1(f->nsites, f->nf, index - xs + ys, n1)]
	- r2*f->data[addr_rank1(f->nsites, f->nf, index - xs - ys, n1)]
	+ r1*f->data[addr_rank1(f->nsites, f->nf, index      + ys, n1)]
	- r1*f->data[addr_rank1(f->nsites, f->nf, index      - ys, n1)]
	+ r2*f->data[addr_rank1(f->nsites, f->nf, index + xs + ys, n1)]
	- r2*f->data[addr_rank1(f->nsites, f->nf, index + xs - ys, n1)];

      fgrad->grad[addr_rank2(f->nsites, f->nf, NVECTOR, index, n1, Z)] = 0.0;

      /* delsq */
      fgrad->delsq[addr_rank1(f->nsites, f->nf, index, n1)] =
	+ w1*f->data[addr_rank1(f->nsites, f->nf, index + xs,      n1)]
	+ w1*f->data[addr_rank1(f->nsites, f->nf, index - xs,      n1)]
	+ w1*f->data[addr_rank1(f->nsites, f->nf, index      + ys, n1)]
	+ w1*f->data[addr_rank1(f->nsites, f->nf, index      - ys, n1)]
	+ w2*f->data[addr_rank1(f->nsites, f->nf, index + xs + ys, n1)]
	+ w2*f->data[addr_rank1(f->nsites, f->nf, index + xs - ys, n1)]
	+ w2*f->data[addr_rank1(f->nsites, f->nf, index - xs + ys, n1)]
	+ w2*f->data[addr_rank1(f->nsites, f->nf, index - xs - ys, n1)]
	- w0*f->data[addr_rank1(f->nsites, f->nf, index,           n1)];
    }
  }

  return;
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
