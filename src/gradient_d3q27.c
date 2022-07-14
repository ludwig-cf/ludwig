/*****************************************************************************
 *
 *  gradient_d3q27.c
 *
 *  Fluid only 27 point stencil based on weighted d3q27.
 *  No Lees-Edwards planes at this time, as this is somewhat
 *  experimental.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "lb_d3q27.h"
#include "gradient_d3q27.h"

__global__ void gradient_d3q27_d2_kernel(kernel_ctxt_t * ktx,
					 field_grad_t * fg);

/*****************************************************************************
 *
 *  gradient_d3q27_d2
 *
 *  Driver to compute grad and delsq.
 *
 *****************************************************************************/

__host__ int gradient_d3q27_d2(field_grad_t * fg) {

  dim3 nblk, ntpb;
  kernel_info_t limits = {0};
  kernel_ctxt_t * ctxt = NULL;

  assert(fg);

  {
    int nextra;
    int nlocal[3];
    cs_t * cs = fg->field->cs;
    
    cs_nhalo(cs, &nextra);
    cs_nlocal(cs, nlocal);
    nextra -= 1;

    limits.imin = 1 - nextra;  limits.imax = nlocal[X] + nextra;
    limits.jmin = 1 - nextra;  limits.jmax = nlocal[Y] + nextra;
    limits.kmin = 1 - nextra;  limits.kmax = nlocal[Z] + nextra;

    kernel_ctxt_create(cs, 1, limits, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb); 
  }

  tdpLaunchKernel(gradient_d3q27_d2_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, fg->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_d3q27_d2_kernel
 *
 *  Stencil operations for grad / delsq
 *
 *****************************************************************************/

__global__ void gradient_d3q27_d2_kernel(kernel_ctxt_t * ktx,
					 field_grad_t * fg) {
  int kindex;
  int kiterations;

  assert(ktx);
  assert(fg);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    LB_CV_D3Q27(cv);
    LB_WEIGHTS_D3Q27(wv);

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);

    field_t * f = fg->field;

    /* For each component, gradient and delsq */
    for (int n = 0; n < f->nf; n++) {

      double phi0 = f->data[addr_rank1(f->nsites, f->nf, index, n)];

      double gradx = 0.0;
      double grady = 0.0;
      double gradz = 0.0;
      double delsq = 0.0;

      for (int p = 1; p < NVEL_D3Q27; p++) {
	int ic1 = ic + cv[p][X];
	int jc1 = jc + cv[p][Y];
	int kc1 = kc + cv[p][Z];
	int index1 = kernel_coords_index(ktx, ic1, jc1, kc1);
	double dphi = f->data[addr_rank1(f->nsites, f->nf, index1, n)] - phi0;

	/* Additional factor of 3.0 in gradient, and 6.0 in delsq */
	gradx += 3.0*wv[p]*dphi*cv[p][X];
	grady += 3.0*wv[p]*dphi*cv[p][Y];
	gradz += 3.0*wv[p]*dphi*cv[p][Z];
	delsq += 6.0*wv[p]*dphi;
      }

      fg->grad[addr_rank2(f->nsites, f->nf, 3, index, n, X)]  = gradx;
      fg->grad[addr_rank2(f->nsites, f->nf, 3, index, n, Y)]  = grady;
      fg->grad[addr_rank2(f->nsites, f->nf, 3, index, n, Z)]  = gradz;
      fg->delsq[addr_rank1(f->nsites, f->nf, index, n)]       = delsq; 
    }
  }

  return;
}
