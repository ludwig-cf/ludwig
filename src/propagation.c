/*****************************************************************************
 *
 *  propagation.c
 *
 *  Propagation scheme for all models.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "coords_s.h"
#include "kernel.h"
#include "propagation.h"
#include "lb_model_s.h"
#include "timer.h"

__host__ int lb_propagation_driver(lb_t * lb);
__host__ int lb_model_swapf(lb_t * lb);

__global__ void lb_propagation_kernel(kernel_ctxt_t * ktx, lb_t * lb);
__global__ void lb_propagation_kernel_novector(kernel_ctxt_t * ktx, lb_t * lb);

static __constant__ cs_param_t coords;
static __constant__ lb_collide_param_t lbp;

/*****************************************************************************
 *
 *  lb_propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

__host__ int lb_propagation(lb_t * lb) {

  assert(lb);

  lb_propagation_driver(lb);

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagation_driver
 *
 *****************************************************************************/

__host__ int lb_propagation_driver(lb_t * lb) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  /* The kernel is local domain only */

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  tdpMemcpyToSymbol(tdpSymbol(coords), lb->cs->param,
		    sizeof(cs_param_t), 0, tdpMemcpyHostToDevice);
  tdpMemcpyToSymbol(tdpSymbol(lbp), lb->param,
		    sizeof(lb_collide_param_t), 0,
		    tdpMemcpyHostToDevice);

  kernel_ctxt_create(lb->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(TIMER_PROP_KERNEL);

  tdpLaunchKernel(lb_propagation_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, lb->target);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  TIMER_stop(TIMER_PROP_KERNEL);

  kernel_ctxt_free(ctxt);

  lb_model_swapf(lb);

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagation_kernel_novector
 *
 *  Non-vectorised version, e.g., for testing.
 *
 *****************************************************************************/

__global__ void lb_propagation_kernel_novector(kernel_ctxt_t * ktx, lb_t * lb) {

  int kindex;
  int kiter;

  assert(ktx);
  assert(lb);

  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    int n, p;
    int ic, jc, kc;
    int icp, jcp, kcp;
    int index, indexp;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

    for (n = 0; n < lb->ndist; n++) {
      for (p = 0; p < NVEL; p++) {

	/* Pull from neighbour */

	icp = ic - lbp.cv[p][X];
	jcp = jc - lbp.cv[p][Y];
	kcp = kc - lbp.cv[p][Z];
	indexp = kernel_coords_index(ktx, icp, jcp, kcp);

	lb->fprime[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)] 
	  = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, indexp, n, p)];
      }
    }
    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  lb_propagation_kernel
 *
 *  A vectorised version.
 *
 *  Notes.
 *  GPU: Constants must come from static __constant__ memory
 *  GPU: f and fprime must be declared __restrict__
 *
 *****************************************************************************/

__global__ void lb_propagation_kernel(kernel_ctxt_t * ktx, lb_t * lb) {

  int kindex;
  int kiter;
  double * __restrict__ f;
  double * __restrict__ fprime;

  assert(lb);

  kiter = kernel_vector_iterations(ktx);
  f = lb->f;
  fprime = lb->fprime;

  for_simt_parallel(kindex, kiter, NSIMDVL) {

    int iv;
    int n, p;
    int index0;
    int ic[NSIMDVL];
    int jc[NSIMDVL];
    int kc[NSIMDVL];
    int maskv[NSIMDVL];
    int indexp[NSIMDVL];

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    index0 = kernel_baseindex(ktx, kindex);

    for (n = 0; n < lbp.ndist; n++) {
      for (p = 0; p < NVEL; p++) {

	/* If this is a halo site, just copy, else pull from neighbour */ 

	for_simd_v(iv, NSIMDVL) {
	  indexp[iv] = index0 + iv - maskv[iv]*(lbp.cv[p][X]*coords.str[X] +
						lbp.cv[p][Y]*coords.str[Y] +
						lbp.cv[p][Z]*coords.str[Z]);
	}

	for_simd_v(iv, NSIMDVL) {
	  fprime[LB_ADDR(lbp.nsite, lbp.ndist, NVEL, index0 + iv, n, p)] 
	    = f[LB_ADDR(lbp.nsite, lbp.ndist, NVEL, indexp[iv], n, p)];
	}
      }
    }
    /* Next sites */
  }

  return;
}

/*****************************************************************************
 *
 *  lb_model_swapf
 *
 *  Switch the "f" and "fprime" pointers.
 *  Intended for use directly after the propagation step.
 *
 *****************************************************************************/

__host__ int lb_model_swapf(lb_t * lb) {

  int ndevice;
  double * tmp1;
  double * tmp2;

  assert(lb);
  assert(lb->target);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    tmp1 = lb->f;
    lb->f = lb->fprime;
    lb->fprime = tmp1;
  }
  else {
    tdpAssert(tdpMemcpy(&tmp1, &lb->target->f, sizeof(double *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(&tmp2, &lb->target->fprime, sizeof(double *),
			tdpMemcpyDeviceToHost)); 

    tdpAssert(tdpMemcpy(&lb->target->f, &tmp2, sizeof(double *),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&lb->target->fprime, &tmp1, sizeof(double *),
			tdpMemcpyHostToDevice));
  }

  return 0;
}
