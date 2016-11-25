/*****************************************************************************
 *
 *  propagation.c
 *
 *  Propagation scheme for all models.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "propagation.h"
#include "lb_model_s.h"
#include "timer.h"

__host__ int lb_propagation_driver(lb_t * lb);
__host__ int lb_model_swapf(lb_t * lb);

__global__ void lb_propagation_kernel(kernel_ctxt_t * ktx, lb_t * lb);
__global__ void lb_propagation_kernel_novector(kernel_ctxt_t * ktx, lb_t * lb);

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

  coords_nlocal(nlocal);

  /* The kernel is local domain only */

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  /* SHIT Encapsulate or remove requirement. */
  copyConstToTarget(tc_cv, cv, NVEL*3*sizeof(int)); 

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(TIMER_PROP_KERNEL);

  __host_launch(lb_propagation_kernel, nblk, ntpb, ctxt->target, lb->target);
  targetDeviceSynchronise();

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
  __shared__ int kiter;

  assert(ktx);
  assert(lb);

  kiter = kernel_iterations(ktx);

  __target_simt_parallel_for(kindex, kiter, 1) {

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
	icp = ic - tc_cv[p][X];
	jcp = jc - tc_cv[p][Y];
	kcp = kc - tc_cv[p][Z];
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
 *****************************************************************************/

__global__ void lb_propagation_kernel(kernel_ctxt_t * ktx, lb_t * lb) {

  int kindex;
  __shared__ int kiter;

  assert(lb);

  kiter = kernel_vector_iterations(ktx);

  __targetTLP__ (kindex, kiter) {

    int iv;
    int n, p;
    int index0;
    int ic[NSIMDVL], icp[NSIMDVL];
    int jc[NSIMDVL], jcp[NSIMDVL];
    int kc[NSIMDVL], kcp[NSIMDVL];
    int maskv[NSIMDVL];
    int indexp[NSIMDVL];

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_mask_v(ktx, ic, jc, kc, maskv);

    index0 = kernel_coords_index(ktx, ic[0], jc[0], kc[0]);

    for (n = 0; n < lb->ndist; n++) {
      for (p = 0; p < NVEL; p++) {

	/* If this is a halo site, just copy, else pull from neighbour */ 

	__target_simd_for(iv, NSIMDVL) {
	  icp[iv] = ic[iv] - maskv[iv]*tc_cv[p][X];
	  jcp[iv] = jc[iv] - maskv[iv]*tc_cv[p][Y];
	  kcp[iv] = kc[iv] - maskv[iv]*tc_cv[p][Z];
	}

	kernel_coords_index_v(ktx, icp, jcp, kcp, indexp);

	__target_simd_for(iv, NSIMDVL) {
	  lb->fprime[LB_ADDR(lb->nsite, lb->ndist, NVEL, index0 + iv, n, p)] 
	    = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, indexp[iv], n, p)];
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
 *  Intended for use after the propagation step.
 *
 *****************************************************************************/

__host__ int lb_model_swapf(lb_t * lb) {

  int ndevice;
  double * tmp1;
  double * tmp2;

  assert(lb);
  assert(lb->target);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    tmp1 = lb->f;
    lb->f = lb->fprime;
    lb->fprime = tmp1;
  }
  else {
    copyFromTarget(&tmp1, &lb->target->f, sizeof(double *)); 
    copyFromTarget(&tmp2, &lb->target->fprime, sizeof(double *)); 

    copyToTarget(&lb->target->f, &tmp2, sizeof(double *));
    copyToTarget(&lb->target->fprime, &tmp1, sizeof(double *));
  }

  return 0;
}
