/*****************************************************************************
 *
 *  test_prop
 *
 *  Test propagation stage.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 * 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 Ths University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "memory.h"
#include "lb_model_s.h"
#include "propagation.h"
#include "tests.h"

static __host__ int do_test_velocity(lb_halo_enum_t halo);
static __host__ int do_test_source_destination(lb_halo_enum_t halo);
__host__ int lb_propagation_driver(lb_t * lb);
__global__ void lb_propagation_kernel(lb_t * lb);
__global__ void lb_propagation_kernel_novector(lb_t * lb);
__host__ int lb_model_copy(lb_t * lb, int flag);


/*****************************************************************************
 *
 *  test_lb_prop_suite
 *
 *****************************************************************************/

int test_lb_prop_suite(void) {

  pe_init_quiet();
  coords_init();
  do_test_velocity(LB_HALO_FULL);
  do_test_velocity(LB_HALO_REDUCED);

  do_test_source_destination(LB_HALO_FULL);
  do_test_source_destination(LB_HALO_REDUCED);

  info("PASS     ./unit/test_prop\n");
  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_velocity
 *
 *  Check each distribution ends up with the same velocity index.
 *  This relies on the halo exchange working properly.
 *
 *****************************************************************************/

int do_test_velocity(lb_halo_enum_t halo) {

  int nlocal[3];
  int ic, jc, kc, index, p;
  int nd;
  int nvel;
  int ndist = 2;
  double f_actual;

  lb_t * lb = NULL;

  lb_create(&lb);
  assert(lb);

  lb_ndist_set(lb, ndist);
  lb_init(lb);
  lb_halo_set(lb, halo);
  lb_nvel(lb, &nvel);

  coords_nlocal(nlocal);

  /* Set test values */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < nvel; p++) {
	    lb_f_set(lb, index, p, nd, 1.0*(p + nd));
	  }
	}

      }
    }
  }

  lb_halo(lb);
#ifdef OLD_SHIT
  lb_propagation(lb->tcopy);
#else
  lb_model_copy(lb, cudaMemcpyHostToDevice);
  lb_propagation_driver(lb);
  lb_model_copy(lb, cudaMemcpyDeviceToHost);
#endif

  /* Test */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < nvel; p++) {
	    lb_f(lb, index, p, nd, &f_actual);
	    assert(fabs(f_actual - 1.0*(p + nd)) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_source_destination
 *
 *  Check each element of the distribution has propagated exactly one
 *  lattice spacing in the appropriate direction.
 *
 *  We use the global index as the test of the soruce.
 *  
 *****************************************************************************/

int do_test_source_destination(lb_halo_enum_t halo) {

  int nlocal[3], offset[3];
  int ic, jc, kc, index, p;
  int nd;
  int ndist = 2;
  int nvel;
  int isource, jsource, ksource;
  double f_actual, f_expect;

  lb_t * lb = NULL;

  lb_create(&lb);
  assert(lb);
  lb_ndist_set(lb, ndist);
  lb_init(lb);
  lb_halo_set(lb, halo);
  lb_nvel(lb, &nvel);

  coords_nlocal(nlocal);
  coords_nlocal_offset(offset);

  /* Set test values */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	f_actual = L(Y)*L(Z)*(offset[X] + ic) + L(Z)*(offset[Y] + jc) +
	  (offset[Z] + kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < nvel; p++) {
	    lb_f_set(lb, index, p, nd, f_actual);
	  }
	}

      }
    }
  }

  /* HALO SWAP TO BE CHECKED */
  lb_halo(lb);

  lb_model_copy(lb, cudaMemcpyHostToDevice);
  lb_propagation_driver(lb);
  lb_model_copy(lb, cudaMemcpyDeviceToHost);

  /* Test */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < nvel; p++) {
	    isource = offset[X] + ic - cv[p][X];
	    if (isource == 0) isource += N_total(X);
	    if (isource == N_total(X) + 1) isource = 1;
	    jsource = offset[Y] + jc - cv[p][Y];
	    if (jsource == 0) jsource += N_total(Y);
	    if (jsource == N_total(Y) + 1) jsource = 1;
	    ksource = offset[Z] + kc - cv[p][Z];
	    if (ksource == 0) ksource += N_total(Z);
	    if (ksource == N_total(Z) + 1) ksource = 1;

	    f_expect = L(Y)*L(Z)*isource + L(Z)*jsource + ksource;
	    lb_f(lb, index, p, nd, &f_actual);

	    /* In case of d2q9, propagation is only for kc = 1 */
	    if (NDIM == 2 && kc > 1) f_actual = f_expect;

	    assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	  }
	}

	/* Next site */
      }
    }
  }

  lb_free(lb);

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
  __host__ int lb_model_swapf(lb_t * lb);

  assert(lb);

  coords_nlocal(nlocal);

  /* The kernel is local domain only */

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  /* Encapsulate. lb_kernel_commit(lb); */
  copyConstToTarget(tc_cv, cv, NVEL*3*sizeof(int)); 

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  /* NEED TO EDIT CURRENTLY UNTIL ALIASING SORTED IN model.c */
  __host_launch_kernel(lb_propagation_kernel_novector, nblk, ntpb, lb);
  targetDeviceSynchronise();

  kernel_ctxt_free(ctxt);

  lb_model_swapf(lb);

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagation_kernel_novector
 *
 *  Non-vectorised version, just for testing.
 *
 *****************************************************************************/

__global__ void lb_propagation_kernel_novector(lb_t * lb) {

  int kindex;

  assert(lb);

  __target_simt_parallel_for(kindex, kernel_iterations(), 1) {

    int n, p;
    int ic, jc, kc;
    int icp, jcp, kcp;
    int index, indexp;

    ic = kernel_coords_ic(kindex);
    jc = kernel_coords_jc(kindex);
    kc = kernel_coords_kc(kindex);
    index = kernel_coords_index(ic, jc, kc);

    for (n = 0; n < lb->ndist; n++) {
      for (p = 0; p < NVEL; p++) {

	/* Pull from neighbour */ 
	icp = ic - tc_cv[p][X];
	jcp = jc - tc_cv[p][Y];
	kcp = kc - tc_cv[p][Z];
	indexp = kernel_coords_index(icp, jcp, kcp);
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
 *  Ultimately an optimised version.
 *
 *****************************************************************************/

__global__ void lb_propagation_kernel(lb_t * lb) {

  int kindex;

  assert(lb);

  __targetTLP__ (kindex, kernel_vector_iterations()) {

    int iv, indexp;
    int n, p;
    int icp, jcp, kcp;
    int icv[NSIMDVL];
    int jcv[NSIMDVL];
    int kcv[NSIMDVL];
    int maskv[NSIMDVL];
    int index[NSIMDVL];

    __targetILP__(iv) {

      icv[iv] = kernel_coords_icv(kindex, iv);
      jcv[iv] = kernel_coords_jcv(kindex, iv);
      kcv[iv] = kernel_coords_kcv(kindex, iv);

      index[iv] = kernel_coords_index(icv[0], jcv[0], kcv[0]);
      maskv[iv] = kernel_mask(icv[iv], jcv[iv], kcv[iv]);
    }

    for (n = 0; n < lb->ndist; n++) {
      for (p = 0; p < NVEL; p++) {

	__targetILP__(iv) {
	  /* If this is a halo site, just copy, else pull from neighbour */ 
	  icp = icv[iv] - maskv[iv]*tc_cv[p][X];
	  jcp = jcv[iv] - maskv[iv]*tc_cv[p][Y];
	  kcp = kcv[iv] - maskv[iv]*tc_cv[p][Z];
	  indexp = kernel_coords_index(icp, jcp, kcp);
	  lb->fprime[LB_ADDR(lb->nsite, lb->ndist, NVEL, index[iv], n, p)] 
	    = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, indexp, n, p)];
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
  double * tmp;

  assert(lb);
  assert(lb->tcopy);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    tmp = lb->f;
    lb->f = lb->fprime;
    lb->fprime = tmp;
  }
  else {
    double * tmp1;
    double * tmp2;

    copyFromTarget(&tmp1, &lb->tcopy->f, sizeof(double *)); 
    copyFromTarget(&tmp2, &lb->tcopy->fprime, sizeof(double *)); 

    copyToTarget(&lb->tcopy->f, &tmp2, sizeof(double *));
    copyToTarget(&lb->tcopy->fprime, &tmp1, sizeof(double *));
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_model_copy
 *
 *****************************************************************************/

__host__ int lb_model_copy(lb_t * lb, int flag) {

  lb_t * target;
  int ndevice;
  double * tmpf = NULL;

  assert(lb);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Make sure we alias IN THE END */
    /* lb->tcopy = lb; */
  }
  else {

    assert(lb->tcopy);
    target = lb->tcopy;

    copyFromTarget(&tmpf, &target->f, sizeof(double *)); 
    if (flag == cudaMemcpyHostToDevice) {
      copyToTarget(tmpf, lb->f, NVEL*lb->nsite*lb->ndist*sizeof(double));
    }
    else {
      copyFromTarget(lb->f, tmpf, NVEL*lb->nsite*lb->ndist*sizeof(double));
    }
  }

  return 0;
}
