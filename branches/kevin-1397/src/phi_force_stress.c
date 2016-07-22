/*****************************************************************************
 *
 *  phi_force_stress.c
 *  
 *  Wrapper functions for stress computation.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "timer.h"
#include "kernel.h"
#include "pth_s.h"
#include "phi_force_stress.h"

__global__ void pth_kernel_novector(kernel_ctxt_t * ktx, pth_t * pth,
				    fe_t * fe);
__global__ void pth_kernel(kernel_ctxt_t * ktx, pth_t * pth, fe_t * fe);

/*****************************************************************************
 *
 *  pth_create
 *
 *  The stress is always 3x3 tensor (to allow an anti-symmetric
 *  contribution), if it is required.
 *
 *****************************************************************************/

__host__ int pth_create(int method, pth_t ** pobj) {

  int ndevice;
  double * tmp;
  pth_t * obj = NULL;

  assert(pobj);

  obj = (pth_t *) calloc(1, sizeof(pth_t));
  if (obj == NULL) fatal("calloc(pth_t) failed\n");

  obj->method = method;
  obj->nsites = coords_nsites();

  /* If memory required */

  if (method == PTH_METHOD_DIVERGENCE) {
    obj->str = (double *) calloc(3*3*obj->nsites, sizeof(double));
    if (obj->str == NULL) fatal("calloc(pth->str) failed\n");
  }

  /* Allocate target memory, or alias */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {

    targetCalloc((void **) &obj->target, sizeof(pth_t));
    copyToTarget(&obj->target->nsites, &obj->nsites, sizeof(int));

    if (method == PTH_METHOD_DIVERGENCE) {
      targetCalloc((void **) &tmp, 3*3*obj->nsites*sizeof(double));
      copyToTarget(&obj->target->str, &tmp, sizeof(double *));
    }
  }

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  pth_free
 *
 *****************************************************************************/

__host__ int pth_free(pth_t * pth) {

  int ndevice;
  double * tmp = NULL;

  assert(pth);

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    copyFromTarget(&tmp, &pth->target->str, sizeof(double *));
    if (tmp) targetFree(tmp);
    targetFree(pth->target);
  }

  if (pth->str) free(pth->str);
  free(pth);

  return 0;
}

/*****************************************************************************
 *
 *  pth_memcpy
 *
 *****************************************************************************/

__host__ int pth_memcpy(pth_t * pth, int flag) {

  int ndevice;

  assert(pth);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(pth->target == pth);
  }
  else {
    assert(0); /* Copy */
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_stress_compute
 *
 *  Compute the stress everywhere and store.
 *
 *****************************************************************************/

__host__ int pth_stress_compute(pth_t * pth, fe_t * fe) {

  int nextra;
  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  fe_t * fe_target = NULL;

  assert(pth);
  assert(fe);
  assert(fe->func->target);

  coords_nlocal(nlocal);
  nextra = 1; /* Limits extend one point into the halo */

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  fe->func->target(fe, &fe_target);

  __host_launch(pth_kernel, nblk, ntpb, ctxt->target,
		pth->target, fe_target);

  targetDeviceSynchronise();

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  pth_kernel_novector
 *
 *****************************************************************************/

__global__ void pth_kernel_novector(kernel_ctxt_t * ktx, pth_t * pth,
				    fe_t * fe) {

  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(pth);
  assert(fe);
  assert(fe->func->stress);

  kiter = kernel_iterations(ktx);

  __target_simt_parallel_for(kindex, kiter, 1) {

    int ic, jc, kc, index;
    double s[3][3];

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

    fe->func->stress(fe, index, s);
    pth_stress_set(pth, index, s);
  }

  return;
}

/*****************************************************************************
 *
 *  pth_kernel
 *
 *****************************************************************************/

#include "blue_phase.h"

__global__
void pth_kernel(kernel_ctxt_t * ktx, pth_t * pth, fe_t * fe) {

  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(pth);
  assert(fe);
  /*assert(fe->func->stress_v);*/

  kiter = kernel_vector_iterations(ktx);

  __target_simt_parallel_for(kindex, kiter, NSIMDVL) {

    int index;
    int ia, ib, iv;

    double s[3][3][NSIMDVL];
    fe_lc_t * lc = (fe_lc_t *) fe;

    index = kernel_baseindex(ktx, kindex);

    fe_lc_stress_v(lc, index, s);
    /*fe->func->stress(fe, index, s);*/

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	for (iv = 0; iv < NSIMDVL; iv++) {
	  pth->str[addr_rank2(pth->nsites,3,3,index+iv,ia,ib)] = s[ia][ib][iv];
	}
      }
    }

    /* Next block */
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_set
 *
 *****************************************************************************/

__host__  __device__
void pth_stress_set(pth_t * pth, int index, double p[3][3]) {

  int ia, ib;

  assert(pth);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)] = p[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress
 *
 *****************************************************************************/

__host__  __device__
void pth_stress(pth_t * pth, int index, double p[3][3]) {

  int ia, ib;

  assert(pth);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)];
    }
  }

  return;
}
