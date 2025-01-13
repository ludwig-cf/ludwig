/*****************************************************************************
 *
 *  phi_force_stress.c
 *
 *  Storage and computation of the "thermodynamic" stress which
 *  depends on the choice of free energy.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "timer.h"
#include "kernel.h"
#include "phi_force_stress.h"

__global__ void pth_kernel(kernel_3d_t k3d, pth_t * pth, fe_t * fe);
__global__ void pth_kernel_v(kernel_3d_v_t k3v, pth_t * pth, fe_t * fe);
__global__ void pth_kernel_a_v(kernel_3d_v_t k3v, pth_t * pth, fe_t * fe);

/*****************************************************************************
 *
 *  pth_create
 *
 *  The stress is always 3x3 tensor (to allow an anti-symmetric
 *  contribution), if it is required.
 *
 *****************************************************************************/

__host__ int pth_create(pe_t * pe, cs_t * cs, int method, pth_t ** pobj) {

  int ndevice;
  double * tmp;
  pth_t * obj = NULL;

  assert(pobj);

  obj = (pth_t *) calloc(1, sizeof(pth_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(pth_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->method = method;
  cs_nsites(cs, &obj->nsites);

  /* malloc() here in all cases on host (even if not required). */

  obj->str = (double *) malloc(3*3*obj->nsites*sizeof(double));
  assert(obj->str);
  if (obj->str == NULL) pe_fatal(pe, "malloc(pth->str) failed\n");

  /* Allocate target memory, or alias */

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    /* Allocate data only if really required */
    int imem = (method == FE_FORCE_METHOD_STRESS_DIVERGENCE)
            || (method == FE_FORCE_METHOD_RELAXATION_ANTI);

    tdpAssert( tdpMalloc((void **) &obj->target, sizeof(pth_t)) );
    tdpAssert( tdpMemset(obj->target, 0, sizeof(pth_t)) );
    tdpAssert( tdpMemcpy(&obj->target->nsites, &obj->nsites, sizeof(int),
			 tdpMemcpyHostToDevice) );

    if (imem) {
      tdpAssert( tdpMalloc((void **) &tmp, 3*3*obj->nsites*sizeof(double)) );
      tdpAssert( tdpMemcpy(&obj->target->str, &tmp, sizeof(double *),
			   tdpMemcpyHostToDevice) );
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

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice > 0) {
    tdpAssert( tdpMemcpy(&tmp, &pth->target->str, sizeof(double *),
			 tdpMemcpyDeviceToHost) );
    if (tmp) tdpAssert( tdpFree(tmp) );
    tdpAssert( tdpFree(pth->target) );
  }

  free(pth->str);
  free(pth);

  return 0;
}

/*****************************************************************************
 *
 *  pth_memcpy
 *
 *****************************************************************************/

__host__ int pth_memcpy(pth_t * pth, tdpMemcpyKind flag) {

  int ndevice;
  size_t nsz;

  assert(pth);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(pth->target == pth);
  }
  else {
    double * tmp = NULL;

    nsz = 9*pth->nsites*sizeof(double);
    tdpAssert( tdpMemcpy(&tmp, &pth->target->str, sizeof(double *),
			 tdpMemcpyDeviceToHost) );

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpAssert( tdpMemcpy(tmp, pth->str, nsz, flag) );
      break;
    case tdpMemcpyDeviceToHost:
      tdpAssert( tdpMemcpy(pth->str, tmp, nsz, flag) );
      break;
    default:
      pe_fatal(pth->pe, "Bad flag in pth_memcpy\n");
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  pth_stress_compute
 *
 *  Compute the stress everywhere and store. This allows that the
 *  full stress, or just the antisymmetric part is needed.
 *
 *****************************************************************************/

__host__ int pth_stress_compute(pth_t * pth, fe_t * fe) {

  int nextra;
  int nlocal[3];
  fe_t * fe_target = NULL;

  assert(pth);
  assert(fe);
  assert(fe->func->target);

  cs_nlocal(pth->cs, nlocal);
  nextra = 1; /* Limits extend one point into the halo */

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {
      .imin = 1 - nextra, .imax = nlocal[X] + nextra,
      .jmin = 1 - nextra, .jmax = nlocal[Y] + nextra,
      .kmin = 1 - nextra, .kmax = nlocal[Z] + nextra
    };
    kernel_3d_v_t k3v = kernel_3d_v(pth->cs, lim, NSIMDVL);

    kernel_3d_launch_param(k3v.kiterations, &nblk, &ntpb);

    fe->func->target(fe, &fe_target);

    if (fe->use_stress_relaxation) {
      /* Antisymmetric part only required; if no antisymmetric part,
       * do nothing. */
      if (fe->func->str_anti != NULL) {
	tdpLaunchKernel(pth_kernel_a_v, nblk, ntpb, 0, 0,
			k3v, pth->target, fe_target);
      }
    }
    else {
      /* Full stress */
      tdpLaunchKernel(pth_kernel_v, nblk, ntpb, 0, 0,
		      k3v, pth->target, fe_target);
    }

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}

/*****************************************************************************
 *
 *  pth_kernel
 *
 *  No-vectorised version retained for reference.
 *
 *****************************************************************************/

__global__ void pth_kernel(kernel_3d_t k3d, pth_t * pth, fe_t * fe) {

  int kindex = 0;

  assert(pth);
  assert(fe);
  assert(fe->func->stress);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    double s[3][3] = {0};

    fe->func->stress(fe, index, s);
    pth_stress_set(pth, index, s);
  }

  return;
}

/*****************************************************************************
 *
 *  pth_kernel_v
 *
 *****************************************************************************/

__global__ void pth_kernel_v(kernel_3d_v_t k3v, pth_t * pth, fe_t * fe) {

  int kindex = 0;

  assert(pth);
  assert(fe);
  assert(fe->func->stress_v);

  for_simt_parallel(kindex, k3v.kiterations, NSIMDVL) {

    int index = k3v.kindex0 + kindex;
    double s[3][3][NSIMDVL] = {0};

    fe->func->stress_v(fe, index, s);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	int iv = 0;
	for_simd_v(iv, NSIMDVL) {
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
 *  pth_kernel_a_v
 *
 *****************************************************************************/

__global__ void pth_kernel_a_v(kernel_3d_v_t k3v, pth_t * pth, fe_t * fe) {

  int kindex = 0;

  assert(pth);
  assert(fe);
  assert(fe->func->str_anti_v);

  for_simt_parallel(kindex, k3v.kiterations, NSIMDVL) {

    int index = k3v.kindex0 + kindex;
    double s[3][3][NSIMDVL];

    fe->func->str_anti_v(fe, index, s);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	int iv = 0;
	for_simd_v(iv, NSIMDVL) {
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
