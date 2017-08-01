/*****************************************************************************
 *
 *  phi_force_stress.c
 *  
 *  Storage and computation of the "thermodynamic" stress which
 *  depends on the choice of free energy.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2017 The University of Edinburgh
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
#include "pth_s.h"
#include "phi_force_stress.h"

__global__ void pth_kernel(kernel_ctxt_t * ktx, pth_t * pth, fe_t * fe);
__global__ void pth_kernel_v(kernel_ctxt_t * ktx, pth_t * pth, fe_t * fe);

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
  if (obj == NULL) pe_fatal(pe, "calloc(pth_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->method = method;
  cs_nsites(cs, &obj->nsites);

  /* If memory required */

  if (method == PTH_METHOD_DIVERGENCE) {
    obj->str = (double *) calloc(3*3*obj->nsites, sizeof(double));
    if (obj->str == NULL) pe_fatal(pe, "calloc(pth->str) failed\n");
  }

  /* Allocate target memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {

    tdpMalloc((void **) &obj->target, sizeof(pth_t));
    tdpMemset(obj->target, 0, sizeof(pth_t));
    tdpMemcpy(&obj->target->nsites, &obj->nsites, sizeof(int),
	      tdpMemcpyHostToDevice);

    if (method == PTH_METHOD_DIVERGENCE) {
      tdpMalloc((void **) &tmp, 3*3*obj->nsites*sizeof(double));
      tdpMemcpy(&obj->target->str, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice);
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

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpMemcpy(&tmp, &pth->target->str, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    if (tmp) tdpFree(tmp);
    tdpFree(pth->target);
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

__host__ int pth_memcpy(pth_t * pth, tdpMemcpyKind flag) {

  int ndevice;
  size_t nsz;

  assert(pth);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(pth->target == pth);
  }
  else {
    double * tmp = NULL;

    nsz = 9*pth->nsites*sizeof(double);
    tdpMemcpy(&tmp, &pth->target->str, sizeof(double *),
	      tdpMemcpyDeviceToHost);

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpMemcpy(tmp, pth->str, nsz, flag);
      break;
    case tdpMemcpyDeviceToHost:
      tdpMemcpy(pth->str, tmp, nsz, flag);
      break;
    default:
      pe_fatal(pth->pe, "Bad flag in pth_memcpy\n");
    }
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

  cs_nlocal(pth->cs, nlocal);
  nextra = 1; /* Limits extend one point into the halo */

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(pth->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  fe->func->target(fe, &fe_target);

  tdpLaunchKernel(pth_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, pth->target, fe_target);
  tdpDeviceSynchronize();

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  pth_kernel
 *
 *  No-vectorised version retained for reference.
 *
 *****************************************************************************/

__global__ void pth_kernel(kernel_ctxt_t * ktx, pth_t * pth, fe_t * fe) {

  int kiter;
  int kindex;
  int ic, jc, kc, index;
  double s[3][3];

  assert(ktx);
  assert(pth);
  assert(fe);
  assert(fe->func->stress);

  kiter = kernel_iterations(ktx);

  targetdp_simt_for(kindex, kiter, 1) {

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
 *  pth_kernel_v
 *
 *****************************************************************************/

__global__ void pth_kernel_v(kernel_ctxt_t * ktx, pth_t * pth, fe_t * fe) {

  int kiter;
  int kindex;
  int index;
  int ia, ib, iv;

  double s[3][3][NSIMDVL];

  assert(ktx);
  assert(pth);
  assert(fe);
  assert(fe->func->stress_v);

  kiter = kernel_vector_iterations(ktx);

  targetdp_simt_for(kindex, kiter, NSIMDVL) {

    index = kernel_baseindex(ktx, kindex);

    fe->func->stress_v(fe, index, s);

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
