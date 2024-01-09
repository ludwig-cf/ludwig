/*****************************************************************************
 *
 *  kernel.c
 *
 *  Help for kernel execution.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

#include "pe.h"
#include "kernel.h"

/* Static kernel context information */

struct kernel_param_s {
  /* physical side */
  int nhalo;
  int nsites;
  int nlocal[3];
  int kindex0;
  /* kernel side no vectorisation */
  int nklocal[3];
  int kernel_iterations;
  /* With vectorisation */
  int nsimdvl;
  int kernel_vector_iterations;
  int nkv_local[3];
  kernel_info_t lim;
};

/* A static device context is provided to prevent repeated device
 * memory allocations/deallocations. */

static __device__   kernel_ctxt_t  static_ctxt;   
static __constant__ kernel_param_t static_param;

static __host__ int kernel_ctxt_commit(kernel_ctxt_t * ctxt, cs_t * cs,
				       int nsimdvl,
				       kernel_info_t lim);

/*****************************************************************************
 *
 *  kernel_ctxt_create
 *
 *****************************************************************************/

__host__ int kernel_ctxt_create(cs_t * cs, int nsimdvl, kernel_info_t info,
				kernel_ctxt_t ** p) {

  int ndevice;
  kernel_ctxt_t * obj = NULL;

  assert(p);
  assert(cs);

  obj = (kernel_ctxt_t *) calloc(1, sizeof(kernel_ctxt_t));
  assert(obj);
  if (obj == NULL) pe_fatal(cs->pe, "calloc(kernel_ctxt_t) failed\n");

  assert(nsimdvl == 1 || nsimdvl == NSIMDVL);

  obj->param = (kernel_param_t *) calloc(1, sizeof(kernel_param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(cs->pe, "calloc(kernel_param_t) failed\n");

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    kernel_param_t * tmp;
    /* Link to static device memory */
    tdpGetSymbolAddress((void **) &obj->target, tdpSymbol(static_ctxt));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(static_param));
    tdpAssert(tdpMemcpy(&obj->target->param, &tmp, sizeof(kernel_param_t *),
			tdpMemcpyHostToDevice));
  }

  kernel_ctxt_commit(obj, cs, nsimdvl, info);

  *p = obj;

  return 0;
}

/*****************************************************************************
 *
 *  kernel_ctxt_free
 *
 *****************************************************************************/

__host__ int kernel_ctxt_free(kernel_ctxt_t * obj) {

  assert(obj);

  free(obj->param);
  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  kernel_ctxt_launch_param
 *
 *****************************************************************************/

__host__
int kernel_ctxt_launch_param(kernel_ctxt_t * obj, dim3 * nblk, dim3 * ntpb) {

  int iterations;

  assert(obj);

  iterations = obj->param->kernel_iterations;
  if (obj->param->nsimdvl == NSIMDVL) {
    iterations = obj->param->kernel_vector_iterations;
  }

  kernel_launch_param(iterations, nblk, ntpb);

  return 0;
}

/****************************************************************************
 *
 *  kernel_launch_param
 *
 *  A "class" method
 *
 ****************************************************************************/

__host__ int kernel_launch_param(int iterations, dim3 * nblk, dim3 * ntpb) {

  assert(iterations > 0);

  ntpb->x = tdp_get_max_threads();
  ntpb->y = 1;
  ntpb->z = 1;

  nblk->x = (iterations + ntpb->x - 1)/ntpb->x;
  nblk->y = 1;
  nblk->z = 1;

  return 0;
}

/*****************************************************************************
 *
 *  kernel_ctxt_commit
 *
 *****************************************************************************/

static __host__ int kernel_ctxt_commit(kernel_ctxt_t * obj, cs_t * cs,
				       int nsimdvl, kernel_info_t lim) {

  int ndevice;
  int kiter;
  int kv_imin;
  int kv_jmin;
  int kv_kmin;

  assert(cs);

  cs_nhalo(cs, &obj->param->nhalo);
  cs_nsites(cs, &obj->param->nsites);
  cs_nlocal(cs, obj->param->nlocal);

  obj->param->nsimdvl = nsimdvl;
  obj->param->lim = lim;

  obj->param->nklocal[X] = lim.imax - lim.imin + 1;
  obj->param->nklocal[Y] = lim.jmax - lim.jmin + 1;
  obj->param->nklocal[Z] = lim.kmax - lim.kmin + 1;

  obj->param->kernel_iterations
    = obj->param->nklocal[X]*obj->param->nklocal[Y]*obj->param->nklocal[Z];

  /* Vectorised case */

  kv_imin = lim.imin;
  kv_jmin = 1 - obj->param->nhalo;
  kv_kmin = 1 - obj->param->nhalo;

  obj->param->nkv_local[X] = obj->param->nklocal[X];
  obj->param->nkv_local[Y] = obj->param->nlocal[Y] + 2*obj->param->nhalo;
  obj->param->nkv_local[Z] = obj->param->nlocal[Z] + 2*obj->param->nhalo;

  /* Offset of first site must be start of a SIMD vector block */

  kiter = cs_index(cs, kv_imin, kv_jmin, kv_kmin);
  obj->param->kindex0 = (kiter/NSIMDVL)*NSIMDVL;

  /* Extent of the contiguous block ... */
  kiter = obj->param->nkv_local[X]*obj->param->nkv_local[Y]*obj->param->nkv_local[Z];
  obj->param->kernel_vector_iterations = kiter;

  /* Copy the results to device memory */

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpMemcpyToSymbol(tdpSymbol(static_param), obj->param,
		      sizeof(kernel_param_t), 0, tdpMemcpyHostToDevice);
  }

  return 0;
}

/*****************************************************************************
 *
 *  kernel_baseindex
 *
 *  Return the coordinate index associated with kernel index kindex.
 *
 *****************************************************************************/

__host__ __device__ int kernel_baseindex(kernel_ctxt_t * obj, int kindex) {

  assert(obj);

  return obj->param->kindex0 + kindex;
}

/*****************************************************************************
 *
 *  kernel_coords_ic
 *
 *****************************************************************************/

__host__ __device__ int kernel_coords_ic(kernel_ctxt_t * obj, int kindex) {

  int ic;

  assert(obj);

  ic = obj->param->lim.imin
    + kindex/(obj->param->nklocal[Y]*obj->param->nklocal[Z]);

  assert(1 - obj->param->nhalo <= ic);
  assert(ic <= obj->param->nlocal[X] + obj->param->nhalo);

  return ic;
}

/*****************************************************************************
 *
 *  kernel_coords_jc
 *
 *****************************************************************************/

__host__ __device__ int kernel_coords_jc(kernel_ctxt_t * obj, int kindex) {

  int ic;
  int jc;
  int xs;

  assert(obj);

  xs = obj->param->nklocal[Y]*obj->param->nklocal[Z];

  ic = kindex/xs;
  jc = obj->param->lim.jmin + (kindex - ic*xs)/obj->param->nklocal[Z];

  assert(1 - obj->param->nhalo <= jc);
  assert(jc <= obj->param->nlocal[Y] + obj->param->nhalo);

  return jc;
}

/*****************************************************************************
 *
 *  kernel_coords_kc
 *
 *****************************************************************************/

__host__ __device__ int kernel_coords_kc(kernel_ctxt_t * obj, int kindex) {

  int ic;
  int jc;
  int kc;
  int xs;

  xs = obj->param->nklocal[Y]*obj->param->nklocal[Z];

  ic = kindex/xs;
  jc = (kindex - ic*xs)/obj->param->nklocal[Z];
  kc = obj->param->lim.kmin + kindex - ic*xs - jc*obj->param->nklocal[Z];

  assert(1 - obj->param->nhalo <= kc);
  assert(kc <= obj->param->nlocal[Z] + obj->param->nhalo);

  return kc;
}

/*****************************************************************************
 *
 *  kernel_coords_v
 *
 *****************************************************************************/

__host__ __device__ int kernel_coords_v(kernel_ctxt_t * obj,
					int kindex0,
					int ic[NSIMDVL],
					int jc[NSIMDVL], int kc[NSIMDVL]) {
  int iv;
  int index;
  int xs;
  int * __restrict__ icv = ic;
  int * __restrict__ jcv = jc;
  int * __restrict__ kcv = kc;

  assert(obj);
  xs = obj->param->nkv_local[Y]*obj->param->nkv_local[Z];

  for_simd_v(iv, NSIMDVL) {
    index = obj->param->kindex0 + kindex0 + iv;

    icv[iv] = index/xs;
    jcv[iv] = (index - icv[iv]*xs)/obj->param->nkv_local[Z];
    kcv[iv] = index - icv[iv]*xs - jcv[iv]*obj->param->nkv_local[Z];
  }

  for_simd_v(iv, NSIMDVL) {
    icv[iv] = icv[iv] - (obj->param->nhalo - 1);
    jcv[iv] = jcv[iv] - (obj->param->nhalo - 1);
    kcv[iv] = kcv[iv] - (obj->param->nhalo - 1);

    assert(1 - obj->param->nhalo <= icv[iv]);
    assert(1 - obj->param->nhalo <= jcv[iv]);
    assert(icv[iv] <= obj->param->nlocal[X] + obj->param->nhalo);
    assert(jcv[iv] <= obj->param->nlocal[Y] + obj->param->nhalo);
    assert(kcv[iv] <= obj->param->nlocal[Z] + obj->param->nhalo);
  }

  return 0;
}

/*****************************************************************************
 *
 *  kernel_mask
 *
 *****************************************************************************/

__host__ __device__
int kernel_mask(kernel_ctxt_t * obj, int ic, int jc, int kc) {

  if (ic < obj->param->lim.imin || ic > obj->param->lim.imax ||
      jc < obj->param->lim.jmin || jc > obj->param->lim.jmax ||
      kc < obj->param->lim.kmin || kc > obj->param->lim.kmax) return 0;

  return 1;
}

/*****************************************************************************
 *
 *  kernel_mask_v
 *
 *****************************************************************************/

__host__ __device__ int kernel_mask_v(kernel_ctxt_t * obj,
				      int ic[NSIMDVL],
				      int jc[NSIMDVL],
				      int kc[NSIMDVL],
				      int mask[NSIMDVL]) {
  int iv;
  int * __restrict__ icv = ic;
  int * __restrict__ jcv = jc;
  int * __restrict__ kcv = kc;
  int * __restrict__ maskv = mask;

  assert(obj);

  for_simd_v(iv, NSIMDVL) {
    maskv[iv] = 1;
  }

  for_simd_v(iv, NSIMDVL) {
    if (icv[iv] < obj->param->lim.imin || icv[iv] > obj->param->lim.imax ||
	jcv[iv] < obj->param->lim.jmin || jcv[iv] > obj->param->lim.jmax ||
	kcv[iv] < obj->param->lim.kmin || kcv[iv] > obj->param->lim.kmax) {
      maskv[iv] = 0;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  kernel_coords_index
 *
 *****************************************************************************/

__host__ __device__
int kernel_coords_index(kernel_ctxt_t * obj, int ic, int jc, int kc) {

  int index;
  int nhalo;
  int xfac, yfac;

  assert(obj);

  nhalo = obj->param->nhalo;
  yfac = obj->param->nlocal[Z] + 2*nhalo;
  xfac = yfac*(obj->param->nlocal[Y] + 2*nhalo);

  index = xfac*(nhalo + ic - 1) + yfac*(nhalo + jc - 1) + nhalo + kc - 1; 

  return index;
}

/*****************************************************************************
 *
 *  kernel_coords_index
 *
 *****************************************************************************/

__host__ __device__ int kernel_coords_index_v(kernel_ctxt_t * obj,
					      int ic[NSIMDVL],
					      int jc[NSIMDVL],
					      int kc[NSIMDVL],
					      int index[NSIMDVL]) {
  int iv;
  int nhalo;
  int xfac, yfac;
  int * __restrict__ icv = ic;
  int * __restrict__ jcv = jc;
  int * __restrict__ kcv = kc;

  assert(obj);

  nhalo = obj->param->nhalo;
  yfac = obj->param->nlocal[Z] + 2*nhalo;
  xfac = yfac*(obj->param->nlocal[Y] + 2*nhalo);

  for_simd_v(iv, NSIMDVL) {
    index[iv] = xfac*(nhalo + icv[iv] - 1)
      + yfac*(nhalo + jcv[iv] - 1) + nhalo + kcv[iv] - 1; 
  }

  return 0;
}

/*****************************************************************************
 *
 *  kernel_iterations
 *
 *****************************************************************************/

__host__ __device__ int kernel_iterations(kernel_ctxt_t * obj) {

  assert(obj);

  return obj->param->kernel_iterations;
}

/*****************************************************************************
 *
 *  kernel_vector_iterations
 *
 *****************************************************************************/

__host__ __device__ int kernel_vector_iterations(kernel_ctxt_t * obj) {

  assert(obj);

  return obj->param->kernel_vector_iterations;
}

/*****************************************************************************
 *
 *  kernel_ctxt_info
 *
 *****************************************************************************/

__host__ int kernel_ctxt_info(kernel_ctxt_t * obj, kernel_info_t * lim) {

  assert(obj);
  assert(lim);

  *lim = obj->param->lim;

  return 0;
}
