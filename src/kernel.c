/*****************************************************************************
 *
 *  kernel.c
 *
 *  Help for kernel execution.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "memory.h"

/* Static kernel context information */

struct kernel_ctxt_s {
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

static kernel_ctxt_t hlimits;                 /* Host copy */
static __constant__ kernel_ctxt_t klimits;    /* Device copy */

static __host__ int kernel_ctxt_commit(int nsimdvl, kernel_info_t lim);

/*****************************************************************************
 *
 *  kernel_ctxt_create
 *
 *****************************************************************************/

__host__ int kernel_ctxt_create(int nsimdvl, kernel_info_t info,
				kernel_ctxt_t ** p) {

  kernel_ctxt_t * obj = NULL;

  assert(p);

  obj = (kernel_ctxt_t *) calloc(1, sizeof(kernel_ctxt_t));
  if (obj == NULL) fatal("calloc(kernel_ctxt_t) failed\n");

  assert(nsimdvl == 1 || nsimdvl == NSIMDVL);

  kernel_ctxt_commit(nsimdvl, info);
  *p = obj;

  return 0;
}

/*****************************************************************************
 *
 *  kernel_ctxt_free
 *
 *****************************************************************************/

__host__ int kernel_ctxt_free(kernel_ctxt_t * obj) {

  kernel_ctxt_t zero = {0};

  assert(obj);

  hlimits = zero;
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

  iterations = hlimits.kernel_iterations;
  if (hlimits.nsimdvl == NSIMDVL) iterations = hlimits.kernel_vector_iterations;

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

  ntpb->x = __host_threads_per_block();
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

static __host__ int kernel_ctxt_commit(int nsimdvl, kernel_info_t lim) {

  int kiter;
  int kv_imin;
  int kv_jmin;
  int kv_kmin;

  hlimits.nhalo = coords_nhalo();
  hlimits.nsites = coords_nsites();
  coords_nlocal(hlimits.nlocal);

  hlimits.nsimdvl = nsimdvl;
  hlimits.lim = lim;

  hlimits.nklocal[X] = lim.imax - lim.imin + 1;
  hlimits.nklocal[Y] = lim.jmax - lim.jmin + 1;
  hlimits.nklocal[Z] = lim.kmax - lim.kmin + 1;

  hlimits.kernel_iterations
    = hlimits.nklocal[X]*hlimits.nklocal[Y]*hlimits.nklocal[Z];

  /* Vectorised case */

  kv_imin = lim.imin;
  kv_jmin = 1 - hlimits.nhalo;
  kv_kmin = 1 - hlimits.nhalo;

  hlimits.nkv_local[X] = hlimits.nklocal[X];
  hlimits.nkv_local[Y] = hlimits.nlocal[Y] + 2*hlimits.nhalo;
  hlimits.nkv_local[Z] = hlimits.nlocal[Z] + 2*hlimits.nhalo;

  /* Offset of first site must be start of a SIMD vector block */
  kiter = coords_index(kv_imin, kv_jmin, kv_kmin);
  hlimits.kindex0 = (kiter/NSIMDVL)*NSIMDVL;

  /* Extent of the contiguous block ... */
  kiter = hlimits.nkv_local[X]*hlimits.nkv_local[Y]*hlimits.nkv_local[Z];
  hlimits.kernel_vector_iterations = kiter;

  /* Copy the results to device memory */

  copyConstToTarget(&klimits, &hlimits, sizeof(kernel_ctxt_t));

  return 0;
}

/*****************************************************************************
 *
 *  kernel_host_target
 *
 *  A convenience to prevent ifdefs all over the place.
 *  Note this must be a real function, not a macro, as macro expansion
 *  occurs after preprocessor directive execution.
 *
 *****************************************************************************/

__host__ __target__ __inline__
static kernel_ctxt_t kernel_host_target(void) {

#ifdef __CUDA_ARCH__
  return klimits;
#else
  return hlimits;
#endif
}

/*****************************************************************************
 *
 *  kernel_coords_ic
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_ic(int kindex) {

  int ic;
  const kernel_ctxt_t limits = kernel_host_target();

  ic = limits.lim.imin + kindex/(limits.nklocal[Y]*limits.nklocal[Z]);
  assert(1 - limits.nhalo <= ic);
  assert(ic <= limits.nlocal[X] + limits.nhalo);

  return ic;
}

/*****************************************************************************
 *
 *  kernel_coords_jc
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_jc(int kindex) {

  int ic;
  int jc;
  const kernel_ctxt_t limits = kernel_host_target();

  ic = kindex/(limits.nklocal[Y]*limits.nklocal[Z]);
  jc = limits.lim.jmin +
    (kindex - ic*limits.nklocal[Y]*limits.nklocal[Z])/limits.nklocal[Z];
  assert(1 - limits.nhalo <= jc);
  assert(jc <= limits.nlocal[Y] + limits.nhalo);

  return jc;
}

/*****************************************************************************
 *
 *  kernel_coords_kc
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_kc(int kindex) {

  int ic;
  int jc;
  int kc;
  const kernel_ctxt_t limits = kernel_host_target();

  ic = kindex/(limits.nklocal[Y]*limits.nklocal[Z]);
  jc = (kindex - ic*limits.nklocal[Y]*limits.nklocal[Z])/limits.nklocal[Z];
  kc = limits.lim.kmin +
    kindex - ic*limits.nklocal[Y]*limits.nklocal[Z] - jc*limits.nklocal[Z];
  assert(1 - limits.nhalo <= kc);
  assert(kc <= limits.nlocal[Z] + limits.nhalo);

  return kc;
}

/*****************************************************************************
 *
 *  kernel_coords_icv
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_icv(int kindex, int iv) {

  int ic;
  int index;
  const kernel_ctxt_t limits = kernel_host_target();

  index = limits.kindex0 + kindex + iv;

  ic = index/(limits.nkv_local[Y]*limits.nkv_local[Z]);
  assert(1 - limits.nhalo <= ic);
  assert(ic <= limits.nlocal[X] + limits.nhalo);

  return ic;
}

/*****************************************************************************
 *
 *  kernel_coords_jcv
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_jcv(int kindex, int iv) {

  int jc;
  int ic;
  int index;
  const kernel_ctxt_t limits = kernel_host_target();

  index = limits.kindex0 + kindex + iv;

  ic = index/(limits.nkv_local[Y]*limits.nkv_local[Z]);
  jc = (index - ic*limits.nkv_local[Y]*limits.nkv_local[Z])/limits.nkv_local[Z];
  assert(1 - limits.nhalo <= jc);
  assert(jc <= limits.nlocal[Y] + limits.nhalo);

  return jc;
}

/*****************************************************************************
 *
 *  kernel_coords_kcv
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_kcv(int kindex, int iv) {

  int kc;
  int jc;
  int ic;
  int index;
  const kernel_ctxt_t limits = kernel_host_target();

  index = limits.kindex0 + kindex + iv;

  ic = index/(limits.nkv_local[Y]*limits.nkv_local[Z]);
  jc = (index - ic*limits.nkv_local[Y]*limits.nkv_local[Z])/limits.nkv_local[Z];
  kc = index - ic*limits.nkv_local[Y]*limits.nkv_local[Z] - jc*limits.nkv_local[Z];

  assert(1 - limits.nhalo <= jc);
  assert(jc <= limits.nlocal[Y] + limits.nhalo);
  assert(kc <= limits.nlocal[Z] + limits.nhalo);

  return kc;
}

/*****************************************************************************
 *
 *  kernel_mask
 *
 *****************************************************************************/

__host__ __target__ int kernel_mask(int ic, int jc, int kc) {

  const kernel_ctxt_t kern = kernel_host_target();

  if (ic < kern.lim.imin || ic > kern.lim.imax ||
      jc < kern.lim.jmin || jc > kern.lim.jmax ||
      kc < kern.lim.kmin || kc > kern.lim.kmax) return 0;

  return 1;
}

/*****************************************************************************
 *
 *  kernel_coords_index
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_index(int ic, int jc, int kc) {

  int index;
  int nhalo;
  int xfac, yfac;
  const kernel_ctxt_t limits = kernel_host_target();

  nhalo = limits.nhalo;
  yfac = limits.nlocal[Z] + 2*nhalo;
  xfac = yfac*(limits.nlocal[Y] + 2*nhalo);

  index = xfac*(nhalo + ic - 1) + yfac*(nhalo + jc - 1) + nhalo + kc - 1; 

  return index;
}

/*****************************************************************************
 *
 *  kernel_iterations
 *
 *****************************************************************************/

__host__ __target__ int kernel_iterations(void) {

  const kernel_ctxt_t limits = kernel_host_target();

  return limits.kernel_iterations;
}

/*****************************************************************************
 *
 *  kernel_vector_iterations
 *
 *****************************************************************************/

__host__ __target__ int kernel_vector_iterations(void) {

  const kernel_ctxt_t limits = kernel_host_target();

  return limits.kernel_vector_iterations;
}

/*****************************************************************************
 *
 *  kernel_ctxt_info
 *
 *****************************************************************************/

__host__ int kernel_ctxt_info(kernel_info_t * lim) {

  assert(lim);

  *lim = hlimits.lim;

  return 0;
}
