/*****************************************************************************
 *
 *  test_kernel.c
 *
 *  Test kernel coverage depending on target, vector length, and so on.
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
#include "memory.h"

#define __host__ __targetHost__
#define __kernel__ __targetEntry__
#define targetLaunch(kernel_function, ...) kernel_function(__VA_ARGS__)

typedef struct kernel_limit_s kernel_limit_t;
struct kernel_limit_s {
  /* physical side */
  int nhalo;
  int nsites;
  int nlocal[3];
  int imin;
  int imax;
  int jmin;
  int jmax;
  int kmin;
  int kmax;
  /* kernel side no vectorisation */
  int nklocal[3];
  int kindex0;
  int kernel_iterations;
  /* With vectorisation */
  int kv_imin;
  int kv_jmin;
  int kv_kmin;
  int kernel_vector_iterations;
  int nkv_local[3];
};

static __targetConst__ kernel_limit_t klimits;

__host__ int do_test_kernel(kernel_limit_t * limits);
__host__ int do_host_kernel(kernel_limit_t * limits, int * mask);
__host__ int do_check(int * iref, int * itarget);
__kernel__ void do_target_kernel1(kernel_limit_t * limits, int * mask);
__kernel__ void do_target_kernel2(kernel_limit_t * limits, int * mask);

__host__            int kernel_coords_commit(kernel_limit_t * limits);
__host__ __target__ int kernel_coords_ic(int kindex);
__host__ __target__ int kernel_coords_jc(int kindex);
__host__ __target__ int kernel_coords_kc(int kindex);
__host__ __target__ int kernel_coords_icv(int kindex, int iv);
__host__ __target__ int kernel_coords_jcv(int kindex, int iv);
__host__ __target__ int kernel_coords_kcv(int kindex, int iv);
__host__ __target__ int kernel_mask(int ic, int jc, int kc);
__host__ __target__ int kernel_coords_index(int ic, int jc, int kc);

/*****************************************************************************
 *
 *  test_kernel_suite
 *
 *****************************************************************************/

int test_kernel_suite(void) {

  int nlocal[3];
  kernel_limit_t lim;
  kernel_limit_t * limits = &lim;

  pe_init_quiet();
  coords_init();
  coords_nlocal(nlocal);

  lim.imin = 1; lim.imax = nlocal[X];
  lim.jmin = 1; lim.jmax = nlocal[Y];
  lim.kmin = 1; lim.kmax = nlocal[Z];
  do_test_kernel(limits);

  lim.imin = 0; lim.imax = nlocal[X] + 1;
  lim.jmin = 0; lim.jmax = nlocal[Y] + 1;
  lim.kmin = 0; lim.kmax = nlocal[Z] + 1;
  do_test_kernel(limits);

  printf("Target vector length was %d\n", NSIMDVL);

  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_kernel
 *
 *  For given configuration run the following
 *   1. __host__   "kernel" with standard form to generate reference result
 *   2. __kernel__ with no explicit vectorisation
 *   3. __kernel__ with explicit vectorisation
 *
 *****************************************************************************/

__host__ int do_test_kernel(kernel_limit_t * limits) {

  int nsites;
  int * iref = NULL;
  int * itarget1 = NULL;
  int * itarget2 = NULL;

  /* Allocate space for reference */

  nsites = coords_nsites();

  /* Additional issue. The assertions in memory.c actaully catch
   * legal address owing to the division by NSIMDVL in the macros.
   * It might be worth moving the division to after the assertion
   * in memory.c */

  /* In the meantime, we need... */
  assert(nsites % NSIMDVL == 0);

  iref = (int *) calloc(nsites, sizeof(int));
  itarget1 = (int *) calloc(nsites, sizeof(int));
  itarget2 = (int *) calloc(nsites, sizeof(int));
  assert(iref);
  assert(itarget1);
  assert(itarget2);

  kernel_coords_commit(limits);

  do_host_kernel(limits, iref);
  /*
  targetLaunch(do_target_kernel1, limits, itarget1);
  do_check(iref, itarget1);
  */
  targetLaunch(do_target_kernel2, limits, itarget2);
  do_check(iref, itarget2);

  free(itarget2);
  free(itarget1);
  free(iref);

  return 0;
}

/*****************************************************************************
 *
 *  do_host_kernel
 *
 *  Set the relevant sites as "visited".
 *
 *****************************************************************************/

__host__ int do_host_kernel(kernel_limit_t * limits, int * mask) {

  int index;
  int ifail;
  int ic, jc, kc;
  int nsites;

  nsites = coords_nsites();
  printf("nsites %d\n", nsites);

  for (ic = limits->imin; ic <= limits->imax; ic++) {
    for (jc = limits->jmin; jc <= limits->jmax; jc++) {
      for (kc = limits->kmin; kc <= limits->kmax; kc++) {

	/* We are at ic,jc,kc */

	index = coords_index(ic, jc, kc);
	/* printf("%3d %3d %3d %8d %8d\n", ic, jc, kc, index, nsites);*/
	ifail = addr_rank0(nsites, index);
	mask[mem_addr_rank0(nsites, index)] = index;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  do_target_kernel1
 *
 *  Target kernel with no explixit vectorisation.
 *
 *****************************************************************************/

__kernel__ void do_target_kernel1(kernel_limit_t * limits, int * mask) {

  int kindex;

  __targetTLPNoStride__(kindex, limits->kernel_iterations) {

    int ic, jc, kc;
    int index;
    int nsites;

    ic = kernel_coords_ic(kindex);
    jc = kernel_coords_jc(kindex);
    kc = kernel_coords_kc(kindex);

    /* We are at ic, jc, kc */

    index = kernel_coords_index(ic, jc, kc);

    nsites = limits->nsites;
    mask[mem_addr_rank0(nsites, index)] = index;
  }

  return;
}

/*****************************************************************************
 *
 *  do_target_kernel2
 *
 *  Target kernel with explicit vectorisation.
 *
 *****************************************************************************/

__kernel__ void do_target_kernel2(kernel_limit_t * limits, int * mask) {

  int kindex;

  __targetTLP__(kindex, limits->kernel_vector_iterations) {

    int iv;
    int ic[NSIMDVL];
    int jc[NSIMDVL];
    int kc[NSIMDVL];
    int index[NSIMDVL];
    int kmask[NSIMDVL];
    int nsites;

    __targetILP__(iv) ic[iv] = kernel_coords_icv(kindex, iv);
    __targetILP__(iv) jc[iv] = kernel_coords_jcv(kindex, iv);
    __targetILP__(iv) kc[iv] = kernel_coords_kcv(kindex, iv);
    __targetILP__(iv) index[iv] = kernel_coords_index(ic[iv], jc[iv], kc[iv]);
    __targetILP__(iv) kmask[iv] = kernel_mask(ic[iv], jc[iv], kc[iv]);

    nsites = limits->nsites;

    __targetILP__(iv) {
      mask[mem_addr_rank0(nsites, index[iv])] = kmask[iv]*index[iv];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  do_check
 *
 *****************************************************************************/

__host__ int do_check(int * iref, int * itarget) {

  int ic, jc, kc, index;
  int nhalo;
  int nlocal[3];

  assert(iref);
  assert(itarget);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
 
  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = coords_index(ic, jc, kc);
	if (iref[index] == itarget[index]) {
	  /* ok */
	}
	else {
	  printf("%3d %3d %3d %8d %8d\n", ic, jc, kc, iref[index], itarget[index]);
	  assert(0);
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  kernel_coords_commit
 *
 *****************************************************************************/

__host__ int kernel_coords_commit(kernel_limit_t * limits) {

  int kiter;

  assert(limits);

  limits->nhalo = coords_nhalo();
  limits->nsites = coords_nsites();
  coords_nlocal(limits->nlocal);

  limits->nklocal[X] = limits->imax - limits->imin + 1;
  limits->nklocal[Y] = limits->jmax - limits->jmin + 1;
  limits->nklocal[Z] = limits->kmax - limits->kmin + 1;

  limits->kernel_iterations
    = limits->nklocal[X]*limits->nklocal[Y]*limits->nklocal[Z];

  /* Check vector length */

  limits->kv_imin = limits->imin;
  limits->kv_jmin = 1 - limits->nhalo;
  limits->kv_kmin = 1 - limits->nhalo;

  limits->nkv_local[X] = limits->nklocal[X];
  limits->nkv_local[Y] = limits->nlocal[Y] + 2*limits->nhalo;
  limits->nkv_local[Z] = limits->nlocal[Z] + 2*limits->nhalo;

  /* Offset of first site must be start of a SIMD vector block */
  kiter = coords_index(limits->kv_imin, limits->kv_jmin, limits->kv_kmin);
  limits->kindex0 = (kiter/NSIMDVL)*NSIMDVL;

  /* Extent of the contiguous block ... */
  kiter = limits->nkv_local[X]*limits->nkv_local[Y]*limits->nkv_local[Z];
  limits->kernel_vector_iterations = kiter;

  printf("kindex0 %d\n", limits->kindex0);
  printf("vec_itr %d\n", limits->kernel_vector_iterations);

  copyConstToTarget(&klimits, limits, sizeof(kernel_limit_t));

  return 0;
}

/*****************************************************************************
 *
 *  kernel_coords_ic
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_ic(int kindex) {

  int ic;

  ic = klimits.imin + kindex/(klimits.nklocal[Y]*klimits.nklocal[Z]);
  assert(1 - klimits.nhalo <= ic);
  assert(ic <= klimits.nlocal[X] + klimits.nhalo);

  return ic;
}

__host__ __target__ int kernel_coords_jc(int kindex) {

  int ic;
  int jc;

  ic = kindex/(klimits.nklocal[Y]*klimits.nklocal[Z]);
  jc = klimits.jmin +
    (kindex - ic*klimits.nklocal[Y]*klimits.nklocal[Z])/klimits.nklocal[Z];
  assert(1 - klimits.nhalo <= jc);
  assert(jc <= klimits.nlocal[Y] + klimits.nhalo);

  return jc;
}

__host__ __target__ int kernel_coords_kc(int kindex) {

  int ic;
  int jc;
  int kc;

  ic = kindex/(klimits.nklocal[Y]*klimits.nklocal[Z]);
  jc = (kindex - ic*klimits.nklocal[Y]*klimits.nklocal[Z])/klimits.nklocal[Z];
  kc = klimits.kmin +
    kindex - ic*klimits.nklocal[Y]*klimits.nklocal[Z] - jc*klimits.nklocal[Z];
  assert(1 - klimits.nhalo <= kc);
  assert(kc <= klimits.nlocal[Z] + klimits.nhalo);

  return kc;
}



__host__ __target__ int kernel_coords_icv(int kindex, int iv) {

  int ic;
  int index;

  index = klimits.kindex0 + kindex + iv;

  ic = index/(klimits.nkv_local[Y]*klimits.nkv_local[Z]);
  assert(1 - klimits.nhalo <= ic);
  assert(ic <= klimits.nlocal[X] + klimits.nhalo);

  return ic;
}

__host__ __target__ int kernel_coords_jcv(int kindex, int iv) {

  int jc;
  int ic;
  int index;

  index = klimits.kindex0 + kindex + iv;

  ic = index/(klimits.nkv_local[Y]*klimits.nkv_local[Z]);
  jc = (index - ic*klimits.nkv_local[Y]*klimits.nkv_local[Z])/klimits.nkv_local[Z];
  assert(1 - klimits.nhalo <= jc);
  assert(jc <= klimits.nlocal[Y] + klimits.nhalo);

  return jc;
}

__host__ __target__ int kernel_coords_kcv(int kindex, int iv) {

  int kc;
  int jc;
  int ic;
  int index;

  index = klimits.kindex0 + kindex + iv;

  ic = index/(klimits.nkv_local[Y]*klimits.nkv_local[Z]);
  jc = (index - ic*klimits.nkv_local[Y]*klimits.nkv_local[Z])/klimits.nkv_local[Z];
  kc = index - ic*klimits.nkv_local[Y]*klimits.nkv_local[Z] - jc*klimits.nkv_local[Z];

  assert(1 - klimits.nhalo <= jc);
  assert(jc <= klimits.nlocal[Y] + klimits.nhalo);
  assert(kc <= klimits.nlocal[Z] + klimits.nhalo);

  return kc;
}

/*****************************************************************************
 *
 *  kernel_mask
 *
 *****************************************************************************/

__host__ __target__ int kernel_mask(int ic, int jc, int kc) {

  if (ic < klimits.imin || ic > klimits.imax ||
      jc < klimits.jmin || jc > klimits.jmax ||
      kc < klimits.kmin || kc > klimits.kmax) return 0;

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

  nhalo = klimits.nhalo;
  yfac = klimits.nlocal[Z] + 2*nhalo;
  xfac = yfac*(klimits.nlocal[Y] + 2*nhalo);

  index = xfac*(nhalo + ic - 1) + yfac*(nhalo + jc - 1) + nhalo + kc - 1; 

  return index;
}

