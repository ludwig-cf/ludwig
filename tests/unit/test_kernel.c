/*****************************************************************************
 *
 *  test_kernel.c
 *
 *  Test kernel coverage depending on target.
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
#include <stdio.h>

#include "pe.h"
#include "coords.h"
#include "memory.h"

#define __host__ __targetHost__
#define __kernel__ __targetEntry__

typedef struct kernel_limit_s kernel_limit_t;
struct kernel_limit_s {
  int imin;
  int imax;
  int jmin;
  int jmax;
  int kmin;
  int kmax;
  int nhalo;
  int nsites;
  int nlocal[3];
  int nklocal[3];
  int kernel_iterations;
};

static __targetConst__ kernel_limit_t klimits;

__host__ int do_test_kernel(kernel_limit_t * limits);
__host__ int do_host_kernel(kernel_limit_t * limits, int * mask);
__kernel__ void do_target_kernel1(kernel_limit_t * limits, int * mask);
__kernel__ void do_target_kernel2(kernel_limit_t * limits, int * mask);

__host__            int kernel_coords_commit(kernel_limit_t * limits);
__host__ __target__ int kernel_coords_ic(int kindex);
__host__ __target__ int kernel_coords_jc(int kindex);
__host__ __target__ int kernel_coords_kc(int kindex);
__host__ __target__ int kernel_coords_icv(int kindex, int iv);
__host__ __target__ int kernel_coords_jcv(int kindex, int iv);
__host__ __target__ int kernel_coords_kcv(int kindex, int iv);
__host__ __target__ int kernel_coords_index(int ic, int jc, int kc);

/*****************************************************************************
 *
 *  test_kernel_suite
 *
 *****************************************************************************/

int test_kernel_suite(void) {

  pe_init_quiet();

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
  int * ihost = NULL;
  int * itarget1 = NULL;
  int * itarget2 = NULL;

  /* Allocate space for masks */

  nsites = coords_nsites();
  ihost = (int *) calloc(nsites, sizeof(int));
  itarget1 = (int *) calloc(nsites, sizeof(int));
  itarget2 = (int *) calloc(nsites, sizeof(int));

  assert(ihost);
  assert(itarget1);
  assert(itarget2);

  kernel_coords_commit(limits);

  do_host_kernel(limits, ihost);

  /* targetLaunch(do_target_kernel1); */
  /* targetLaunch(do_target_kernel2); */

  free(itarget2);
  free(itarget1);
  free(ihost);

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
  int ic, jc, kc;
  int nsites;

  nsites = coords_nsites();

  for (ic = limits->imin; ic <= limits->imax; ic++) {
    for (jc = limits->jmin; jc <= limits->jmax; jc++) {
      for (kc = limits->kmin; kc <= limits->kmax; kc++) {

	/* We are at ic,jc,kc */

	index = coords_index(ic, jc, kc);
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

    /* We are at ic,jc,kc */

    nsites = limits->nsites;
    index = kernel_coords_index(ic, jc, kc);
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

  __targetTLP__(kindex, limits->kernel_iterations) {

    int iv;
    int ic[NSIMDVL];
    int jc[NSIMDVL];
    int kc[NSIMDVL];
    int index[NSIMDVL];
    int nsites;

    __targetILP__(iv) ic[iv] = kernel_coords_icv(kindex, iv);
    __targetILP__(iv) jc[iv] = kernel_coords_jcv(kindex, iv);
    __targetILP__(iv) kc[iv] = kernel_coords_kcv(kindex, iv);
    __targetILP__(iv) index[iv] = kernel_coords_index(ic[iv], jc[iv], kc[iv]);

    nsites = limits->nsites;

    /*
    __targetILP__(iv) {
      mask[mem_vaddr_rank0(nsites, indexv[iv])] = 1;
    }
    */
  }

  return;
}

/*****************************************************************************
 *
 *  kernel_coords_commit
 *
 *****************************************************************************/

__host__ int kernel_coords_commit(kernel_limit_t * limits) {

  assert(limits);

  limits->nklocal[X] = limits->imax - limits->imin + 1;
  limits->nklocal[Y] = limits->jmax - limits->jmin + 1;
  limits->nklocal[Z] = limits->kmax - limits->kmin + 1;

  /* Check vector length */

  limits->kernel_iterations = limits->nklocal[X]*limits->nklocal[Y]*limits->nklocal[Z];

  copyConstToTarget(&klimits, limits, sizeof(kernel_limit_t));

  return 0;
}

/*****************************************************************************
 *
 *  kernel_coords_ic
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_ic(int kindex) {

  int ic = 0;

  return ic;
}

__host__ __target__ int kernel_coords_jc(int kindex) {

  int jc = 0;

  return jc;
}

__host__ __target__ int kernel_coords_kc(int kindex) {

  int kc = 0;

  return kc;
}



__host__ __target__ int kernel_coords_icv(int kindex, int iv) {

  int ic = 0;

  return ic + iv;
}

__host__ __target__ int kernel_coords_jcv(int kindex, int iv) {

  int jc = 0;

  return jc + iv;
}

__host__ __target__ int kernel_coords_kcv(int kindex, int iv) {

  int kc = 0;

  return kc + iv;
}


__host__ __target__ int kernel_coords_index(int ic, int jc, int kc) {

  int index = 0;

  return index;
}

