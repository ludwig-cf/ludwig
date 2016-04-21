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
#include "kernel.h"
#include "memory.h"

typedef struct data_s data_t;

struct data_s {
  int nsites;
  int isum;
  int * idata;
  data_t * target;
};

__host__ int do_test_kernel(kernel_lim_t limits, data_t * data);
__host__ int do_host_kernel(int * mask, int * isum);
__host__ int do_check(int * iref, int * itarget);
__global__ void do_target_kernel1(data_t * data);
__global__ void do_target_kernel2(data_t * data);
__global__ void do_target_kernel1r(data_t * data);

__host__ int data_create(data_t * data);
__host__ int data_free(data_t * data);
__host__ int data_zero(data_t * data);
__host__ int data_copy(data_t * data, int flag);

/*****************************************************************************
 *
 *  test_kernel_suite
 *
 *****************************************************************************/

__host__ int test_kernel_suite(void) {

  int nlocal[3];
  data_t sdata;
  data_t * data = &sdata;
  kernel_lim_t lim;

  pe_init_quiet();
  coords_init();
  coords_nlocal(nlocal);

  target_thread_info();

  data_create(data);

  lim.imin = 1; lim.imax = nlocal[X];
  lim.jmin = 1; lim.jmax = nlocal[Y];
  lim.kmin = 1; lim.kmax = nlocal[Z];
  do_test_kernel(lim, data);

  lim.imin = 0; lim.imax = nlocal[X] + 1;
  lim.jmin = 0; lim.jmax = nlocal[Y] + 1;
  lim.kmin = 0; lim.kmax = nlocal[Z] + 1;
  do_test_kernel(lim, data);

  data_free(data);

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

__host__ int do_test_kernel(kernel_lim_t limits, data_t * data) {

  int nsites;
  int * iref = NULL;
  dim3 ntpb;
  dim3 nblk;
  int isum;

  /* Allocate space for reference */

  nsites = coords_nsites();

  /* Additional issue. The assertions in memory.c actaully catch
   * legal address owing to the division by NSIMDVL in the macros.
   * It might be worth moving the division to after the assertion
   * in memory.c */

  /* In the meantime, we need... */
  assert(nsites % NSIMDVL == 0);

  iref = (int *) calloc(nsites, sizeof(int));
  assert(iref);

  kernel_coords_commit(limits);

  isum = 0;
  do_host_kernel(iref, &isum);
  printf("Host kernel returns isum = %d\n", isum);

  /* Target */

  kernel_launch_param(1, &nblk, &ntpb);

  __host_launch_kernel(do_target_kernel1, nblk, ntpb, data->target);
  targetDeviceSynchronise();

  printf("Finish kernel 1\n");
  data_copy(data, 1);
  do_check(iref, data->idata);

  /* Vectorised */

  kernel_launch_param(NSIMDVL, &nblk, &ntpb);

  __host_launch_kernel(do_target_kernel2, nblk, ntpb, data->target);
  targetDeviceSynchronise();

  printf("Finish kernel 2\n");
  data_copy(data, 1);
  do_check(iref, data->idata);

  /* Reduction kernels */

  data_zero(data);
  __host_launch_kernel(do_target_kernel1r, nblk, ntpb, data->target);
  targetDeviceSynchronise();

  printf("Finish kernel 3\n");
  data_copy(data, 1);
  do_check(iref, data->idata);
  printf("isum %d data->Isum %d\n", isum, data->isum);
  assert(isum == data->isum);

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

__host__ int do_host_kernel(int * mask, int * isum) {

  int index;
  int ifail;
  int ic, jc, kc;
  int nsites;
  kernel_lim_t limits;

  assert(mask);
  assert(isum);

  kernel_lim(&limits);
  nsites = coords_nsites();

  *isum = 0;

  for (ic = limits.imin; ic <= limits.imax; ic++) {
    for (jc = limits.jmin; jc <= limits.jmax; jc++) {
      for (kc = limits.kmin; kc <= limits.kmax; kc++) {

	/* We are at ic,jc,kc */

	index = coords_index(ic, jc, kc);
	ifail = addr_rank0(nsites, index);
	assert(ifail >= 0 && ifail < nsites);

	mask[mem_addr_rank0(nsites, index)] = index;
	*isum += 1;
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

__global__ void do_target_kernel1(data_t * data) {

  int kindex;

  __target_simt_parallel_for(kindex, kernel_iterations(), 1) {

    int ic, jc, kc;
    int index;

    ic = kernel_coords_ic(kindex);
    jc = kernel_coords_jc(kindex);
    kc = kernel_coords_kc(kindex);

    /* We are at ic, jc, kc */

    index = kernel_coords_index(ic, jc, kc);

    data->idata[mem_addr_rank0(data->nsites, index)] = index;
  }

  return;
}

/*****************************************************************************
 *
 *  do_target_kernel1r
 *
 *  Target kernel with no explixit vectorisation. Reduction.
 *
 *****************************************************************************/

__global__ void do_target_kernel1r(data_t * data) {

  int kindex;
  __shared__ int psum[TARGET_MAX_THREADS_PER_BLOCK];

  __target_simt_parallel_region() {

    int ic, jc, kc;
    int index;
    int ia;
    __target_simt_threadIdx_init();
    psum[threadIdx.x] = 0;

    __target_simt_for(kindex, kernel_iterations(), 1) {

      ic = kernel_coords_ic(kindex);
      jc = kernel_coords_jc(kindex);
      kc = kernel_coords_kc(kindex);

      /* We are at ic, jc, kc */

      index = kernel_coords_index(ic, jc, kc);

      data->idata[mem_addr_rank0(data->nsites, index)] = index;
      psum[threadIdx.x] += 1;
    }

    /* Reduction (nthreads power of two) */

    for (ia = blockDim.x/2; ia > 0; ia /= 2) {
      __target_syncthreads();
      if (threadIdx.x < ia) {
	printf("sum %d %d %d\n", blockDim.x, threadIdx.x, threadIdx.x+ia);
	psum[threadIdx.x] += psum[threadIdx.x + ia];
      }
    }

    if (threadIdx.x == 0) {
      /* Kludge: only valid for 1 element of psum (psum[0]) */
      target_atomic_add_int(&data->isum, psum, 1);
    }
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

__global__ void do_target_kernel2(data_t * data) {

  int kindex;

  __targetTLP__(kindex, kernel_vector_iterations()) {

    int iv;
    int ic[NSIMDVL];
    int jc[NSIMDVL];
    int kc[NSIMDVL];
    int index[NSIMDVL];
    int kmask[NSIMDVL];

    __targetILP__(iv) ic[iv] = kernel_coords_icv(kindex, iv);
    __targetILP__(iv) jc[iv] = kernel_coords_jcv(kindex, iv);
    __targetILP__(iv) kc[iv] = kernel_coords_kcv(kindex, iv);
    __targetILP__(iv) index[iv] = kernel_coords_index(ic[iv], jc[iv], kc[iv]);
    __targetILP__(iv) kmask[iv] = kernel_mask(ic[iv], jc[iv], kc[iv]);

    __targetILP__(iv) {
      data->idata[mem_addr_rank0(data->nsites, index[iv])] = kmask[iv]*index[iv];
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
 *  data_create
 *
 *****************************************************************************/

__host__ int data_create(data_t * data) {

  int ndevice;
  int nsites;
  int * tmp;

  /* host */
  nsites = coords_nsites();
  data->nsites = nsites;
  data->idata = (int *) calloc(nsites, sizeof(int));
  assert(data->idata);

  targetGetDeviceCount(&ndevice);
  printf("Number of devices: %d\n", ndevice);

  if (ndevice == 0) {
    data->target = data;
  }
  else {
    printf("Create device copy\n");
    targetCalloc((void **) &(data->target), sizeof(data_t));
    targetCalloc((void **) &tmp, nsites*sizeof(int));
    copyToTarget(&(data->target->idata), &tmp, sizeof(int *));
    copyToTarget(&data->target->nsites, &nsites, sizeof(int));
  }

  return 0;
}

/*****************************************************************************
 *
 *  data_free
 *
 *****************************************************************************/

__host__ int data_free(data_t * data) {

  int ndevice;
  int * tmp;

  free(data->idata);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* No action */
  }
  else {
    copyFromTarget(&tmp, &(data->target->idata), sizeof(int *));
    targetFree(tmp);
    targetFree(data->target);
  }

  return 0;
}

/*****************************************************************************
 *
 *  data_zero
 *
 *****************************************************************************/

__host__ int data_zero(data_t * data) {

  int n;
  int nsites;

  assert(data);

  nsites = coords_nsites();

  data->isum = 0;
  for (n = 0; n < nsites; n++) {
    data->idata[n] = 0;
  }

  data_copy(data, 0);

  return 0;
}

/*****************************************************************************
 *
 *  data_copy
 *
 *****************************************************************************/

__host__ int data_copy(data_t * data, int flag) {

  int ndevice;
  int nsites;
  int * tmp;

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Alias is enough */
  }
  else {
    nsites = data->nsites;
    copyFromTarget(&tmp, &(data->target->idata), sizeof(int *));
    if (flag == 0) {
      copyToTarget(&data->target->nsites, &nsites, sizeof(int));
      copyToTarget(&data->target->isum, &data->isum, sizeof(int));
      copyToTarget(tmp, data->idata, nsites*sizeof(int));
    }
    if (flag == 1) {
      copyFromTarget(&data->isum, &data->target->isum, sizeof(int));
      copyFromTarget(data->idata, tmp, nsites*sizeof(int));
    }
  }

  return 0;
}
