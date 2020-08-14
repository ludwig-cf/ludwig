/*****************************************************************************
 *
 *  test_kernel.c
 *
 *  Test kernel coverage depending on target, vector length, and so on.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016-2019 The University of Edinburgh
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
#include "coords.h"
#include "kernel.h"
#include "memory.h"

#include "tests.h"

typedef struct data_s data_t;

struct data_s {
  int nsites;
  int isum;
  int * idata;
  data_t * target;
};

__host__ int do_test_kernel(cs_t * cs, kernel_info_t limits, data_t * data);
__host__ int do_host_kernel(cs_t * cs, kernel_info_t limits, int * mask, int * isum);
__host__ int do_check(cs_t * cs, int * iref, int * itarget);
__host__ int do_test_attributes(pe_t * pe);

__global__ void do_target_kernel1(kernel_ctxt_t * ktx, data_t * data);
__global__ void do_target_kernel2(kernel_ctxt_t * ktx, data_t * data);
__global__ void do_target_kernel1r(kernel_ctxt_t * ktx, data_t * data);

__host__ int data_create(int nsites, data_t * data);
__host__ int data_free(data_t * data);
__host__ int data_zero(data_t * data);
__host__ int data_copy(data_t * data, tdpMemcpyKind flag);

/*****************************************************************************
 *
 *  test_kernel_suite
 *
 *****************************************************************************/

__host__ int test_kernel_suite(void) {

  int nlocal[3];
  int nsites;
  data_t sdata;
  data_t * data = &sdata;
  kernel_info_t lim;
  cs_t * cs = NULL;
  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);
  cs_nlocal(cs, nlocal);

  do_test_attributes(pe);

  /* target_thread_info(); */

  cs_nsites(cs, &nsites);
  data_create(nsites, data);

  lim.imin = 1; lim.imax = nlocal[X];
  lim.jmin = 1; lim.jmax = nlocal[Y];
  lim.kmin = 1; lim.kmax = nlocal[Z];
  do_test_kernel(cs, lim, data);

  lim.imin = 0; lim.imax = nlocal[X] + 1;
  lim.jmin = 0; lim.jmax = nlocal[Y] + 1;
  lim.kmin = 0; lim.kmax = nlocal[Z] + 1;

  do_test_kernel(cs, lim, data);

  data_free(data);

  cs_free(cs);
  pe_info(pe, "PASS     ./unit/test_kernel\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_kernel
 *
 *  For given configuration run the following
 *   1. __host__   "kernel" with standard form to generate reference result
 *   2. __global__ with no explicit vectorisation
 *   3. __global__ with explicit vectorisation
 *
 *****************************************************************************/

__host__ int do_test_kernel(cs_t * cs, kernel_info_t limits, data_t * data) {

  int isum;
  int nsites;
  int nexpect;
  int * iref = NULL;
  dim3 ntpb;
  dim3 nblk;
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);

  /* Allocate space for reference */

  cs_nsites(cs, &nsites);

  /* Additional issue. The assertions in memory.c actaully catch
   * legal address owing to the division by NSIMDVL in the macros.
   * It might be worth moving the division to after the assertion
   * in memory.c */

  /* In the meantime, we need... */
  test_assert(nsites % NSIMDVL == 0);

  isum = 0;
  iref = (int *) calloc(nsites, sizeof(int));
  assert(iref);


  /* Here we need a context, as we use it in this particular
   * host kernel */
  kernel_ctxt_create(cs, 1, limits, &ctxt);
  do_host_kernel(cs, limits, iref, &isum);

  nexpect = (limits.imax - limits.imin + 1)*
            (limits.jmax - limits.jmin + 1)*
            (limits.kmax - limits.kmin + 1);
  test_assert(isum == nexpect);

  /* Target */

  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(do_target_kernel1, nblk, ntpb, 0, 0,
		  ctxt->target, data->target);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  data_copy(data, tdpMemcpyDeviceToHost);
  do_check(cs, iref, data->idata);


  /* Reduction kernel */

  data_zero(data);
  tdpLaunchKernel(do_target_kernel1r, nblk, ntpb, 0, 0,
		 ctxt->target, data->target);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  data_copy(data, tdpMemcpyDeviceToHost);
  do_check(cs, iref, data->idata);

  test_assert(isum == data->isum);

  kernel_ctxt_free(ctxt);


  /* Vectorised */

  kernel_ctxt_create(cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  data_zero(data);
  tdpLaunchKernel(do_target_kernel2, nblk, ntpb, 0, 0,
		  ctxt->target, data->target);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  data_copy(data, tdpMemcpyDeviceToHost);
  do_check(cs, iref, data->idata);

  kernel_ctxt_free(ctxt);

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

__host__ int do_host_kernel(cs_t * cs, kernel_info_t limits, int * mask,
			    int * isum) {

  int index;
  int ifail;
  int ic, jc, kc;
  int nsites;

  assert(mask);
  assert(isum);

  cs_nsites(cs, &nsites);

  *isum = 0;

  for (ic = limits.imin; ic <= limits.imax; ic++) {
    for (jc = limits.jmin; jc <= limits.jmax; jc++) {
      for (kc = limits.kmin; kc <= limits.kmax; kc++) {

	/* We are at ic,jc,kc */

	index = cs_index(cs, ic, jc, kc);
	ifail = mem_addr_rank0(nsites, index);
	test_assert(ifail >= 0 && ifail < nsites);

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

__global__ void do_target_kernel1(kernel_ctxt_t * ktx, data_t * data) {

  int kiter;
  int kindex;

  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    int ic, jc, kc;
    int index;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    /* We are at ic, jc, kc */

    index = kernel_coords_index(ktx, ic, jc, kc);

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

__global__ void do_target_kernel1r(kernel_ctxt_t * ktx, data_t * data) {

  int kiter;
  int kindex;
  int ic, jc, kc;
  int index;
  int block_sum;
  __shared__ int psum[TARGET_MAX_THREADS_PER_BLOCK];

  psum[threadIdx.x] = 0;
  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    /* We are at ic, jc, kc */

    index = kernel_coords_index(ktx, ic, jc, kc);

    data->idata[mem_addr_rank0(data->nsites, index)] = index;
    psum[threadIdx.x] += 1;
  }

  /* Reduction, two part */

  block_sum = tdpAtomicBlockAddInt(psum);

  if (threadIdx.x == 0) {
    tdpAtomicAddInt(&data->isum, block_sum);
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

__global__ void do_target_kernel2(kernel_ctxt_t * ktx, data_t * data) {

  int kindex;
  int kiter;

  int iv;
  int ic[NSIMDVL];
  int jc[NSIMDVL];
  int kc[NSIMDVL];
  int index[NSIMDVL];
  int kmask[NSIMDVL];

  assert(ktx);

  kiter = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiter, NSIMDVL) {

    kernel_coords_v(ktx, kindex, ic, jc, kc);
    kernel_coords_index_v(ktx, ic, jc, kc, index);
    kernel_mask_v(ktx, ic, jc, kc, kmask);

    for_simd_v(iv, NSIMDVL) {
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

__host__ int do_check(cs_t * cs, int * iref, int * itarget) {

  int ic, jc, kc, index;
  int nhalo;
  int nlocal[3];

  assert(iref);
  assert(itarget);

  cs_nhalo(cs, &nhalo);
  cs_nlocal(cs, nlocal);
 
  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = cs_index(cs, ic, jc, kc);
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

__host__ int data_create(int nsites, data_t * data) {

  int ndevice;

  /* host */

  data->nsites = nsites;
  data->idata = (int *) calloc(nsites, sizeof(int));
  assert(data->idata);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    data->target = data;
  }
  else {
    int * tmp;
    tdpAssert(tdpMalloc((void **) &(data->target), sizeof(data_t)));
    tdpAssert(tdpMalloc((void **) &tmp, nsites*sizeof(int)));
    tdpAssert(tdpMemset(tmp, 0, nsites*sizeof(int)));
    tdpAssert(tdpMemcpy(&data->target->idata, &tmp, sizeof(int *),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&data->target->nsites, &nsites, sizeof(int),
			tdpMemcpyHostToDevice));
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

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* No action */
  }
  else {
    tdpAssert(tdpMemcpy(&tmp, &(data->target->idata), sizeof(int *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpFree(tmp));
    tdpAssert(tdpFree(data->target));
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

  assert(data);

  data->isum = 0;
  for (n = 0; n < data->nsites; n++) {
    data->idata[n] = 0;
  }

  data_copy(data, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  data_copy
 *
 *****************************************************************************/

__host__ int data_copy(data_t * data, tdpMemcpyKind flag) {

  int ndevice;
  int nsites;
  int * tmp;

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Alias is enough */
  }
  else {
    nsites = data->nsites;
    tdpAssert(tdpMemcpy(&tmp, &(data->target->idata), sizeof(int *),
			tdpMemcpyDeviceToHost));
    if (flag == tdpMemcpyHostToDevice) {
      tdpAssert(tdpMemcpy(&data->target->nsites, &nsites, sizeof(int), flag));
      tdpAssert(tdpMemcpy(&data->target->isum, &data->isum, sizeof(int), flag));
      tdpAssert(tdpMemcpy(tmp, data->idata, nsites*sizeof(int), flag));
    }
    if (flag == tdpMemcpyDeviceToHost) {
      tdpAssert(tdpMemcpy(&data->isum, &data->target->isum, sizeof(int), flag));
      tdpAssert(tdpMemcpy(data->idata, tmp, nsites*sizeof(int), flag));
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  do_test_attributes
 *
 *****************************************************************************/

int do_test_attributes(pe_t * pe) {

  int ndevice;
  int device;
  int value;

  assert(pe);

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpAssert(tdpGetDevice(&device));
//    tdpAssert(tdpDeviceGetAttribute(&value, tdpDevAttrManagedMemory, device));
    pe_info(pe, "Device:                 %d\n", device);
//    pe_info(pe, "tdpDevAttrManagedMemory %d\n", value);
  }

  return 0;
}
