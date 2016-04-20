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

typedef struct kernel_s kernel_t;
typedef struct data_s data_t;

struct data_s {
  int isum;
  int * idata;
  data_t * target;
};

struct kernel_s {
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

static kernel_t hlimits;
static __targetConst__ kernel_t klimits;

__host__ int do_test_kernel(kernel_t * limits, data_t * data);
__host__ int do_host_kernel(int * mask, int * isum);
__host__ int do_check(int * iref, int * itarget);
__global__ void do_target_kernel1(data_t * data);
__global__ void do_target_kernel2(data_t * data);
__global__ void do_target_kernel1r(data_t * data);

__host__           dim3 kernel_blocks(kernel_t * limits, dim3 ntpb);
__host__            int kernel_coords_commit(kernel_t * limits);
__host__ __target__ int kernel_coords_ic(int kindex);
__host__ __target__ int kernel_coords_jc(int kindex);
__host__ __target__ int kernel_coords_kc(int kindex);
__host__ __target__ int kernel_coords_icv(int kindex, int iv);
__host__ __target__ int kernel_coords_jcv(int kindex, int iv);
__host__ __target__ int kernel_coords_kcv(int kindex, int iv);
__host__ __target__ int kernel_mask(int ic, int jc, int kc);
__host__ __target__ int kernel_coords_index(int ic, int jc, int kc);

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
  kernel_t lim;
  kernel_t * limits = &lim;

  pe_init_quiet();
  coords_init();
  coords_nlocal(nlocal);

  target_thread_info();

  data_create(data);

  lim.imin = 1; lim.imax = nlocal[X];
  lim.jmin = 1; lim.jmax = nlocal[Y];
  lim.kmin = 1; lim.kmax = nlocal[Z];
  do_test_kernel(limits, data);

  lim.imin = 0; lim.imax = nlocal[X] + 1;
  lim.jmin = 0; lim.jmax = nlocal[Y] + 1;
  lim.kmin = 0; lim.kmax = nlocal[Z] + 1;
  do_test_kernel(limits, data);

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

__host__ int do_test_kernel(kernel_t * limits, data_t * data) {

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

  ntpb.x = __host_threads_per_block();
  ntpb.y = 1;
  ntpb.z = 1;
  nblk = kernel_blocks(limits, ntpb);

  __host_launch_kernel(do_target_kernel1, nblk, ntpb, data->target);
  targetDeviceSynchronise();

  printf("Finish kernel 1\n");
  data_copy(data, 1);
  do_check(iref, data->idata);

  /* TODO zero data between invocations. */
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

  nsites = coords_nsites();

  *isum = 0;

  for (ic = hlimits.imin; ic <= hlimits.imax; ic++) {
    for (jc = hlimits.jmin; jc <= hlimits.jmax; jc++) {
      for (kc = hlimits.kmin; kc <= hlimits.kmax; kc++) {

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

  __target_simt_parallel_for(kindex, klimits.kernel_iterations, 1) {

    int ic, jc, kc;
    int index;
    int nsites;

    ic = kernel_coords_ic(kindex);
    jc = kernel_coords_jc(kindex);
    kc = kernel_coords_kc(kindex);

    /* We are at ic, jc, kc */

    index = kernel_coords_index(ic, jc, kc);

    nsites = klimits.nsites;
    data->idata[mem_addr_rank0(nsites, index)] = index;
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
    int nsites;
    int ia;
    __target_simt_threadIdx_init();
    psum[threadIdx.x] = 0;

    __target_simt_for(kindex, klimits.kernel_iterations, 1) {

      ic = kernel_coords_ic(kindex);
      jc = kernel_coords_jc(kindex);
      kc = kernel_coords_kc(kindex);

      /* We are at ic, jc, kc */

      index = kernel_coords_index(ic, jc, kc);

      nsites = klimits.nsites;
      data->idata[mem_addr_rank0(nsites, index)] = index;
      psum[threadIdx.x] += 1;
    }

    /* Reduction */

    for (ia = blockDim.x/2; ia > 0; ia /= 2) {
      __target_syncthreads();
      if (threadIdx.x < ia) {
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

  __targetTLP__(kindex, klimits.kernel_vector_iterations) {

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

    nsites = klimits.nsites;

    __targetILP__(iv) {
      data->idata[mem_addr_rank0(nsites, index[iv])] = kmask[iv]*index[iv];
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
  data->idata = (int *) calloc(nsites, sizeof(int));
  assert(data->idata);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    data->target = data;
  }
  else {
    printf("Create device copy\n");
    targetCalloc((void **) &(data->target), sizeof(data_t));
    targetCalloc((void **) &tmp, nsites*sizeof(int));
    copyToTarget(&(data->target->idata), &tmp, sizeof(int *));
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
    nsites = coords_nsites();
    copyFromTarget(&tmp, &(data->target->idata), sizeof(int *));
    if (flag == 0) {
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

/*****************************************************************************
 *
 *  kernel_coords_commit
 *
 *****************************************************************************/

__host__ int kernel_coords_commit(kernel_t * limits) {

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

  hlimits = *limits;
  copyConstToTarget(&klimits, limits, sizeof(kernel_t));

  return 0;
}


/*****************************************************************************
 *
 *  kernel_coords_ic
 *
 *****************************************************************************/

__host__ __target__ int kernel_coords_ic(int kindex) {

  int ic;

#ifdef __CUDA_ARCH__
  const kernel_t limits = klimits;
#else
  const kernel_t limits = hlimits;
#endif

  ic = limits.imin + kindex/(limits.nklocal[Y]*limits.nklocal[Z]);
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

#ifdef __CUDA_ARCH__
  const kernel_t limits = klimits;
#else
  const kernel_t limits = hlimits;
#endif

  ic = kindex/(limits.nklocal[Y]*limits.nklocal[Z]);
  jc = limits.jmin +
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

#ifdef __CUDA_ARCH__
  const kernel_t limits = klimits;
#else
  const kernel_t limits = hlimits;
#endif

  ic = kindex/(limits.nklocal[Y]*limits.nklocal[Z]);
  jc = (kindex - ic*limits.nklocal[Y]*limits.nklocal[Z])/limits.nklocal[Z];
  kc = limits.kmin +
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

#ifdef __CUDA_ARCH__
  const kernel_t limits = klimits;
#else
  const kernel_t limits = hlimits;
#endif

  index = limits.kindex0 + kindex + iv;

  ic = index/(limits.nkv_local[Y]*limits.nkv_local[Z]);
  assert(1 - limits.nhalo <= ic);
  assert(ic <= limits.nlocal[X] + limits.nhalo);

  return ic;
}

__host__ __target__ int kernel_coords_jcv(int kindex, int iv) {

  int jc;
  int ic;
  int index;

#ifdef __CUDA_ARCH__
  const kernel_t limits = klimits;
#else
  const kernel_t limits = hlimits;
#endif

  index = limits.kindex0 + kindex + iv;

  ic = index/(limits.nkv_local[Y]*limits.nkv_local[Z]);
  jc = (index - ic*limits.nkv_local[Y]*limits.nkv_local[Z])/limits.nkv_local[Z];
  assert(1 - limits.nhalo <= jc);
  assert(jc <= limits.nlocal[Y] + limits.nhalo);

  return jc;
}

__host__ __target__ int kernel_coords_kcv(int kindex, int iv) {

  int kc;
  int jc;
  int ic;
  int index;

#ifdef __CUDA_ARCH__
  const kernel_t limits = klimits;
#else
  const kernel_t limits = hlimits;
#endif

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

#ifdef __CUDA_ARCH__
  const kernel_t limits = klimits;
#else
  const kernel_t limits = hlimits;
#endif

  if (ic < limits.imin || ic > limits.imax ||
      jc < limits.jmin || jc > limits.jmax ||
      kc < limits.kmin || kc > limits.kmax) return 0;

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

#ifdef __CUDA_ARCH__
  const kernel_t limits = klimits;
#else
  const kernel_t limits = hlimits;
#endif

  nhalo = limits.nhalo;
  yfac = limits.nlocal[Z] + 2*nhalo;
  xfac = yfac*(limits.nlocal[Y] + 2*nhalo);

  index = xfac*(nhalo + ic - 1) + yfac*(nhalo + jc - 1) + nhalo + kc - 1; 

  return index;
}

/*****************************************************************************
 *
 *  kernel_blocks
 *
 *****************************************************************************/

__host__ dim3 kernel_blocks(kernel_t * limits, dim3 ntpb) {

  dim3 nblocks; 

  nblocks.x = (limits->kernel_iterations + ntpb.x - 1)/ntpb.x;
  nblocks.y = 1;
  nblocks.z = 1;

  return nblocks;
}

