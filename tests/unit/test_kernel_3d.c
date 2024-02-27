/*****************************************************************************
 *
 *  test_kernel_3d.c
 *
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2024 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "kernel_3d.h"

int test_kernel_3d(pe_t * pe);
int test_kernel_3d_ic(pe_t * pe);
int test_kernel_3d_jc(pe_t * pe);
int test_kernel_3d_kc(pe_t * pe);
int test_kernel_3d_cs_index(pe_t * pe);

__global__ void test_kernel_3d_ic_kernel(kernel_3d_t k3d);
__global__ void test_kernel_3d_jc_kernel(kernel_3d_t k3d);
__global__ void test_kernel_3d_kc_kernel(kernel_3d_t k3d);
__global__ void test_kernel_3d_cs_index_kernel(kernel_3d_t k3d, cs_t * cs);

/*****************************************************************************
 *
 *  test_kernel_3d_suite
 *
 *****************************************************************************/

int test_kernel_3d_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_kernel_3d(pe);
  test_kernel_3d_ic(pe);
  test_kernel_3d_jc(pe);
  test_kernel_3d_kc(pe);
  test_kernel_3d_cs_index(pe);

  /* Hard limit 64 bytes for pass-by-value object must not be exceeded */
  assert(sizeof(kernel_3d_t) <= 64);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_kernel_3d
 *
 *****************************************************************************/

int test_kernel_3d(pe_t * pe) {

  int ifail = 0;
  int nhalo = 2;
  int ntotal[3] = {64, 32, 16};
  int nlocal[3] = {0};
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);
  cs_nlocal(cs, nlocal);

  /* kernel halo = (0, 0, 0) */
  {
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(cs, lim);
    assert(k3d.nhalo       == nhalo);
    assert(k3d.nlocal[X]   == nlocal[X]);
    assert(k3d.nlocal[Y]   == nlocal[Y]);
    assert(k3d.nlocal[Z]   == nlocal[Z]);
    assert(k3d.kindex0     == cs_index(cs, 1, 1, 1));
    assert(k3d.kiterations == nlocal[X]*nlocal[Y]*nlocal[Z]);
    assert(k3d.nklocal[X]  == nlocal[X]);
    assert(k3d.nklocal[Y]  == nlocal[Y]);
    assert(k3d.nklocal[Z]  == nlocal[Z]);
    assert(k3d.lim.imin    == lim.imin);
    assert(k3d.lim.imax    == lim.imax);
    assert(k3d.lim.jmin    == lim.jmin);
    assert(k3d.lim.jmax    == lim.jmax);
    assert(k3d.lim.kmin    == lim.kmin);
    assert(k3d.lim.kmax    == lim.kmax);
  }

  /* kernel halo = (1, 1, 0) */
  {
    cs_limits_t lim = {0, nlocal[X]+1, 0, nlocal[Y]+1, 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(cs, lim);
    assert(k3d.kindex0     == cs_index(cs, 0, 0, 1));
    assert(k3d.kiterations == (nlocal[X] + 2)*(nlocal[Y] + 2)*nlocal[Z]);
    assert(k3d.nklocal[X]  == nlocal[X] + 2);
    assert(k3d.nklocal[Y]  == nlocal[Y] + 2);
    assert(k3d.nklocal[Z]  == nlocal[Z]    );
    assert(k3d.lim.imin    == 0);
    assert(k3d.lim.imax    == nlocal[X] + 1);
    assert(k3d.lim.jmin    == 0);
    assert(k3d.lim.jmax    == nlocal[Y] + 1);
    assert(k3d.lim.kmin    == 1);
    assert(k3d.lim.kmax    == nlocal[Z]);
  }

  /* kernel halo = (2, 2, 2) */
  {
    cs_limits_t lim = {-1, nlocal[X]+2, -1, nlocal[Y]+2, -1, nlocal[Z]+2};
    kernel_3d_t k3d = kernel_3d(cs, lim);
    assert(k3d.nhalo       == nhalo);
    assert(k3d.kindex0     == cs_index(cs, -1, -1, -1));
    assert(k3d.kiterations == (nlocal[X]+4)*(nlocal[Y]+4)*(nlocal[Z]+4));
    assert(k3d.nklocal[X]  == nlocal[X] + 4);
    assert(k3d.nklocal[Y]  == nlocal[Y] + 4);
    assert(k3d.nklocal[Z]  == nlocal[Z] + 4);
    assert(k3d.lim.imin    == -1);
    assert(k3d.lim.imax    == nlocal[X] + 2);
    assert(k3d.lim.jmin    == -1);
    assert(k3d.lim.jmax    == nlocal[Y] + 2);
    assert(k3d.lim.kmin    == -1);
    assert(k3d.lim.kmax    == nlocal[Z] + 2);
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_kernel_3d_ic
 *
 *****************************************************************************/

int test_kernel_3d_ic(pe_t * pe) {

  int ifail = 0;
  int nhalo = 1;
  int ntotal[3] = {64, 32, 16};
  int nlocal[3] = {0};
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);
  cs_nlocal(cs, nlocal);

  /* Kernel */
  {
    dim3 ntpb = {};
    dim3 nblk = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);
    tdpLaunchKernel(test_kernel_3d_ic_kernel, nblk, ntpb, 0, 0, k3d);
    tdpAssert( tdpDeviceSynchronize() );
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_kernel_3d_jc
 *
 *****************************************************************************/

int test_kernel_3d_jc(pe_t * pe) {

  int ifail = 0;
  int nhalo = 2;
  int ntotal[3] = {64, 32, 16};
  int nlocal[3] = {0};
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);
  cs_nlocal(cs, nlocal);

  /* Kernel */
  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 0, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);
    tdpLaunchKernel(test_kernel_3d_jc_kernel, nblk, ntpb, 0, 0, k3d);
    tdpAssert( tdpDeviceSynchronize() );
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_kernel_3d_kc
 *
 *****************************************************************************/

int test_kernel_3d_kc(pe_t * pe) {

  int ifail = 0;
  int nhalo = 1;
  int ntotal[3] = {64, 32, 16};
  int nlocal[3] = {0};
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);
  cs_nlocal(cs, nlocal);

  /* Kernel */
  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 0, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);
    tdpLaunchKernel(test_kernel_3d_kc_kernel, nblk, ntpb, 0, 0, k3d);
    tdpAssert( tdpDeviceSynchronize() );
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_kernel_3d_cs_index
 *
 *****************************************************************************/

int test_kernel_3d_cs_index(pe_t * pe) {

  int ifail = 0;
  int nhalo = 2;
  int ntotal[3] = {64, 32, 16};
  int nlocal[3] = {0};
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);
  cs_nlocal(cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {-1, nlocal[X]+2, -1, nlocal[Y]+2, -1, nlocal[Z]+2};
    kernel_3d_t k3d = kernel_3d(cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(test_kernel_3d_cs_index_kernel, nblk, ntpb, 0, 0,
		    k3d, cs->target);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpDeviceSynchronize() );
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_kernel_3d_ic_kernel
 *
 *****************************************************************************/

__global__ void test_kernel_3d_ic_kernel(kernel_3d_t k3d) {

  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {
    int ic = kernel_3d_ic(&k3d, kindex);
    assert(k3d.lim.imin <= ic && ic <= k3d.lim.imax);
  }

  return;
}

/*****************************************************************************
 *
 *  test_kernel_3d_jc_kernel
 *
 *****************************************************************************/

__global__ void test_kernel_3d_jc_kernel(kernel_3d_t k3d) {

  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {
    int jc = kernel_3d_jc(&k3d, kindex);
    assert(k3d.lim.jmin <= jc && jc <= k3d.lim.jmax);
  }

  return;
}

/*****************************************************************************
 *
 *  test_kernel_3d_kc_kernel
 *
 *****************************************************************************/

__global__ void test_kernel_3d_kc_kernel(kernel_3d_t k3d) {

  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {
    int kc = kernel_3d_kc(&k3d, kindex);
    assert(k3d.lim.kmin <= kc && kc <= k3d.lim.kmax);
  }

  return;
}

/*****************************************************************************
 *
 *  test_kernel_3d_cs_index_kernel
 *
 *****************************************************************************/

__global__ void test_kernel_3d_cs_index_kernel(kernel_3d_t k3d, cs_t * cs) {

  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {
    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int i0 = kernel_3d_cs_index(&k3d, ic, jc, kc);
    {
      int index = cs_index(cs, ic, jc, kc);
      assert(index == i0);
    }
  }

  return;
}
