/*****************************************************************************
 *
 *  cahn_hilliard_stats.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2024 The University of Edinburgh
 *
 *  Contributions:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "phi_cahn_hilliard.h"
#include "util.h"
#include "util_sum.h"

/* Utility container for global statistics */

typedef struct phi_stats_s phi_stats_t;

struct phi_stats_s {
  kahan_t sum1;      /* Glboal sum: single compensation */
  klein_t sum2;      /* Global sum: doubly compensated. */
  double  sum;       /* Current result for sum */
  double  var;       /* Variance */
  double  min;       /* Minimum */
  double  max;       /* Maximum */
  double  vol;       /* Current fluid volume */
};

__host__ int cahn_stats_reduce(phi_ch_t * pch, field_t * obj, map_t * map,
			       phi_stats_t * stats,  int root, MPI_Comm comm);

__global__ void cahn_stats_kahan_sum_kernel(kernel_3d_t k3d, field_t * phi,
					    map_t * map, phi_stats_t * stats);
__global__ void cahn_stats_klein_sum_kernel(kernel_3d_t k3d, field_t * phi,
					    field_t * csum, map_t * map,
					    phi_stats_t * stats);
__global__ void cahn_stats_var_kernel(kernel_3d_t k3d, field_t * phi,
				      map_t * map, phi_stats_t * stats);
__global__ void cahn_stats_min_kernel(kernel_3d_t k3d, field_t * phi,
				      map_t * map, phi_stats_t * stats);

/*****************************************************************************
 *
 * cahn_hilliard_stats_time0
 *
 * Compute an initial order parameter sum in the same way as any later sum.
 *
 *****************************************************************************/

__host__ int cahn_hilliard_stats_time0(phi_ch_t * pch, field_t * phi,
				       map_t * map) {

  phi_stats_t stats = {0};
  MPI_Comm comm = MPI_COMM_NULL;

  assert(pch);
  assert(phi);
  assert(map);

  pe_mpi_comm(pch->pe, &comm);
  cahn_stats_reduce(pch, phi, map, &stats, 0, comm);

  phi->field_init_sum = stats.sum;
  MPI_Bcast(&phi->field_init_sum, 1, MPI_DOUBLE, 0, comm);

  return 0;
}

/*****************************************************************************
 *
 *  cahn_hilliard_stats
 *
 *****************************************************************************/

__host__ int cahn_hilliard_stats(phi_ch_t * pch, field_t * phi, map_t * map) {

  phi_stats_t stats = {.sum1 = {0},
		       .sum2 = {0},
		       .sum  = 0.0,
		       .var  = 0.0,
		       .min  = +FLT_MAX,
		       .max =  -FLT_MAX,
		       .vol  = 0.0};
  MPI_Comm comm = MPI_COMM_NULL;

  assert(pch);
  assert(phi);
  assert(map);

  pe_mpi_comm(pch->pe, &comm);
  cahn_stats_reduce(pch, phi, map, &stats, 0, comm);

  {
    double rvol = 1.0 / stats.vol;

    double fbar = rvol*stats.sum;                /* mean */
    double fvar = rvol*stats.var - fbar*fbar;    /* variance */

    pe_info(pch->pe, "[phi] %14.7e %14.7e%14.7e %14.7e%14.7e\n",
	    stats.sum, fbar, fvar, stats.min, stats.max);
  }

  return 0;
}

/*****************************************************************************
 *
 *  cahn_stats_reduce
 *
 *  Form the local sums are reduce to rank "root".
 *
 *****************************************************************************/

__host__ int cahn_stats_reduce(phi_ch_t * pch, field_t * phi,
			       map_t * map, phi_stats_t * stats,
			       int root, MPI_Comm comm) {
  phi_stats_t local = {0};
  phi_stats_t * stats_d = NULL;

  int nlocal[3] = {0};

  assert(pch);
  assert(phi);
  assert(map);
  assert(stats);

  cs_nlocal(pch->cs, nlocal);

  /* Initialise device values */
  tdpAssert(tdpMalloc((void **) &stats_d, sizeof(phi_stats_t)));
  tdpAssert(tdpMemcpy(stats_d, stats, sizeof(phi_stats_t),
		      tdpMemcpyHostToDevice));

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(pch->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    if (pch->info.conserve) {
      tdpLaunchKernel(cahn_stats_klein_sum_kernel, nblk, ntpb, 0, 0,
		      k3d, phi->target, pch->csum->target, map->target,
		      stats_d);
      tdpAssert(tdpPeekAtLastError());
    }
    else {
      tdpLaunchKernel(cahn_stats_kahan_sum_kernel, nblk, ntpb, 0, 0,
		      k3d, phi->target, map->target, stats_d);
      tdpAssert(tdpPeekAtLastError());
    }

    tdpLaunchKernel(cahn_stats_var_kernel, nblk, ntpb, 0, 0,
		    k3d, phi->target, map->target, stats_d);
    tdpAssert(tdpPeekAtLastError());

    tdpLaunchKernel(cahn_stats_min_kernel, nblk, ntpb, 0, 0,
		    k3d, phi->target, map->target, stats_d);
    tdpAssert(tdpPeekAtLastError());

    tdpAssert(tdpDeviceSynchronize());
  }

  tdpAssert(tdpMemcpy(&local, stats_d, sizeof(phi_stats_t),
		      tdpMemcpyDeviceToHost));

  /* Kernel required */
  {
    int ivol = 0;
    map_volume_local(map, MAP_FLUID, &ivol);
    local.vol = 1.0*ivol;
  }

  {
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Op op = MPI_OP_NULL;

    if (pch->info.conserve) {
      klein_mpi_datatype(&dt);
      klein_mpi_op_sum(&op);

      MPI_Reduce(&local.sum2, &stats->sum2, 1, dt, op, root, comm);
      stats->sum = klein_sum(&stats->sum2);
    }
    else {
      kahan_mpi_datatype(&dt);
      kahan_mpi_op_sum(&op);

      MPI_Reduce(&local.sum1, &stats->sum1, 1, dt, op, root, comm);
      stats->sum = kahan_sum(&stats->sum1);
    }

    MPI_Op_free(&op);
    MPI_Type_free(&dt);
  }

  MPI_Reduce(&local.var, &stats->var, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  MPI_Reduce(&local.min, &stats->min, 1, MPI_DOUBLE, MPI_MIN, root, comm);
  MPI_Reduce(&local.max, &stats->max, 1, MPI_DOUBLE, MPI_MAX, root, comm);
  MPI_Reduce(&local.vol, &stats->vol, 1, MPI_DOUBLE, MPI_SUM, root, comm);

  tdpFree(stats_d);

  return 0;
}

/*****************************************************************************
 *
 *  cahn_stats_kahan_sum_kernel
 *
 *  Sum when no compensation in the time evolution is present.
 *
 *****************************************************************************/

__global__ void cahn_stats_kahan_sum_kernel(kernel_3d_t k3d, field_t * phi,
					    map_t * map, phi_stats_t * stats) {
  int kindex = 0;
  int tid = threadIdx.x;
  __shared__ kahan_t phib[TARGET_MAX_THREADS_PER_BLOCK];

  assert(phi);

  phib[tid].sum = 0.0;
  phib[tid].cs  = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int status = 0;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double phi0 = phi->data[addr_rank1(phi->nsites, 1, index, 0)];
      kahan_add_double(&phib[tid], phi0);
    }
  }

  __syncthreads();

  if (tid == 0) {
    kahan_t sum = kahan_zero();

    for (int it = 0; it < blockDim.x; it++) {
      kahan_add(&sum, phib[it]);
    }

    /* Final result */

    while (atomicCAS(&(stats->sum1.lock), 0, 1) != 0)
      ;

    __threadfence();

    kahan_add(&stats->sum1, sum);

    __threadfence();
    atomicExch(&(stats->sum1.lock), 0);
  }

  return;
}

/*****************************************************************************
 *
 *  cahn_stats_klein_sum_kernel
 *
 *  Compensated sum. We do care how the sum is computed
 *  (a conserved quantity).
 *
 *****************************************************************************/

__global__ void cahn_stats_klein_sum_kernel(kernel_3d_t k3d, field_t * phi,
					    field_t * csum, map_t * map,
					    phi_stats_t * stats) {
  int kindex = 0;
  int tid = threadIdx.x;
  __shared__ klein_t phib[TARGET_MAX_THREADS_PER_BLOCK];

  assert(phi);

  phib[tid].sum = 0.0;
  phib[tid].cs  = 0.0;
  phib[tid].ccs = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int status = 0;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double phi0 = phi->data[addr_rank1(phi->nsites, 1, index, 0)];
      double cmp  = csum->data[addr_rank1(csum->nsites, 1, index, 0)];

      klein_add_double(&phib[tid], cmp);
      klein_add_double(&phib[tid], phi0);
    }
  }

  __syncthreads();

  if (tid == 0) {
    klein_t sum = klein_zero();

    for (int it = 0; it < blockDim.x; it++) {
      klein_add(&sum, phib[it]);
    }

    /* Final result */

    while (atomicCAS(&(stats->sum2.lock), 0, 1) != 0)
      ;

    __threadfence();

    klein_add(&stats->sum2, sum);

    __threadfence();
    atomicExch(&(stats->sum2.lock), 0);
  }

  return;
}

/*****************************************************************************
 *
 *  cahn_stats_var_kernel
 *
 *  Accumulate the variance in phi field (don't care about exact sum).
 *
 *****************************************************************************/

__global__ void cahn_stats_var_kernel(kernel_3d_t k3d, field_t * phi,
				      map_t * map, phi_stats_t * stats) {
  int kindex = 0;
  int tid = threadIdx.x;
  __shared__ double bvar[TARGET_MAX_THREADS_PER_BLOCK];

  assert(phi);
  assert(map);
  assert(stats);

  bvar[tid] = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int status = 0;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double phi0 = phi->data[addr_rank1(phi->nsites, 1, index, 0)];

      bvar[tid] += phi0*phi0;
    }
  }

  __syncthreads();

  if (tid == 0) {
    double var = 0.0;

    for (int it = 0; it < blockDim.x; it++) {
      var += bvar[it];
    }

    /* Final result */
    tdpAtomicAddDouble(&stats->var, var);
  }

  return;
}

/*****************************************************************************
 *
 *  cahn_stats_min_kernel
 *
 *  Local min/max of phi field.
 *
 *****************************************************************************/

__global__ void cahn_stats_min_kernel(kernel_3d_t k3d, field_t * phi,
				      map_t * map, phi_stats_t * stats) {
  int kindex = 0;
  int tid = threadIdx.x;
  __shared__ double bmin[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double bmax[TARGET_MAX_THREADS_PER_BLOCK];

  assert(phi);
  assert(map);
  assert(stats);

  bmin[tid] = +FLT_MAX;
  bmax[tid] = -FLT_MAX;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int status = 0;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double phi0 = phi->data[addr_rank1(phi->nsites, 1, index, 0)];

      bmin[tid] = dmin(bmin[tid], phi0);
      bmax[tid] = dmax(bmax[tid], phi0);
    }
  }

  __syncthreads();

  if (tid == 0) {
    double min = +FLT_MAX;
    double max = -FLT_MAX;

    for (int it = 0; it < blockDim.x; it++) {
      min = dmin(bmin[it], min);
      max = dmax(bmax[it], max);
    }

    /* Final result */

    tdpAtomicMinDouble(&stats->min, min);
    tdpAtomicMaxDouble(&stats->max, max);
  }

  return;
}
