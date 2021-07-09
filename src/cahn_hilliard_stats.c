/*****************************************************************************
 *
 *  cahn_hilliard_stats.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
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
#include "field_s.h"
#include "map_s.h"
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

__global__ void cahn_stats_kahan_sum_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    map_t * map, phi_stats_t * stats);
__global__ void cahn_stats_klein_sum_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    field_t * csum, map_t * map,
					    phi_stats_t * stats);
__global__ void cahn_stats_var_kernel(kernel_ctxt_t * ktx, field_t * phi,
				      map_t * map, phi_stats_t * stats);
__global__ void cahn_stats_min_kernel(kernel_ctxt_t * ktx, field_t * phi,
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

  phi_stats_t stats = {};
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

  phi_stats_t stats = {};
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
  phi_stats_t local = {};
  phi_stats_t * stats_d = NULL;

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(pch);
  assert(phi);
  assert(map);
  assert(stats);

  cs_nlocal(pch->cs, nlocal);
  
  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(pch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpAssert(tdpMalloc((void **) &stats_d, sizeof(phi_stats_t)));

  if (pch->info.conserve) {
    tdpLaunchKernel(cahn_stats_klein_sum_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, phi->target, pch->csum->target, map->target,
		    stats_d);
    tdpAssert(tdpPeekAtLastError());
  }
  else {
    tdpLaunchKernel(cahn_stats_kahan_sum_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, phi->target, map->target, stats_d);
    tdpAssert(tdpPeekAtLastError());
  }

  tdpLaunchKernel(cahn_stats_var_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, phi->target, map->target, stats_d);
  tdpAssert(tdpPeekAtLastError());

  tdpLaunchKernel(cahn_stats_min_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, phi->target, map->target, stats_d);
  tdpAssert(tdpPeekAtLastError());

  tdpAssert(tdpDeviceSynchronize());
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

  kernel_ctxt_free(ctxt);
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

__global__ void cahn_stats_kahan_sum_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    map_t * map, phi_stats_t * stats) {
  int kindex;
  int tid;
  int kiterations;
  __shared__ kahan_t phib[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(phi);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;

  phib[tid].sum = 0.0;
  phib[tid].cs  = 0.0;

  stats->sum1 = kahan_zero();

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = cs_index(phi->cs, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double phi0 = phi->data[addr_rank1(phi->nsites, 1, index, 0)];
      kahan_add(&phib[tid], phi0);
    }
  }

  __syncthreads();

  if (tid == 0) {
    kahan_t sum = kahan_zero();

    for (int it = 0; it < blockDim.x; it++) {
      kahan_add(&sum, phib[it].cs);
      kahan_add(&sum, phib[it].sum);
    }

    /* Final result */
    kahan_atomic_add(&stats->sum1, sum.cs);
    kahan_atomic_add(&stats->sum1, sum.sum);
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

__global__ void cahn_stats_klein_sum_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    field_t * csum, map_t * map,
					    phi_stats_t * stats) {
  int kindex;
  int tid;
  __shared__ int kiterations;
  __shared__ klein_t phib[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(phi);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;

  phib[tid].sum = 0.0;
  phib[tid].cs  = 0.0;
  phib[tid].ccs = 0.0;

  stats->sum2 = klein_zero();
  
  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;
    
    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = cs_index(phi->cs, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double phi0 = phi->data[addr_rank1(phi->nsites, 1, index, 0)];
      double cmp  = csum->data[addr_rank1(csum->nsites, 1, index, 0)];

      klein_add(&phib[tid], cmp);
      klein_add(&phib[tid], phi0);
    }
  }

  __syncthreads();
  
  if (tid == 0) {
    klein_t sum = klein_zero();

    for (int it = 0; it < blockDim.x; it++) {
      klein_add(&sum, phib[it].cs);
      klein_add(&sum, phib[it].ccs);
      klein_add(&sum, phib[it].sum);
    }

    /* Final result */
    klein_atomic_add(&stats->sum2, sum.cs);
    klein_atomic_add(&stats->sum2, sum.ccs);
    klein_atomic_add(&stats->sum2, sum.sum);
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

__global__ void cahn_stats_var_kernel(kernel_ctxt_t * ktx, field_t * phi,
				      map_t * map, phi_stats_t * stats) {
  int kindex;
  int tid;
  __shared__ int kiterations;
  __shared__ double bvar[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(phi);
  assert(map);
  assert(stats);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;
  bvar[tid] = 0.0;

  stats->var = 0.0;

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = cs_index(phi->cs, ic, jc, kc);
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

__global__ void cahn_stats_min_kernel(kernel_ctxt_t * ktx, field_t * phi,
				      map_t * map, phi_stats_t * stats) {
  int kindex;
  int tid;
  __shared__ int kiterations;
  __shared__ double bmin[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double bmax[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(phi);
  assert(map);
  assert(stats);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;
  bmin[tid] = +FLT_MAX;
  bmax[tid] = -FLT_MAX;

  stats->min = +FLT_MAX;
  stats->max = -FLT_MAX;

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = cs_index(phi->cs, ic, jc, kc);
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
