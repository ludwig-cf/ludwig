/*****************************************************************************
 *
 *  heat_equation_stats.c
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

#include "heat_equation.h"
#include "field_s.h"
#include "util.h"
#include "util_sum.h"

/* Utility container for global statistics */

typedef struct temperature_stats_s temperature_stats_t;

struct temperature_stats_s {
  kahan_t sum1;      /* Glboal sum: single compensation */
  klein_t sum2;      /* Global sum: doubly compensated. */
  double  sum;       /* Current result for sum */
  double  var;       /* Variance */
  double  min;       /* Minimum */
  double  max;       /* Maximum */
  double  vol;       /* Current fluid volume */
};

__host__ int heq_stats_reduce(heq_t * heq, field_t * obj, map_t * map,
			       temperature_stats_t * stats,  int root, MPI_Comm comm);

__global__ void heq_stats_kahan_sum_kernel(kernel_ctxt_t * ktx, field_t * temperature,
					    map_t * map, temperature_stats_t * stats);
__global__ void heq_stats_klein_sum_kernel(kernel_ctxt_t * ktx, field_t * temperature,
					    field_t * csum, map_t * map,
					    temperature_stats_t * stats);
__global__ void heq_stats_var_kernel(kernel_ctxt_t * ktx, field_t * temperature,
				      map_t * map, temperature_stats_t * stats);
__global__ void heq_stats_min_kernel(kernel_ctxt_t * ktx, field_t * temperature,
				      map_t * map, temperature_stats_t * stats);

/*****************************************************************************
 *
 * heat_equation_time0
 *
 * Compute an initial temperature sum in the same way as any later sum.
 *
 *****************************************************************************/

__host__ int heat_equation_stats_time0(heq_t * heq, field_t * temperature,
				       map_t * map) {

  temperature_stats_t stats = {};
  MPI_Comm comm = MPI_COMM_NULL;

  assert(heq);
  assert(temperature);
  assert(map);

  pe_mpi_comm(heq->pe, &comm);
  heq_stats_reduce(heq, temperature, map, &stats, 0, comm);

  temperature->field_init_sum = stats.sum;
  MPI_Bcast(&temperature->field_init_sum, 1, MPI_DOUBLE, 0, comm);

  return 0;
}

/*****************************************************************************
 *
 *  heat_equation_stats
 *
 *****************************************************************************/

__host__ int heat_equation_stats(heq_t * heq, field_t * temperature, map_t * map) {

  temperature_stats_t stats = {.sum1 = {},
		       .sum2 = {},
		       .sum  = 0.0,
		       .var  = 0.0,
		       .min  = +FLT_MAX,
		       .max =  -FLT_MAX,
		       .vol  = 0.0};
  MPI_Comm comm = MPI_COMM_NULL;

  assert(heq);
  assert(temperature);
  assert(map);

  pe_mpi_comm(heq->pe, &comm);
  heq_stats_reduce(heq, temperature, map, &stats, 0, comm);

  {
    double rvol = 1.0 / stats.vol;

    double fbar = rvol*stats.sum;                /* mean */
    double fvar = rvol*stats.var - fbar*fbar;    /* variance */

    pe_info(heq->pe, "[temperature] %14.7e %14.7e%14.7e %14.7e%14.7e\n",
	    stats.sum, fbar, fvar, stats.min, stats.max);
  }

  return 0;
}

/*****************************************************************************
 *
 *  heq_stats_reduce
 *
 *  Form the local sums are reduce to rank "root".
 *
 *****************************************************************************/

__host__ int heq_stats_reduce(heq_t * heq, field_t * temperature,
			       map_t * map, temperature_stats_t * stats,
			       int root, MPI_Comm comm) {
  temperature_stats_t local = {};
  temperature_stats_t * stats_d = NULL;

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(heq);
  assert(temperature);
  assert(map);
  assert(stats);

  cs_nlocal(heq->cs, nlocal);
  
  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  /* Initialise device values */
  tdpAssert(tdpMalloc((void **) &stats_d, sizeof(temperature_stats_t)));
  tdpAssert(tdpMemcpy(stats_d, stats, sizeof(temperature_stats_t),
		      tdpMemcpyHostToDevice));

  if (heq->info.conserve) {
    tdpLaunchKernel(heq_stats_klein_sum_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, temperature->target, heq->csum->target, map->target,
		    stats_d);
    tdpAssert(tdpPeekAtLastError());
  }
  else {
    tdpLaunchKernel(heq_stats_kahan_sum_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, temperature->target, map->target, stats_d);
    tdpAssert(tdpPeekAtLastError());
  }

  tdpLaunchKernel(heq_stats_var_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, temperature->target, map->target, stats_d);
  tdpAssert(tdpPeekAtLastError());

  tdpLaunchKernel(heq_stats_min_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, temperature->target, map->target, stats_d);
  tdpAssert(tdpPeekAtLastError());

  tdpAssert(tdpDeviceSynchronize());
  tdpAssert(tdpMemcpy(&local, stats_d, sizeof(temperature_stats_t),
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

    if (heq->info.conserve) {
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
 *  heq_stats_kahan_sum_kernel
 *
 *  Sum when no compensation in the time evolution is present.
 *
 *****************************************************************************/

__global__ void heq_stats_kahan_sum_kernel(kernel_ctxt_t * ktx, field_t * temperature,
					    map_t * map, temperature_stats_t * stats) {
  int kindex;
  int tid;
  int kiterations;
  __shared__ kahan_t temperatureb[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(temperature);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;

  temperatureb[tid].sum = 0.0;
  temperatureb[tid].cs  = 0.0;

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = cs_index(temperature->cs, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double temperature0 = temperature->data[addr_rank1(temperature->nsites, 1, index, 0)];
      kahan_add_double(&temperatureb[tid], temperature0);
    }
  }

  __syncthreads();

  if (tid == 0) {
    kahan_t sum = kahan_zero();

    for (int it = 0; it < blockDim.x; it++) {
      kahan_add(&sum, temperatureb[it]);
    }

    /* Final result */

    while (atomicCAS(&(stats->sum1.lock), 0, 1) != 0);
    __threadfence();

    kahan_add(&stats->sum1, sum);

    __threadfence();
    atomicExch(&(stats->sum1.lock), 0);
  }

  return;
}

/*****************************************************************************
 *
 *  heq_stats_klein_sum_kernel
 *
 *  Compensated sum. We do care how the sum is computed
 *  (a conserved quantity).
 *
 *****************************************************************************/

__global__ void heq_stats_klein_sum_kernel(kernel_ctxt_t * ktx, field_t * temperature,
					    field_t * csum, map_t * map,
					    temperature_stats_t * stats) {
  int kindex;
  int tid;
  __shared__ int kiterations;
  __shared__ klein_t temperatureb[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(temperature);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;

  temperatureb[tid].sum = 0.0;
  temperatureb[tid].cs  = 0.0;
  temperatureb[tid].ccs = 0.0;

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;
    
    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = cs_index(temperature->cs, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double temperature0 = temperature->data[addr_rank1(temperature->nsites, 1, index, 0)];
      double cmp  = csum->data[addr_rank1(csum->nsites, 1, index, 0)];

      klein_add_double(&temperatureb[tid], cmp);
      klein_add_double(&temperatureb[tid], temperature0);
    }
  }

  __syncthreads();
  
  if (tid == 0) {
    klein_t sum = klein_zero();

    for (int it = 0; it < blockDim.x; it++) {
      klein_add(&sum, temperatureb[it]);
    }

    /* Final result */

    while (atomicCAS(&(stats->sum2.lock), 0, 1) != 0);
    __threadfence();

    klein_add(&stats->sum2, sum);

    __threadfence();
    atomicExch(&(stats->sum2.lock), 0);
  }

  return;
}

/*****************************************************************************
 *
 *  heq_stats_var_kernel
 *
 *  Accumulate the variance in temperature field (don't care about exact sum).
 *
 *****************************************************************************/

__global__ void heq_stats_var_kernel(kernel_ctxt_t * ktx, field_t * temperature,
				      map_t * map, temperature_stats_t * stats) {
  int kindex;
  int tid;
  __shared__ int kiterations;
  __shared__ double bvar[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(temperature);
  assert(map);
  assert(stats);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;
  bvar[tid] = 0.0;

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = cs_index(temperature->cs, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double temperature0 = temperature->data[addr_rank1(temperature->nsites, 1, index, 0)];

      bvar[tid] += temperature0*temperature0;
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
 *  heq_stats_min_kernel
 *
 *  Local min/max of temperature field.
 *
 *****************************************************************************/

__global__ void heq_stats_min_kernel(kernel_ctxt_t * ktx, field_t * temperature,
				      map_t * map, temperature_stats_t * stats) {
  int kindex;
  int tid;
  __shared__ int kiterations;
  __shared__ double bmin[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double bmax[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(temperature);
  assert(map);
  assert(stats);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;

  bmin[tid] = +FLT_MAX;
  bmax[tid] = -FLT_MAX;

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = cs_index(temperature->cs, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double temperature0 = temperature->data[addr_rank1(temperature->nsites, 1, index, 0)];

      bmin[tid] = dmin(bmin[tid], temperature0);
      bmax[tid] = dmax(bmax[tid], temperature0);
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
