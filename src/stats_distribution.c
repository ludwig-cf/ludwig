/*****************************************************************************
 *
 *  stats_distribution.c
 *
 *  Various routines to compute statistics associated with the
 *  distribution (that is, the density).
 *
 *  If there is more than one distribution, it is assumed the relevant
 *  statistics are produced in the order parameter sector.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "util.h"
#include "util_sum.h"
#include "stats_distribution.h"

/* Kernel utility container */

typedef struct gm_util_s {
  int8_t cv[27][3];
} gm_util_t;

static __constant__ gm_util_t util_;

__host__ int stats_distribution_serial(lb_t * lb, map_t * map);

__host__ int stats_distribution_float(lb_t * lb, map_t * map);
__global__ void stats_distribution_float_kernel(kernel_3d_t k3d, double rho0,
						lb_t * lb,
						map_t * map,
						double *stat_local);

__host__ int stats_distribution_momentum_serial(lb_t * lb, map_t * map,
						double g[3]);

__host__ int stats_distribution_momentum_kahan_t(lb_t * lb, map_t * map,
						 int root,
						 MPI_Comm comm,
						 double gm[3]);

__global__ void stats_distribution_momentum_kahan_t_kernel(kernel_3d_t k3d,
							   lb_t * lb,
							   map_t * map,
							   kahan_t * gm);

__host__ int stats_distribution_momentum_float_t(lb_t * lb, map_t * map,
						 int root,
						 MPI_Comm comm, double gm[3]);
__global__ void stats_distribution_momentum_float_t_kernel(kernel_3d_t k3d,
							   lb_t * lb,
							   map_t * map,
							   double * sum);

/*****************************************************************************
 *
 *  Utilities
 *
 *****************************************************************************/

__host__ __device__ static inline double double_min(double x, double y) {
  return (x < y) ? x : y;
}

__host__ __device__ static inline double double_max(double x, double y) {
  return (x > y) ? x : y;
}

/*****************************************************************************
 *
 *  stats_distribution_print
 *
 *****************************************************************************/

int stats_distribution_print(lb_t * lb, map_t * map) {

  if (lb->opts.istatdensity == LB_STAT_DENSITY_FLOAT) {
    stats_distribution_float(lb, map);
  }
  else {
    /* Default */
    lb_memcpy(lb, tdpMemcpyDeviceToHost);
    stats_distribution_serial(lb, map);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_distribution_serial
 *
 *  This routine prints some statistics related to the first distribution
 *  (always assumed to be the density).
 *
 *  This is serial and really should be pensioned-off. It is retained to
 *  keep the variance reported in the regrssion tests.
 *
 *****************************************************************************/

int stats_distribution_serial(lb_t * lb, map_t * map) {

  int ic, jc, kc, index;
  int nlocal[3];
  int status;

  double stat_local[5];
  double stat_total[5];
  double rho;
  double rhomean;
  double rhovar;

  MPI_Comm comm;

  assert(lb);
  assert(map);

  cs_nlocal(lb->cs, nlocal);
  pe_mpi_comm(lb->pe, &comm);

  stat_local[0] = 0.0;       /* Volume */
  stat_local[1] = 0.0;       /* total mass (or density) */
  stat_local[2] = 0.0;       /* variance rho^2 */
  stat_local[3] = +DBL_MAX;  /* min local density */
  stat_local[4] = -DBL_MAX;  /* max local density */

  for (ic = 1;  ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(lb->cs, ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	lb_0th_moment(lb, index, LB_RHO, &rho);
	stat_local[0] += 1.0;
	stat_local[1] += rho;
	stat_local[2] += rho*rho;
	stat_local[3] = dmin(rho, stat_local[3]);
	stat_local[4] = dmax(rho, stat_local[4]);
      }
    }
  }

  MPI_Reduce(stat_local, stat_total, 3, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(stat_local + 3, stat_total + 3, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(stat_local + 4, stat_total + 4, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  /* Compute mean density, and the variance, and print. We
   * assume the fluid volume (stat_total[0]) is not zero... */

  /* In a uniform state the variance can be a truncation error
   * below zero, hence fabs(rhovar) */

  rhomean = stat_total[1]/stat_total[0];
  rhovar  = (stat_total[2]/stat_total[0]) - rhomean*rhomean;

  pe_info(lb->pe, "\nScalars - total mean variance min max\n");
  pe_info(lb->pe, "[rho] %14.2f %14.11f %14.7e %14.11f %14.11f\n",
       stat_total[1], rhomean, fabs(rhovar), stat_total[3], stat_total[4]);

  return 0;
}

/*****************************************************************************
 *
 *  stats_distribution_float
 *
 *  A parallel version accumulating the anomaly to reduce round-off.
 *
 *****************************************************************************/

int stats_distribution_float(lb_t * lb, map_t * map) {

  const int nsz        = 5*sizeof(double);
  int nlocal[3]        = {0};

  double rho0          = lb->param->rho0;
  double stat_local[5] = {0};
  double stat_total[5] = {0};

  double * d_stat_local = NULL;
  MPI_Comm comm         = MPI_COMM_NULL;

  assert(lb);
  assert(map);

  pe_mpi_comm(lb->pe, &comm);
  cs_nlocal(lb->cs, nlocal);

  stat_local[0] = 0.0;       /* Volume (could be integer) */
  stat_local[1] = 0.0;       /* total mass (or density) as anomaly */
  stat_local[2] = 0.0;       /* variance rho^2 as anomaly */
  stat_local[3] = +DBL_MAX;  /* min local density */
  stat_local[4] = -DBL_MAX;  /* max local density */

  tdpAssert( tdpMalloc((void ** ) &d_stat_local, nsz) );
  tdpAssert( tdpMemcpy(d_stat_local, stat_local, nsz, tdpMemcpyHostToDevice) );

  {
    /* Kernel */
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(lb->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);
    tdpLaunchKernel(stats_distribution_float_kernel, nblk, ntpb, 0, 0,
		    k3d, rho0, lb->target, map->target, d_stat_local);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpStreamSynchronize(0) );
  }

  tdpAssert( tdpMemcpy(stat_local, d_stat_local, nsz, tdpMemcpyDeviceToHost) );

  MPI_Reduce(stat_local + 0, stat_total + 0, 3, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(stat_local + 3, stat_total + 3, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(stat_local + 4, stat_total + 4, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  /* Compute mean density, and the variance, and print. We
   * assume the fluid volume (stat_total[0]) is not zero... */

  {
    double rhomean = rho0 + stat_total[1]/stat_total[0];
    double rhovar  = stat_total[2]/stat_total[0];

    pe_info(lb->pe, "\nScalars - total mean variance min max\n");
    pe_info(lb->pe, "[rho] %14.2f %14.11f %14.7e %14.11f %14.11f\n",
	    stat_total[0]*rhomean, rhomean, rhovar, stat_total[3], stat_total[4]);
  }

  tdpAssert( tdpFree(d_stat_local) );

  return 0;
}

/*****************************************************************************
 *
 *  stats_distribution_kernel_float
 *
 *  Standard floating point sum.
 *
 *****************************************************************************/

__global__ void stats_distribution_float_kernel(kernel_3d_t k3d, double rho0,
						lb_t * lb,
						map_t * map,
						double * stat_local) {
  int kindex = 0;
  int tid = threadIdx.x;

  __shared__ double tvol[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double trho[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tvar[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tmin[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tmax[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  /* Per thread accumulators */
  tvol[TARGET_PAD*tid] = 0.0;
  trho[TARGET_PAD*tid] = 0.0;
  tvar[TARGET_PAD*tid] = 0.0;
  tmin[TARGET_PAD*tid] = +DBL_MAX;
  tmax[TARGET_PAD*tid] = -DBL_MAX;


  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index  = kernel_3d_cs_index(&k3d, ic, jc, kc);
    int status = MAP_BOUNDARY;

    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double rho = 0.0; /* rho is this site; subtract rho0 for the anomaly */
      lb_0th_moment(lb, index, LB_RHO, &rho);
      tvol[TARGET_PAD*tid] += 1.0;
      trho[TARGET_PAD*tid] += (rho - rho0);
      tvar[TARGET_PAD*tid] += (rho - rho0)*(rho - rho0);
      tmin[TARGET_PAD*tid]  = double_min(tmin[TARGET_PAD*tid], rho);
      tmax[TARGET_PAD*tid]  = double_max(tmax[TARGET_PAD*tid], rho);
    }
  }

  __syncthreads();

  if (tid == 0) {
    /* Accumulate block totals */
    double bvol = 0.0;
    double brho = 0.0;
    double bvar = 0.0;
    double bmin = +DBL_MAX;
    double bmax = -DBL_MIN;

    for (int it = 0; it < blockDim.x; it++) {
      bvol += tvol[TARGET_PAD*it];
      brho += trho[TARGET_PAD*it];
      bvar += tvar[TARGET_PAD*it];
      bmin  = double_min(bmin, tmin[TARGET_PAD*it]);
      bmax  = double_max(bmax, tmax[TARGET_PAD*it]);
    }

    atomicAdd(stat_local + 0, bvol);
    atomicAdd(stat_local + 1, brho);
    atomicAdd(stat_local + 2, bvar);
    atomicMin(stat_local + 3, bmin);
    atomicMax(stat_local + 4, bmax);
  }

  return;
}


/*****************************************************************************
 *
 *  stats_distribution_momentum
 *
 *****************************************************************************/

__host__ int stats_distribution_momentum(lb_t * lb, map_t * map, double g[3]) {

  MPI_Comm comm = MPI_COMM_NULL;

  assert(lb);
  assert(map);
  assert(g);

  /* Reduce to rank 0 in pe comm for output. */

  pe_mpi_comm(lb->pe, &comm);

  if (lb->opts.istatmomentum == LB_STAT_MOMENTUM_FLOAT) {
    stats_distribution_momentum_float_t(lb, map, 0, comm, g);
  }
  else {
    /* Default */
    stats_distribution_momentum_kahan_t(lb, map, 0, comm, g);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_distribution_momentum_serial
 *
 *  Returns the fluid momentum (always distribution 0).
 *  Serial routine retained for reference only.
 *
 *****************************************************************************/

int stats_distribution_momentum_serial(lb_t * lb, map_t * map, double g[3]) {

  int ic, jc, kc, index;
  int nlocal[3];
  int status;

  double g_local[3];
  double g_site[3];
  MPI_Comm comm;

  assert(lb);
  assert(map);
  assert(g);

  pe_mpi_comm(lb->pe, &comm);
  cs_nlocal(lb->cs, nlocal);

  g_local[X] = 0.0;
  g_local[Y] = 0.0;
  g_local[Z] = 0.0;

  for (ic = 1;  ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(lb->cs, ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	lb_1st_moment(lb, index, LB_RHO, g_site);
	g_local[X] += g_site[X];
	g_local[Y] += g_site[Y];
	g_local[Z] += g_site[Z];
      }
    }
  }

  MPI_Reduce(g_local, g, 3, MPI_DOUBLE, MPI_SUM, 0, comm);

  return 0;
}

/*****************************************************************************
 *
 *  stats_distribution_momentum_float_t
 *
 *****************************************************************************/

int stats_distribution_momentum_float_t(lb_t * lb, map_t * map, int root,
					MPI_Comm comm, double gm[3]) {

  assert(lb);
  assert(map);

  const int nsz = 3*sizeof(double);
  int nlocal[3] = {0};

  double sum_local[3] = {0};
  double * d_sum      = NULL;
  gm_util_t util      = {0};

  /* Device memory for stats */

  tdpAssert( tdpMalloc((void **) &d_sum, nsz) );
  tdpAssert( tdpMemcpy(d_sum, sum_local, nsz, tdpMemcpyHostToDevice));

  for (int p = 0; p < lb->model.nvel; p++) {
    util.cv[p][X] = lb->model.cv[p][X];
    util.cv[p][Y] = lb->model.cv[p][Y];
    util.cv[p][Z] = lb->model.cv[p][Z];
  }
  tdpMemcpyToSymbol(tdpSymbol(util_), &util, sizeof(gm_util_t), 0,
		    tdpMemcpyHostToDevice);

  /* Local kernel */

  cs_nlocal(lb->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(lb->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(stats_distribution_momentum_float_t_kernel, nblk, ntpb, 0, 0,
		    k3d, lb->target, map->target, d_sum);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpStreamSynchronize(0) );
  }

  /* Copy back local result */
  tdpAssert( tdpMemcpy(sum_local, d_sum, nsz, tdpMemcpyDeviceToHost) );

  /* Reduction for global result */

  MPI_Reduce(sum_local, gm, 3, MPI_DOUBLE, MPI_SUM, root, comm);

  tdpAssert( tdpFree(d_sum) );

  return 0;
}

/*****************************************************************************
 *
 *  stats_distribution_momentum_float_t_kernel
 *
 *****************************************************************************/

__global__ void stats_distribution_momentum_float_t_kernel(kernel_3d_t k3d,
							   lb_t * lb,
							   map_t * map,
							   double * sum) {
  int kindex = 0;
  int tid = threadIdx.x;

  __shared__ double gx[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double gy[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double gz[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  gx[TARGET_PAD*tid] = 0.0;
  gy[TARGET_PAD*tid] = 0.0;
  gz[TARGET_PAD*tid] = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index  = kernel_3d_cs_index(&k3d, ic, jc, kc);
    int status = MAP_BOUNDARY;

    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      for (int p = 1; p < lb->nvel; p++) {
	double f = lb->f[LB_ADDR(lb->nsite,lb->ndist,lb->nvel,index,LB_RHO,p)];
	gx[TARGET_PAD*tid] += f*util_.cv[p][X];
	gy[TARGET_PAD*tid] += f*util_.cv[p][Y];
	gz[TARGET_PAD*tid] += f*util_.cv[p][Z];
      }
    }
  }

  __syncthreads();

  if (tid == 0) {
    /* Accumulate block totals */
    double bgx = 0.0;
    double bgy = 0.0;
    double bgz = 0.0;

    for (int it = 0; it < blockDim.x; it++) {
      bgx += gx[TARGET_PAD*it];
      bgy += gy[TARGET_PAD*it];
      bgz += gz[TARGET_PAD*it];
    }

    atomicAdd(sum + X, bgx);
    atomicAdd(sum + Y, bgy);
    atomicAdd(sum + Z, bgz);
  }

  return;
}

/*****************************************************************************
 *
 *  stats_distribution_momentum_kahan_t
 *
 *  Return global total momentum gm[3] with compensated sum.
 *  This driver calls the kernel below.
 *
 *****************************************************************************/

int stats_distribution_momentum_kahan_t(lb_t * lb, map_t * map, int root,
					MPI_Comm comm, double gm[3]) {

  assert(lb);
  assert(map);

  int nlocal[3] = {0};
  gm_util_t util = {0};

  /* Device memory for stats */

  kahan_t sum[3] = {0};
  kahan_t * sum_d = NULL;

  tdpAssert(tdpMalloc((void **) &sum_d, 3*sizeof(kahan_t)));
  tdpAssert(tdpMemcpy(sum_d, sum, 3*sizeof(kahan_t), tdpMemcpyHostToDevice));

  for (int p = 0; p < lb->model.nvel; p++) {
    util.cv[p][X] = lb->model.cv[p][X];
    util.cv[p][Y] = lb->model.cv[p][Y];
    util.cv[p][Z] = lb->model.cv[p][Z];
  }
  tdpMemcpyToSymbol(tdpSymbol(util_), &util, sizeof(gm_util_t), 0,
		    tdpMemcpyHostToDevice);

  /* Local kernel */

  cs_nlocal(lb->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(lb->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(stats_distribution_momentum_kahan_t_kernel, nblk, ntpb, 0, 0,
		    k3d, lb->target, map->target, sum_d);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpDeviceSynchronize() );
  }

  /* Copy back local result */
  tdpAssert(tdpMemcpy(sum, sum_d, 3*sizeof(kahan_t), tdpMemcpyDeviceToHost));

  /* Reduction for global result */

  {
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Op op = MPI_OP_NULL;
    kahan_t gmlocal[3] = {0};

    kahan_mpi_datatype(&dt);
    kahan_mpi_op_sum(&op);

    MPI_Reduce(sum, gmlocal, 3, dt, op, root, comm);
    gm[X] = kahan_sum(&gmlocal[X]);
    gm[Y] = kahan_sum(&gmlocal[Y]);
    gm[Z] = kahan_sum(&gmlocal[Z]);

    MPI_Op_free(&op);
    MPI_Type_free(&dt);
  }

  tdpAssert( tdpFree(sum_d) );

  return 0;
}

/*****************************************************************************
 *
 *  stats_distribution_momentum_kernel_kahan_t
 *
 *  Kernel with compensated sum.
 *
 *****************************************************************************/

__global__ void stats_distribution_momentum_kahan_t_kernel(kernel_3d_t k3d,
							   lb_t * lb,
							   map_t * map,
							   kahan_t * gm) {
  assert(lb);
  assert(map);
  assert(gm);

  int kindex;
  int tid;

  __shared__ kahan_t gx[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ kahan_t gy[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ kahan_t gz[TARGET_MAX_THREADS_PER_BLOCK];

  tid = threadIdx.x;

  gx[tid].sum = 0.0;
  gx[tid].cs  = 0.0;
  gy[tid].sum = 0.0;
  gy[tid].cs  = 0.0;
  gz[tid].sum = 0.0;
  gz[tid].cs  = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int status = 0;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      for (int p = 1; p < lb->nvel; p++) {
	double f = lb->f[LB_ADDR(lb->nsite,lb->ndist,lb->nvel,index,LB_RHO,p)];
	double gxf = f*util_.cv[p][X];
	double gyf = f*util_.cv[p][Y];
	double gzf = f*util_.cv[p][Z];
	kahan_add_double(&gx[tid], gxf);
	kahan_add_double(&gy[tid], gyf);
	kahan_add_double(&gz[tid], gzf);
      }
    }
  }

  __syncthreads();

  if (tid == 0) {
    kahan_t sumx = kahan_zero();
    kahan_t sumy = kahan_zero();
    kahan_t sumz = kahan_zero();

    for (int it = 0; it < blockDim.x; it++) {
      kahan_add(&sumx, gx[it]);
      kahan_add(&sumy, gy[it]);
      kahan_add(&sumz, gz[it]);
    }

    /* Final result */

    while (atomicCAS(&(gm[X].lock), 0, 1) != 0)
      ;

    __threadfence();

    kahan_add(&gm[X], sumx);
    kahan_add(&gm[Y], sumy);
    kahan_add(&gm[Z], sumz);

    __threadfence();
    atomicExch(&(gm[X].lock), 0);
  }

  return;
}
