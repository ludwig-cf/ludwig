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
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "lb_model_s.h"
#include "map_s.h"
#include "util.h"
#include "util_sum.h"
#include "stats_distribution.h"

__host__ int stats_distribution_momentum_serial(lb_t * lb, map_t * map,
						double g[3]);
__host__ int distribution_stats_momentum(lb_t * lb, map_t * map, int root,
					 MPI_Comm comm, double gm[3]);

__global__ void distribution_gm_kernel(kernel_ctxt_t * ktxt, lb_t * lb,
				       map_t * map, kahan_t * gm);


/*****************************************************************************
 *
 *  stats_distribution_print
 *
 *  This routine prints some statistics related to the first distribution
 *  (always assumed to be the density).
 *
 *****************************************************************************/

int stats_distribution_print(lb_t * lb, map_t * map) {

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
  distribution_stats_momentum(lb, map, 0, comm, g);

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
 *  distribution_stats_momentum
 *
 *  Return global total momentum gm[3] with compenstated sum.
 *  This driver calls the kernel below.
 *
 *****************************************************************************/

__host__ int distribution_stats_momentum(lb_t * lb, map_t * map, int root,
					 MPI_Comm comm, double gm[3]) {

  assert(lb);
  assert(map);

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  /* Device memory for stats */

  kahan_t sum[3] = {};
  kahan_t * sum_d = NULL;

  tdpAssert(tdpMalloc((void **) &sum_d, 3*sizeof(kahan_t)));

  /* Local kernel */

  cs_nlocal(lb->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];
  kernel_ctxt_create(lb->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(distribution_gm_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, lb->target, map->target, sum_d);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  /* Copy back local result */
  tdpAssert(tdpMemcpy(sum, sum_d, 3*sizeof(kahan_t), tdpMemcpyDeviceToHost));

  /* Reduction for global result */

  {
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Op op = MPI_OP_NULL;
    kahan_t gmlocal[3] = {};

    kahan_mpi_datatype(&dt);
    kahan_mpi_op_sum(&op);

    MPI_Reduce(sum, gmlocal, 3, dt, op, root, comm);
    gm[X] = kahan_sum(&gmlocal[X]);
    gm[Y] = kahan_sum(&gmlocal[Y]);
    gm[Z] = kahan_sum(&gmlocal[Z]);

    MPI_Op_free(&op);
    MPI_Type_free(&dt);
  }

  kernel_ctxt_free(ctxt);
  tdpFree(sum_d);

  return 0;
}

/*****************************************************************************
 *
 *  distribution_gm_kernel
 *
 *
 *  Kernel with compenstated sum.
 *
 *****************************************************************************/

__global__ void distribution_gm_kernel(kernel_ctxt_t * ktx, lb_t * lb,
				       map_t * map, kahan_t * gm) {

  assert(ktx);
  assert(lb);
  assert(map);
  assert(gm);

  int kindex;
  int tid;
  int kiterations;
  __shared__ kahan_t gx[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ kahan_t gy[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ kahan_t gz[TARGET_MAX_THREADS_PER_BLOCK];

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;

  gx[tid].sum = 0.0;
  gx[tid].cs  = 0.0;
  gy[tid].sum = 0.0;
  gy[tid].cs  = 0.0;
  gz[tid].sum = 0.0;
  gz[tid].cs  = 0.0;

  gm[X] = kahan_zero();
  gm[Y] = kahan_zero();
  gm[Z] = kahan_zero();

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;
    int status = 0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = kernel_coords_index(ktx, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      for (int p = 1; p < NVEL; p++) {
	LB_CV(cv);
	double f = lb->f[LB_ADDR(lb->nsite,lb->ndist,NVEL,index,LB_RHO,p)];
	double gxf = f*cv[p][X];
	double gyf = f*cv[p][Y];
	double gzf = f*cv[p][Z];
	kahan_add(&gx[tid], gxf);
	kahan_add(&gy[tid], gyf);
	kahan_add(&gz[tid], gzf);
      }
    }
  }

  __syncthreads();

  if (tid == 0) {
    kahan_t sumx = kahan_zero();
    kahan_t sumy = kahan_zero();
    kahan_t sumz = kahan_zero();

    for (int it = 0; it < blockDim.x; it++) {
      kahan_add(&sumx, gx[it].cs);
      kahan_add(&sumx, gx[it].sum);
      kahan_add(&sumy, gy[it].cs);
      kahan_add(&sumy, gy[it].sum);
      kahan_add(&sumz, gz[it].cs);
      kahan_add(&sumz, gz[it].sum);
    }

    /* Final result */
    kahan_atomic_add(&gm[X], sumx.cs);
    kahan_atomic_add(&gm[X], sumx.sum);
    kahan_atomic_add(&gm[Y], sumy.cs);
    kahan_atomic_add(&gm[Y], sumy.sum);
    kahan_atomic_add(&gm[Z], sumz.cs);
    kahan_atomic_add(&gm[Z], sumz.sum);
  }

  return;
}
