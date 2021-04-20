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
#include <stdlib.h>
#include <math.h>

#include "phi_cahn_hilliard.h"
#include "field_s.h"
#include "map_s.h"
#include "util.h"
#include "util_sum.h"

/* Utility container for stats */

typedef struct phi_stats_s phi_stats_t;

struct phi_stats_s {
  klein_t sum;
  double  var;
  double  min;
  double  max;
  double  vol;
};

__host__ int cahn_stats_reduce(phi_ch_t * pch, field_t * obj, map_t * map,
			       phi_stats_t * stats,  int root, MPI_Comm comm);

__global__ void cahn_stats_sum_kernel(kernel_ctxt_t * ktx, field_t * phi,
				      field_t * csum, map_t * map,
				      phi_stats_t * stats);

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

    double fbar = rvol*klein_sum(&stats.sum);    /* mean */
    double f2   = rvol*stats.var - fbar*fbar;    /* variance */

    pe_info(pch->pe, "[pew] %14.7e %14.7e%14.7e %14.7e%14.7e\n",
	    klein_sum(&stats.sum), fbar, f2, stats.min, stats.max);
  }

  return 0;
}

/*****************************************************************************
 *
 *  cahn_stats_reduce
 *
 *****************************************************************************/

__host__ int cahn_stats_reduce(phi_ch_t * pch, field_t * phi,
			       map_t * map, phi_stats_t * stats,
			       int root, MPI_Comm comm) {
  phi_stats_t * stats_d = NULL;
  phi_stats_t local = {};

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

  tdpLaunchKernel(cahn_stats_sum_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, phi->target, pch->csum->target, map->target,
		  stats_d);
  /* min, max, vol */

  kernel_ctxt_free(ctxt);

  MPI_Reduce(&local.sum, &stats->sum, 1, MPI_DOUBLE, MPI_MIN, root, comm);
  MPI_Reduce(&local.var, &stats->var, 1, MPI_DOUBLE, MPI_MAX, root, comm);
  MPI_Reduce(&local.min, &stats->min, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  MPI_Reduce(&local.max, &stats->max, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  MPI_Reduce(&local.vol, &stats->vol, 1, MPI_DOUBLE, MPI_SUM, root, comm);

  tdpFree(&stats_d);

  return 0;
}

/*****************************************************************************
 *
 *  cahn_stats_sum_kernel
 *
 *  Compenstated sum and variance.
 *
 *****************************************************************************/

__global__ void cahn_stats_sum_kernel(kernel_ctxt_t * ktx, field_t * phi,
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

  stats->sum = klein_zero();
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
      double cmp  = csum->data[addr_rank1(csum->nsites, 1, index, 0)];

      klein_add(&phib[tid], -cmp);
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

    /*
    printf("PARA  %14.7e %14.7e %14.7e\n", klein_sum(&sum), sum.cs, klein_sum(&sum)+1.1219161e-11);
    */

    /* Final result */
    klein_atomic_add(&stats->sum, sum.cs);
    klein_atomic_add(&stats->sum, sum.ccs);
    klein_atomic_add(&stats->sum, sum.sum);

  }

  return;
}


