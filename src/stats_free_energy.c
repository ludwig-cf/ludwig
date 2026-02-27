/****************************************************************************
 *
 *  stats_free_energy.c
 *
 *  Statistics for free energy density.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2026 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "control.h"
#include "util.h"
#include "stats_free_energy.h"

int fe_stats_fluid_total(fe_t * fe, map_t * map, double sum[3]);
__global__ void fe_stats_fluid_total_kernel(kernel_3d_t k3d, fe_t * fe,
					    map_t * map, double * sum);

/****************************************************************************
 *
 *  stats_free_energy_density
 *
 *  Tots up the free energy density. The loop here totals the fluid,
 *  and there is an additional calculation for different types of
 *  solid surface.
 *
 *  The mechanism to compute surface free energy contributions requires
 *  much reworking and generalisation; it only really covers LC at present.
 *
 ****************************************************************************/

int stats_free_energy_density(pe_t * pe, cs_t * cs, wall_t * wall, fe_t * fe,
			      map_t * map,
			      colloids_info_t * cinfo) {

#define NSTAT 5

  int ntstep;
  int nlocal[3];
  int ncolloid;

  double fe_local[NSTAT];
  double fe_total[NSTAT];
  double rv;
  double ltot[3];
  physics_t * phys = NULL;
  MPI_Comm comm;

  assert(pe);
  assert(cs);
  assert(map);

  if (fe == NULL) return 0;

  pe_mpi_comm(pe, &comm);

  cs_ltot(cs, ltot);
  cs_nlocal(cs, nlocal);
  colloids_info_ntotal(cinfo, &ncolloid);

  fe_local[0] = 0.0; /* Total free energy (fluid all sites) */
  fe_local[1] = 0.0; /* Fluid only free energy */
  fe_local[2] = 0.0; /* Volume of fluid */
  fe_local[3] = 0.0; /* surface free energy */
  fe_local[4] = 0.0; /* other wall free energy (walls only) */

  if (0 == (fe->id == FE_ELECTRO || fe->id == FE_ELECTRO_SYMMETRIC)) {
    fe_stats_fluid_total(fe, map, fe_local);
  }
  else {
    /* We must fall back to the host serial version at the moment
     * as there is no device implementation for the electrokinetics */
    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {

	  double fed = 0.0;
	  int status = MAP_BOUNDARY;
	  int index = cs_index(cs, ic, jc, kc);
	  map_status(map, index, &status);

	  fe->func->fed(fe, index, &fed);
	  fe_local[0] += fed;

	  if (status == MAP_FLUID) {
	    fe_local[1] += fed;
	    fe_local[2] += 1.0;
	  }
	}
      }
    }
  }

  /* A robust mechanism is required to get the surface free energy */

  physics_ref(&phys);
  ntstep = physics_control_timestep(phys);

  if (wall_present(wall)) {

    MPI_Reduce(fe_local, fe_total, NSTAT, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_s1 fs_s2 \n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	    ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	    fe_total[3], fe_total[4]);
  }
  else if (ncolloid > 0) {

    MPI_Reduce(fe_local, fe_total, NSTAT, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_s a f_s/a\n");

    if (fe_total[4] > 0.0) {
      /* Area > 0 means the free energy is available */
      pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	      ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3], fe_total[4], fe_total[3]/fe_total[4]);
    }
    else {
      pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e\n",
	      ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3]);
    }
  }
  else {
    MPI_Reduce(fe_local, fe_total, 3, MPI_DOUBLE, MPI_SUM, 0, comm);
    rv = 1.0/(ltot[X]*ltot[Y]*ltot[Z]);

    pe_info(pe, "\nFree energy density - timestep total fluid\n");
    pe_info(pe, "[fed] %14d %17.10e %17.10e\n", ntstep, rv*fe_total[0],
	    fe_total[1]/fe_total[2]);
  }

#undef NSTAT

  return 0;
}

/*****************************************************************************
 *
 *  fe_stats_fluid_total
 *
 *  The components are: fe total, fluid volume
 *
 *****************************************************************************/

int fe_stats_fluid_total(fe_t * fe, map_t * map, double sum[3]) {

  const int nsz = 3*sizeof(double);
  int nlocal[3] = {0};
  double * dsum = NULL;
  fe_t * target = NULL;

  assert(fe);
  assert(map);

  cs_nlocal(map->cs, nlocal);

  assert(fe->func->target);
  fe->func->target(fe, &target);
  assert(target);

  tdpAssert( tdpMalloc((void ** ) &dsum, nsz) );
  tdpAssert( tdpMemcpy(dsum, sum, nsz, tdpMemcpyHostToDevice) );

  {
    /* Kernel */
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(map->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);
    tdpLaunchKernel(fe_stats_fluid_total_kernel, nblk, ntpb, 0, 0,
                    k3d, target, map->target, dsum);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpStreamSynchronize(0) );
  }

  tdpAssert( tdpMemcpy(sum, dsum, nsz, tdpMemcpyDeviceToHost) );
  tdpAssert( tdpFree(dsum) );

  return 0;
}

/*****************************************************************************
 *
 *  fe_stats_fluid_total_kernel
 *
 *****************************************************************************/

__global__ void fe_stats_fluid_total_kernel(kernel_3d_t k3d, fe_t * fe,
					    map_t * map, double * sum) {
  int kindex = 0;
  int tid = threadIdx.x;

  __shared__ double tall[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tfed[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tvol[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  tall[TARGET_PAD*tid] = 0.0;
  tfed[TARGET_PAD*tid] = 0.0;
  tvol[TARGET_PAD*tid] = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index  = kernel_3d_cs_index(&k3d, ic, jc, kc);
    int status = MAP_BOUNDARY;
    double fed = 0.0;

    map_status(map, index, &status);

    fe->func->fed(fe, index, &fed);
    tall[TARGET_PAD*tid] += fed;

    if (status == MAP_FLUID) {
      tfed[TARGET_PAD*tid] += fed;
      tvol[TARGET_PAD*tid] += 1.0;
    }
  }

  __syncthreads();

  if (tid == 0) {
    /* Accumulate block totals */
    double ball = 0.0;
    double bfed = 0.0;
    double bvol = 0.0;

    for (int it = 0; it < blockDim.x; it++) {
      ball += tall[TARGET_PAD*it];
      bfed += tfed[TARGET_PAD*it];
      bvol += tvol[TARGET_PAD*it];
    }

    atomicAdd(sum + 0, ball);
    atomicAdd(sum + 1, bfed);
    atomicAdd(sum + 2, bvol);
  }

  return;
}
