/****************************************************************************
 *
 *  stats_velocity.c
 *
 *  Basic statistics for the velocity field.
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
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "hydro.h"
#include "stats_velocity.h"
#include "util.h"

int stats_hydro_uminmax(hydro_t * hydro, map_t * map);
int stats_hydro_uminmax_driver(hydro_t * hydro, map_t * map, double umin[3],
			       double umax[3]);
__global__ void stats_hydro_uminmax_kernel(kernel_3d_t k3d, hydro_t * hydro,
					   map_t * map, double * umin,
					   double * umax);

int stats_hydro_volume_flux(hydro_t * hydro, map_t * map);
int stats_hydro_volume_flux_driver(hydro_t * hydro, map_t * map,
				   double uflux[3]);
__global__ void stats_hydro_volume_flux_kernel(kernel_3d_t k3d,
					       hydro_t * hydro, map_t * map,
					       double * uflux);

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

/****************************************************************************
 *
 *  stats_vel_default
 *
 *  Always print minmax at moment.
 *
 ****************************************************************************/

stats_vel_t stats_vel_default(void) {

  stats_vel_t stat = {0};

  return stat;
}

/****************************************************************************
 *
 *  stats_velocity_minmax
 *
 ****************************************************************************/

int stats_velocity_minmax(stats_vel_t * stat, hydro_t * hydro, map_t * map) {

  assert(stat);
  assert(hydro);
  assert(map);

  stats_hydro_uminmax(hydro, map);

  if (stat->print_vol_flux) {
    stats_hydro_volume_flux(hydro, map);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_hydro_volume_flux
 *
 *  The volume flux of is of interest for porous media calculations
 *  of permeability. Note that with the body force density f, the
 *  volume flux is the same as the total momentum  plus 0.5*f per
 *  lattice site. So for complex porous media, the total momentum
 *  can actually look quite wrong (e.g., have the opposite sign to
 *  the flow).
 *
 ****************************************************************************/

int stats_hydro_volume_flux(hydro_t * hydro, map_t * map) {

  double usum_local[3] = {0};

  assert(hydro);
  assert(map);

  stats_hydro_volume_flux_driver(hydro, map, usum_local);

  {
    double usum[3] = {0};
    MPI_Comm comm = MPI_COMM_NULL;

    pe_mpi_comm(hydro->pe, &comm);

    MPI_Reduce(usum_local, usum, 3, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(hydro->pe, "[vol flux] %14.7e %14.7e %14.7e\n",
	    usum[X], usum[Y], usum[Z]);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_hydro_volume_flux_driver
 *
 *****************************************************************************/

int stats_hydro_volume_flux_driver(hydro_t * hydro, map_t * map,
				   double uflux[3]) {

  const int nsz = 3*sizeof(double);
  int nlocal[3] = {0};
  double * dflux = NULL;

  assert(hydro);
  assert(map);

  cs_nlocal(map->cs, nlocal);

  tdpAssert( tdpMalloc((void ** ) &dflux, nsz) );
  tdpAssert( tdpMemcpy(dflux, uflux, nsz, tdpMemcpyHostToDevice) );

  {
    /* Kernel */
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(map->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);
    tdpLaunchKernel(stats_hydro_volume_flux_kernel, nblk, ntpb, 0, 0,
                    k3d, hydro->target, map->target, dflux);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpStreamSynchronize(0) );
  }

  tdpAssert( tdpMemcpy(uflux, dflux, nsz, tdpMemcpyDeviceToHost) );
  tdpAssert( tdpFree(dflux) );

  return 0;
}

/*****************************************************************************
 *
 *  stats_hydro_volume_flux_kernel
 *
 *****************************************************************************/

__global__ void stats_hydro_volume_flux_kernel(kernel_3d_t k3d,
					       hydro_t * hydro,
					       map_t * map,
					       double * uflux) {
  int kindex = 0;
  int tid = threadIdx.x;

  __shared__ double tx[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double ty[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tz[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  tx[TARGET_PAD*tid] = 0.0;
  ty[TARGET_PAD*tid] = 0.0;
  tz[TARGET_PAD*tid] = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index  = kernel_3d_cs_index(&k3d, ic, jc, kc);
    int status = MAP_BOUNDARY;

    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double u[3] = {0};
      hydro_u(hydro, index, u);
      tx[TARGET_PAD*tid] += u[X];
      ty[TARGET_PAD*tid] += u[Y];
      tz[TARGET_PAD*tid] += u[Z];
    }
  }

  __syncthreads();

  if (tid == 0) {
    /* Accumulate block totals */
    double bflux[3] = {0};

    for (int it = 0; it < blockDim.x; it++) {
      bflux[X] += tx[TARGET_PAD*it];
      bflux[Y] += ty[TARGET_PAD*it];
      bflux[Z] += tz[TARGET_PAD*it];
    }

    atomicAdd(uflux + X, bflux[X]);
    atomicAdd(uflux + Y, bflux[Y]);
    atomicAdd(uflux + Z, bflux[Z]);
  }

  return;
}

/*****************************************************************************
 *
 *  stats_hydro_uminmax
 *
 *****************************************************************************/

int stats_hydro_uminmax(hydro_t * hydro, map_t * map) {

  double umin[3]       = {0}; /* x,y,z components */
  double umax[3]       = {0};
  double umin_local[3] = {+DBL_MAX, +DBL_MAX, +DBL_MAX};
  double umax_local[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};

  MPI_Comm comm = MPI_COMM_NULL;

  assert(hydro);
  assert(map);

  pe_mpi_comm(hydro->pe, &comm);

  stats_hydro_uminmax_driver(hydro, map, umin_local, umax_local);

  MPI_Reduce(umin_local, umin, 3, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(umax_local, umax, 3, MPI_DOUBLE, MPI_MAX, 0, comm);

  pe_info(hydro->pe, "\n");
  pe_info(hydro->pe, "Velocity - x y z\n");
  pe_info(hydro->pe, "[minimum ] %14.7e %14.7e %14.7e\n",
	  umin[X], umin[Y], umin[Z]);
  pe_info(hydro->pe, "[maximum ] %14.7e %14.7e %14.7e\n",
	  umax[X], umax[Y], umax[Z]);

  return 0;
}

/*****************************************************************************
 *
 *  stats_hydro_uminmax_driver
 *
 *****************************************************************************/

int stats_hydro_uminmax_driver(hydro_t * hydro, map_t * map, double umin[3],
			       double umax[3]) {

  const int nsz = 3*sizeof(double);
  int nlocal[3] = {0};
  double * dmin = NULL;
  double * dmax = NULL;

  assert(hydro);
  assert(map);

  cs_nlocal(map->cs, nlocal);

  tdpAssert( tdpMalloc((void ** ) &dmin, nsz) );
  tdpAssert( tdpMalloc((void ** ) &dmax, nsz) );
  tdpAssert( tdpMemcpy(dmin, umin, nsz, tdpMemcpyHostToDevice) );
  tdpAssert( tdpMemcpy(dmax, umax, nsz, tdpMemcpyHostToDevice) );

  {
    /* Kernel */
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(map->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);
    tdpLaunchKernel(stats_hydro_uminmax_kernel, nblk, ntpb, 0, 0,
                    k3d, hydro->target, map->target, dmin, dmax);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpStreamSynchronize(0) );
  }

  tdpAssert( tdpMemcpy(umin, dmin, nsz, tdpMemcpyDeviceToHost) );
  tdpAssert( tdpMemcpy(umax, dmax, nsz, tdpMemcpyDeviceToHost) );
  tdpAssert( tdpFree(dmax) );
  tdpAssert( tdpFree(dmin) );

  return 0;
}

/*****************************************************************************
 *
 *  stats_hydro_uminmax_kernel
 *
 *****************************************************************************/

__global__ void stats_hydro_uminmax_kernel(kernel_3d_t k3d, hydro_t * hydro,
					   map_t * map, double * umin,
					   double * umax) {
  int kindex = 0;
  int tid = threadIdx.x;

  __shared__ double txmin[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double txmax[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tymin[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tymax[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tzmin[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double tzmax[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  txmin[TARGET_PAD*tid] = +DBL_MAX;
  txmax[TARGET_PAD*tid] = -DBL_MAX;
  tymin[TARGET_PAD*tid] = +DBL_MAX;
  tymax[TARGET_PAD*tid] = -DBL_MAX;
  tzmin[TARGET_PAD*tid] = +DBL_MAX;
  tzmax[TARGET_PAD*tid] = -DBL_MAX;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index  = kernel_3d_cs_index(&k3d, ic, jc, kc);
    int status = MAP_BOUNDARY;

    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double u[3] = {0};
      hydro_u(hydro, index, u);
      txmin[TARGET_PAD*tid] = double_min(u[X], txmin[TARGET_PAD*tid]);
      txmax[TARGET_PAD*tid] = double_max(u[X], txmax[TARGET_PAD*tid]);
      tymin[TARGET_PAD*tid] = double_min(u[Y], tymin[TARGET_PAD*tid]);
      tymax[TARGET_PAD*tid] = double_max(u[Y], tymax[TARGET_PAD*tid]);
      tzmin[TARGET_PAD*tid] = double_min(u[Z], tzmin[TARGET_PAD*tid]);
      tzmax[TARGET_PAD*tid] = double_max(u[Z], tzmax[TARGET_PAD*tid]);
    }
  }

  __syncthreads();

  if (tid == 0) {
    /* Accumulate block totals */
    double bmin[3] = {+DBL_MAX, +DBL_MAX, +DBL_MAX};
    double bmax[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};

    for (int it = 0; it < blockDim.x; it++) {
      bmin[X]  = double_min(bmin[X], txmin[TARGET_PAD*it]);
      bmax[X]  = double_max(bmax[X], txmax[TARGET_PAD*it]);
      bmin[Y]  = double_min(bmin[Y], tymin[TARGET_PAD*it]);
      bmax[Y]  = double_max(bmax[Y], tymax[TARGET_PAD*it]);
      bmin[Z]  = double_min(bmin[Z], tzmin[TARGET_PAD*it]);
      bmax[Z]  = double_max(bmax[Z], tzmax[TARGET_PAD*it]);
    }

    atomicMin(umin + X, bmin[X]);
    atomicMax(umax + X, bmax[X]);
    atomicMin(umin + Y, bmin[Y]);
    atomicMax(umax + Y, bmax[Y]);
    atomicMin(umin + Z, bmin[Z]);
    atomicMax(umax + Z, bmax[Z]);
  }

  return;
}
