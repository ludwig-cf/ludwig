/*****************************************************************************
 *
 *  fe_ternary_stats.c
 *
 *  Report statistics for ternary model.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "fe_ternary_stats.h"


#define FE_PHI 0
#define FE_PSI 1

__host__ int fe_ternary_bulk(fe_ternary_t * fe, map_t * map, double * feb);
__host__ int fe_ternary_surf(fe_ternary_t * fe, map_t * map, double * fes);

__global__ void fe_ternary_bulk_kernel(kernel_3d_t k3d,
				       fe_ternary_t * fe, map_t * map,
				       double febulk[1]);

__global__ void fe_ternary_surf_kernel(kernel_3d_t k3d,
				       fe_ternary_param_t param,
				       field_t * field, map_t * map,
				       double fes[3]);


/*****************************************************************************
 *
 *  fe_ternary_stats_info
 *
 *  Top level driver for free energy statistics.
 *
 *****************************************************************************/

__host__ int fe_ternary_stats_info(fe_ternary_t * fe, wall_t * wall,
				   map_t * map, int nstep) {

  double fe_local[5];
  double fe_total[5];

  pe_t * pe = NULL;
  MPI_Comm comm;

  assert(fe);
  assert(map);

  pe = fe->pe;
  pe_mpi_comm(pe, &comm);

  fe_local[0] = 0.0; /* Total free energy (fluid all sites) */
  fe_local[1] = 0.0; /* Fluid only free energy */
  fe_local[2] = 0.0; /* Volume of fluid */
  fe_local[3] = 0.0; /* surface free energy */
  fe_local[4] = 0.0; /* other wall free energy (walls only) */

  fe_ternary_bulk(fe, map, fe_local);

  if (wall_present(wall)) {
    double fes_tot = 0.0;

    fe_ternary_surf(fe, map, fe_local + 2);

    MPI_Reduce(fe_local, fe_total, 5, MPI_DOUBLE, MPI_SUM, 0, comm);
    fes_tot = fe_total[2] + fe_total[3] + fe_total[4];

    /* Report is on two lines:
     *        time fes_rho fes_phi fes_psi
     *        time surface fluid   total */

    pe_info(pe, "\nFree energies\n");
    pe_info(pe, "[rho/phi/psi]  %9d %17.10e %17.10e %17.10e\n",
	    nstep, fe_total[2], fe_total[3], fe_total[4]);
    pe_info(pe, "[surf/fl/tot]  %9d %17.10e %17.10e %17.10e\n",
	    nstep, fes_tot, fe_total[0], fe_total[0] + fes_tot);
  }
  else {

    /* Fluid only */

    MPI_Reduce(fe_local, fe_total, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies\n");
    pe_info(pe, "[surf/fl/tot]  %9d %17.10e %17.10e %17.10e\n",
	    nstep, 0.0, fe_total[0], fe_total[0]);
  }

  return 0;
}


/*****************************************************************************
 *
 *  fe_ternary_bulk
 *
 *  Compute (bulk) fluid free energy.
 *
 *****************************************************************************/

__host__ int fe_ternary_bulk(fe_ternary_t * fe, map_t * map, double * feb) {

  int nlocal[3] = {0};

  assert(fe);
  assert(map);

  cs_nlocal(fe->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(fe->cs, lim);

    double * febd = NULL;

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpAssert(tdpMalloc((void **) &febd, sizeof(double)));
    tdpAssert(tdpMemcpy(febd, feb, sizeof(double), tdpMemcpyHostToDevice));

    tdpLaunchKernel(fe_ternary_bulk_kernel, nblk, ntpb, 0, 0,
		    k3d, fe->target, map->target, febd);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    tdpAssert(tdpMemcpy(feb, febd, sizeof(double), tdpMemcpyDeviceToHost));
    tdpAssert(tdpFree(febd));
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_surf
 *
 *  Compute terms in the surface free energy.
 *  Three terms are returned: fes[0] = h rho
 *                            fes[1] = h phi
 *                            fes[2] = h psi
 *
 *  Currently 2d only.
 *
 *****************************************************************************/

__host__ int fe_ternary_surf(fe_ternary_t * fe, map_t * map, double * fes) {

  int nlocal[3];
  fe_ternary_param_t param;

  cs_nlocal(fe->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, 1};
    kernel_3d_t k3d = kernel_3d(fe->cs, lim);

    double * fesd = NULL;

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    param = *fe->param;

    tdpAssert(tdpMalloc((void **) &fesd, 3*sizeof(double)));
    tdpAssert(tdpMemcpy(fesd, fes, 3*sizeof(double), tdpMemcpyHostToDevice));

    tdpLaunchKernel(fe_ternary_surf_kernel, nblk, ntpb, 0, 0,
		    k3d, param, fe->phi->target, map->target, fesd);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    tdpAssert(tdpMemcpy(fes, fesd, 3*sizeof(double), tdpMemcpyDeviceToHost));
    tdpAssert(tdpFree(fesd));
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_bulk_kernel
 *
 *  Accumulate the free energy density at fluid sites.
 *
 *****************************************************************************/

__global__ void fe_ternary_bulk_kernel(kernel_3d_t k3d, fe_ternary_t * fe,
				       map_t * map, double febulk[1]) {
  int kindex = 0;
  int tid;

  double febl;
  __shared__ double fepart[TARGET_MAX_THREADS_PER_BLOCK];

  assert(fe);
  assert(febulk);

  tid = threadIdx.x;
  fepart[tid] = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int status;
    double fed;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      fe_ternary_fed(fe, index, &fed);
      fepart[tid] += fed;
    }
  }

  __syncthreads();

  /* Reduction: block, and inter-block */

  febl = tdpAtomicBlockAddDouble(fepart);
  if (tid == 0) tdpAtomicAddDouble(febulk, febl);
}


/*****************************************************************************
 *
 *  fe_ternary_surf_kernel
 *
 *  Accumulate terms in the surface free energy:
 *    -h_1 (rho + phi - psi)/2
 *    -h_2 (rho - phi - psi)/2
 *    -h_3 psi
 *
 *   or
 *
 *   fes[0] = rho (-h_1 - h_2)/2
 *   fes[1] = phi (-h_1 + h_2)/2
 *   fes[2] = psi (+h_1 + h_2 - 2h_3)/2
 *
 *****************************************************************************/

__global__ void fe_ternary_surf_kernel(kernel_3d_t k3d,
				       fe_ternary_param_t param,
				       field_t * f,
				       map_t * map,
				       double fes[3]) {
  int kindex;
  int tid;
  int bs_cv4[4][2] = {{-1,0}, {0,-1}, {0,1}, {1,0}};

  double fesrhobl;
  double fesphibl;
  double fespsibl;
  __shared__ double fesrho[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fesphi[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fespsi[TARGET_MAX_THREADS_PER_BLOCK];

  assert(f);

  tid = threadIdx.x;
  fesrho[tid] = 0.0;
  fesphi[tid] = 0.0;
  fespsi[tid] = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic1, jc1;
    int index, inext, p;
    int status0, status1;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = 1;

    index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    map_status(map, index, &status0);

    if (status0 == MAP_FLUID) {

      /* Look at each neighbour; count only nearest neighbours */

      for (p = 0; p < 4; p++) {
	ic1 = ic + bs_cv4[p][X];
	jc1 = jc + bs_cv4[p][Y];

	inext = kernel_3d_cs_index(&k3d, ic1, jc1, kc);
	map_status(map, inext, &status1);

        if (status1 == MAP_BOUNDARY) {
	  double rho = 1.0;
	  double phi = f->data[addr_rank1(f->nsites, f->nf, index, FE_PHI)];
	  double psi = f->data[addr_rank1(f->nsites, f->nf, index, FE_PSI)];
	  fesrho[tid] += rho*0.5*(-param.h1 - param.h2);
	  fesphi[tid] += phi*0.5*(-param.h1 + param.h2);
	  fespsi[tid] += psi*0.5*( param.h1 + param.h2 - 2.0*param.h3);
	}
      }
    }
    /* Next site */
  }

  __syncthreads();

  /* Reduction: block, then inter-block */

  fesrhobl = tdpAtomicBlockAddDouble(fesrho);
  fesphibl = tdpAtomicBlockAddDouble(fesphi);
  fespsibl = tdpAtomicBlockAddDouble(fespsi);

  if (tid == 0) {
    tdpAtomicAddDouble(fes + 0, fesrhobl);
    tdpAtomicAddDouble(fes + 1, fesphibl);
    tdpAtomicAddDouble(fes + 2, fespsibl);
  }

  return;
}
