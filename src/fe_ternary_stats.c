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
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "map_s.h"
#include "field_s.h"
#include "fe_ternary_stats.h"


#define FE_PHI 0
#define FE_PSI 1

__host__ int fe_ternary_bulk(fe_ternary_t * fe, map_t * map, double * feb);
__host__ int fe_ternary_surf(fe_ternary_t * fe, map_t * map, double * fes);

__global__ void fe_ternary_bulk_kernel(kernel_ctxt_t * ktx,
				       fe_ternary_t * dphi, map_t * map,
				       double febulk[1]);

__global__ void fe_ternary_surf_kernel(kernel_ctxt_t * ktx,
				       fe_ternary_param_t param,
				       field_t * field, map_t * map,
				       double fes[3]);


/*****************************************************************************
 *
 *  fe_ternary_statis_info
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

    fe_ternary_surf(fe, map, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, 5, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_s1 fs_s2\n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	      nstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3], fe_total[4]);
  }
  else {

    /* Fluid only */

    MPI_Reduce(fe_local, fe_total, 5, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_bulk/v f_grad/v\n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	    nstep, 
	    fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	    fe_total[3]/fe_total[2], fe_total[4]/fe_total[2]);
  }

  return 0;
}


/*****************************************************************************
 *
 *  fe_ternary_fe_bulk
 *
 *  Compute (bulk) fluid free energy.
 *
 *****************************************************************************/

__host__ int fe_ternary_fe_bulk(fe_ternary_t * fe, map_t * map) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  double febulk[1] = {0.0};

  assert(fe);
  assert(map);

  cs_nlocal(fe->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(fe->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(fe_ternary_bulk_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, fe->target, map->target, febulk);
  
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_surf
 *
 *  Compute terms in the surface free energy.
 *
 *  Currently 2d only.
 *
 *****************************************************************************/

__host__ int fe_ternary_surf(fe_ternary_t * fe, map_t * map, double * fes) {

  int nlocal[3];
  dim3 nblk, ntpb;
  fe_ternary_param_t param;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  cs_nlocal(fe->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = 1;

  kernel_ctxt_create(fe->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  param = *fe->param;

  tdpLaunchKernel(fe_ternary_surf_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, param, fe->phi->target, map->target, fes);
  
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_bulk_kernel
 *
 *  Accumulate the free energy density at fluid sites.
 *
 *****************************************************************************/

__global__ void fe_ternary_bulk_kernel(kernel_ctxt_t * ktx, fe_ternary_t * fe,
				       map_t * map, double * febulk) {
  int kindex;
  int kiterations;
  int tid;

  double febl;
  __shared__ double fepart[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(fe);
  assert(febulk);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;
  fepart[tid] = 0.0;

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc, index;
    int status;
    double fed;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = kernel_coords_index(ktx, ic, jc, kc);
    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      fe_ternary_fed(fe, index, &fed);
      fepart[tid] += fed;
    }
  }

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

__global__ void fe_ternary_surf_kernel(kernel_ctxt_t * ktx,
				       fe_ternary_param_t param,
				       field_t * field,
				       map_t * map,
				       double fes[3]) {
  int kindex;
  int kiterations;
  int tid;
  int bs_cv4[4][2] = {{-1,0}, {0,-1}, {0,1}, {1,0}};

  double fesrhobl;
  double fesphibl;
  double fespsibl;
  __shared__ double fesrho[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fesphi[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fespsi[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(dphi);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;
  fesrho[tid] = 0.0;
  fesphi[tid] = 0.0;
  fespsi[tid] = 0.0;

  for_simt_parallel(kindex, kiterations, 1) {

    int nf;
    int ic, jc, kc, ic1, jc1;
    int index, inext, p;
    int status0, status1;

    nf = field->nf;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = 1;

    index = kernel_coords_index(ktx, ic, jc, kc);
    map_status(map, index, &status0);
    
    if (status0 == MAP_FLUID) {

      /* Look at each neighbour; count only nearest neighbours */

      for (p = 0; p < 4; p++) {
	ic1 = ic + bs_cv4[p][X];
	jc1 = jc + bs_cv4[p][Y];

	inext = kernel_coords_index(ktx, ic1, jc1, kc);
	map_status(map, inext, &status1);

        if (status1 == MAP_BOUNDARY) {
	  double rho = 1.0;
	  double phi = field->data[addr_rank1(phi->nsites, nf, index, FE_PHI)];
	  double psi = field->data[addr_rank1(phi->nsites, nf, index, FE_PSI)];
	  fesrho[tid] += rho*0.5*(-param.h1 - param.h2);
	  fesphi[tid] += phi*0.5*(-param.h1 + param.h2);
	  fespsi[tid] += psi*0.5*( param.h1 + param.h2 - 2.0*param.h3);
	}
      }
    }
    /* Next site */
  }

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

