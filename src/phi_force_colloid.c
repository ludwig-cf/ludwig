/*****************************************************************************
 *
 *  phi_force_colloid.c
 *
 *  TODO: there are actually three routines in here:
 *  fluid/colloids/wall
 *  All use divergence of the stress pth. The file should
 *  be renamed.
 *
 *
 *  The case of force from the thermodynamic sector on both fluid and
 *  colloid via the divergence of the chemical stress.
 *
 *  In the absence of solid, the force on the fluid is related
 *  to the divergence of the chemical stress
 *
 *  F_a = - d_b P_ab
 *
 *  Note that the stress is potentially antisymmetric, so this
 *  really is F_a = -d_b P_ab --- not F_a = -d_b P_ba.
 *
 *  The divergence is discretised as, e.g., in the x-direction,
 *  the difference between the interfacial values
 *
 *  d_x P_ab ~= P_ab(x+1/2) - P_ab(x-1/2)
 *
 *  and the interfacial values are based on linear interpolation
 *  P_ab(x+1/2) = 0.5 (P_ab(x) + P_ab(x+1))
 *
 *  etc (and likewise in the other directions).
 *
 *  At an interface, we are not able to interpolate, and we just
 *  use the value of P_ab from the fluid side.
 *
 *  The stress must be integrated over the colloid surface and the
 *  result added to the net force on a given colloid.
 *
 *  The procedure ensures total momentum is conserved, ie., that
 *  leaving the fluid enters the colloid and vice versa.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk) provided device implementations.
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "wall.h"
#include "hydro_s.h" 
#include "colloids_s.h"
#include "map_s.h"
#include "pth_s.h"
#include "phi_force_colloid.h"
#include "timer.h"

int pth_force_driver(pth_t * pth, colloids_info_t * cinfo,
		     hydro_t * hydro, map_t * map, wall_t * wall);

__global__ void pth_force_map_kernel(kernel_ctxt_t * ktx, pth_t * pth,
				     hydro_t * hydro, map_t * map);
__global__ void pth_force_wall_kernel(kernel_ctxt_t * ktx, pth_t * pth,
				      map_t * map, wall_t * wall,
				      double fw[3]);
__global__ void pth_force_fluid_kernel_v(kernel_ctxt_t * ktx, pth_t * pth,
					 hydro_t * hydro);

/*****************************************************************************
 *
 *  pth_force_colloid
 *
 *  If no colloids, and no hydrodynamics, no action is required.
 *
 *****************************************************************************/

__host__ int pth_force_colloid(pth_t * pth, fe_t * fe, colloids_info_t * cinfo,
			       hydro_t * hydro, map_t * map, wall_t * wall) {

  int ncolloid;

  assert(pth);

  colloids_info_ntotal(cinfo, &ncolloid);

  if (hydro == NULL && ncolloid == 0) return 0;

  if (pth->method == PTH_METHOD_DIVERGENCE) {
    pth_stress_compute(pth, fe);
    pth_force_driver(pth, cinfo, hydro, map, wall);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pth_force_driver
 *
 *  Kernel driver. Allows fluid, colloids, walls.
 *
 *  TODO: hydro == NULL case for relaxational dynamics?
 *  TODO: if no wall, wall kernel not required!
 *  TODO: Fix up a kernel for the colloids.
 *
 *****************************************************************************/

static __device__ double fs[3];

__host__ int pth_force_driver(pth_t * pth, colloids_info_t * cinfo,
			      hydro_t * hydro, map_t * map, wall_t * wall) {
  int ia;
  int nlocal[3];
  dim3 nblk, ntpb;
  wall_t * wallt = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  /* Net momentum balance for wall */
  double * fwd = NULL;
  double fw[3] = {0.0, 0.0, 0.0};

  assert(pth);
  assert(cinfo);
  assert(hydro);
  assert(map);

  cs_nlocal(pth->cs, nlocal);
  wall_target(wall, &wallt);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(pth->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpGetSymbolAddress((void **) &fwd, tdpSymbol(fs));
  tdpMemcpy(fwd, fw, 3*sizeof(double), tdpMemcpyHostToDevice);

  TIMER_start(TIMER_PHI_FORCE_CALC);

  tdpLaunchKernel(pth_force_map_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, pth->target, hydro->target, map->target);

  tdpLaunchKernel(pth_force_wall_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, pth->target, map->target, wallt, fwd);
  tdpDeviceSynchronize();
 
  kernel_ctxt_free(ctxt);

  tdpMemcpy(fw, fwd, 3*sizeof(double), tdpMemcpyDeviceToHost);
  wall_momentum_add(wall, fw);

  /* A separate kernel is requred to allow reduction of the
   * force on each particle. A truly parallel version is
   pending... */

  TIMER_start(TIMER_FREE3);

  /* COLLOID "KERNEL" */

  /* Get stress back! */
  pth_memcpy(pth, tdpMemcpyDeviceToHost);


  colloid_t * pc;
  colloid_link_t * p_link;

  /* All colloids, including halo */
  colloids_info_all_head(cinfo, &pc);
 
  for ( ; pc; pc = pc->nextall) {

    p_link = pc->lnk;

    for (; p_link; p_link = p_link->next) {

      if (p_link->status == LINK_FLUID) {
	int id, p;
	int cmod;

	p = p_link->p;
	cmod = cv[p][X]*cv[p][X] + cv[p][Y]*cv[p][Y] + cv[p][Z]*cv[p][Z];

	if (cmod != 1) continue;
	id = -1;
	if (cv[p][X]) id = X;
	if (cv[p][Y]) id = Y;
	if (cv[p][Z]) id = Z;

	for (ia = 0; ia < 3; ia++) {
	  pc->force[ia] += 1.0*cv[p][id]
	    *pth->str[addr_rank2(pth->nsites, 3, 3, p_link->i, ia, id)];
	}
      }
    }
  }

  TIMER_stop(TIMER_FREE3);
  TIMER_stop(TIMER_PHI_FORCE_CALC);

  return 0;
}

/*****************************************************************************
 *
 *  pth_force_fuild_wall_driver
 *
 *  Kernel driver. Fluid in presence of walls.
 *
 *****************************************************************************/

__host__ int pth_force_fluid_wall_driver(pth_t * pth, hydro_t * hydro,
					 map_t * map, wall_t * wall) {
  int nlocal[3];
  dim3 nblk, ntpb;
  wall_t * wallt = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  /* Net momentum balance for wall */
  double * fwd = NULL;
  double fw[3] = {0.0, 0.0, 0.0};

  assert(pth);
  assert(hydro);
  assert(map);

  cs_nlocal(pth->cs, nlocal);
  wall_target(wall, &wallt);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(pth->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpGetSymbolAddress((void **) &fwd, tdpSymbol(fs));
  tdpMemcpy(fwd, fw, 3*sizeof(double), tdpMemcpyHostToDevice);

  tdpLaunchKernel(pth_force_map_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, pth->target, hydro->target, map->target);

  tdpLaunchKernel(pth_force_wall_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, pth->target, map->target, wallt, fwd);
  tdpDeviceSynchronize();
 
  kernel_ctxt_free(ctxt);

  tdpMemcpy(fw, fwd, 3*sizeof(double), tdpMemcpyDeviceToHost);
  wall_momentum_add(wall, fw);

  return 0;
}

/*****************************************************************************
 *
 *  pth_force_fluid_driver
 *
 *  Kernel driver. Fluid only. Nothing else.
 *
 *****************************************************************************/

__host__ int pth_force_fluid_driver(pth_t * pth,  hydro_t * hydro) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  
  assert(pth);
  assert(hydro);

  cs_nlocal(pth->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(pth->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(TIMER_PHI_FORCE_CALC);

  tdpLaunchKernel(pth_force_fluid_kernel_v, nblk, ntpb, 0, 0,
		  ctxt->target, pth->target, hydro->target);
  tdpDeviceSynchronize();

  TIMER_stop(TIMER_PHI_FORCE_CALC);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  pth_force_fluid_kernel_v
 *
 *  Compute force at each lattice site F_a = d_b Pth_ab.
 *
 *  Fluid only present. Vectorised version.
 *
 *  TODO: check residual device constants "nsites".
 *
 *****************************************************************************/

__global__ void pth_force_fluid_kernel_v(kernel_ctxt_t * ktx, pth_t * pth,
					 hydro_t * hydro) {
  int kindex;
  int kiterations;

  assert(ktx);
  assert(pth);
  assert(hydro);

  kiterations = kernel_vector_iterations(ktx);

  targetdp_simt_for(kindex, kiterations, NSIMDVL) {

    int iv;
    int ia, ib;
    int index;                   /* first index in vector block */
    int ic[NSIMDVL];             /* ic for this iteration */
    int jc[NSIMDVL];             /* jc for this iteration */
    int kc[NSIMDVL];             /* kc ditto */
    int pm[NSIMDVL];             /* ordinate +/- 1 */
    int maskv[NSIMDVL];          /* = 0 if not kernel site, 1 otherwise */
    int index1[NSIMDVL];
    double pth0[3][3][NSIMDVL];
    double pth1[3][3][NSIMDVL];
    double force[3][NSIMDVL];


    index = kernel_baseindex(ktx, kindex);
    kernel_coords_v(ktx, kindex, ic, jc, kc);

    kernel_mask_v(ktx, ic, jc, kc, maskv);

    /* Compute pth at current point */
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	targetdp_simd_for(iv, NSIMDVL) {
	  pth0[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index+iv,ia,ib)];
	}
      }
    }

    /* Compute differences */

    targetdp_simd_for(iv, NSIMDVL) pm[iv] = ic[iv] + maskv[iv];
    kernel_coords_index_v(ktx, pm, jc, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	targetdp_simd_for(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      targetdp_simd_for(iv, NSIMDVL) {
	force[ia][iv] = -0.5*(pth1[ia][X][iv] + pth0[ia][X][iv]);
      }
    }


    targetdp_simd_for(iv, NSIMDVL) pm[iv] = ic[iv] - maskv[iv];
    kernel_coords_index_v(ktx, pm, jc, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	targetdp_simd_for(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      targetdp_simd_for(iv, NSIMDVL) {
	force[ia][iv] += 0.5*(pth1[ia][X][iv] + pth0[ia][X][iv]);
      }
    }

    targetdp_simd_for(iv, NSIMDVL) pm[iv] = jc[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ic, pm, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	targetdp_simd_for(iv, NSIMDVL) { 
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      targetdp_simd_for(iv, NSIMDVL) {
	force[ia][iv] -= 0.5*(pth1[ia][Y][iv] + pth0[ia][Y][iv]);
      }
    }

    targetdp_simd_for(iv, NSIMDVL) pm[iv] = jc[iv] - maskv[iv];
    kernel_coords_index_v(ktx, ic, pm, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	targetdp_simd_for(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      targetdp_simd_for(iv, NSIMDVL) {
	force[ia][iv] += 0.5*(pth1[ia][Y][iv] + pth0[ia][Y][iv]);
      }
    }

    targetdp_simd_for(iv, NSIMDVL) pm[iv] = kc[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ic, jc, pm, index1);

    for (ia = 0; ia < 3; ia++){
      for (ib = 0; ib < 3; ib++){
	targetdp_simd_for(iv, NSIMDVL) { 
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      targetdp_simd_for(iv, NSIMDVL) {
	force[ia][iv] -= 0.5*(pth1[ia][Z][iv] + pth0[ia][Z][iv]);
      }
    }

    targetdp_simd_for(iv, NSIMDVL) pm[iv] = kc[iv] - maskv[iv];
    kernel_coords_index_v(ktx, ic, jc, pm, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	targetdp_simd_for(iv, NSIMDVL) { 
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }
    
    for (ia = 0; ia < 3; ia++) {
      targetdp_simd_for(iv, NSIMDVL) {
	force[ia][iv] += 0.5*(pth1[ia][Z][iv] + pth0[ia][Z][iv]);
      }
    }

    /* Store the force on lattice */

    for (ia = 0; ia < 3; ia++) { 
      targetdp_simd_for(iv, NSIMDVL) { 
	hydro->f[addr_rank1(hydro->nsite,NHDIM,index+iv,ia)]
	  += force[ia][iv]*maskv[iv];
      }
    }
    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  pth_force_map_kernel
 *
 *  This computes the force on the fluid, but does not concern solids.
 *
 *****************************************************************************/

__global__ void pth_force_map_kernel(kernel_ctxt_t * ktx, pth_t * pth,
				     hydro_t * hydro, map_t * map) {

  int kindex;
  int kiterations;

  assert(ktx);
  assert(pth);
  assert(hydro);
  assert(map);

  kiterations = kernel_iterations(ktx);

  targetdp_simt_for(kindex, kiterations, 1) {

    int ic, jc, kc;
    int ia, ib;
    int index, index1;

    double pth0[3][3];
    double pth1[3][3];
    double force[3];

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

    if (map->status[index] == MAP_FLUID) {

      /* Compute pth at current point */

      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  pth0[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)];
	}
      }
      
      /* Compute differences */
      
      index1 = kernel_coords_index(ktx, ic+1, jc, kc);
      
      if (map->status[index1] != MAP_FLUID) {
	/* Compute the fluxes at solid/fluid boundary */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -pth0[ia][X];
	}
      }
      else {
	/* This flux is fluid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	}
      }
      
      index1 = kernel_coords_index(ktx, ic-1, jc, kc);
      
      if (map->status[index1] != MAP_FLUID) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][X];
	}
      }
      else {
	/* Fluid - fluid */
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	}
      }
      
      index1 = kernel_coords_index(ktx, ic, jc+1, kc);
      
      if (map->status[index1] != MAP_FLUID) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= pth0[ia][Y];
	}
      }
      else {
	/* Fluid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
      }
      
      index1 = kernel_coords_index(ktx, ic, jc-1, kc);
      
      if (map->status[index1] != MAP_FLUID) {
	/* Solid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][Y];
	}
      }
      else {
	/* Fluid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
      }
      
      index1 = kernel_coords_index(ktx, ic, jc, kc+1);
      
      if (map->status[index1] != MAP_FLUID) {
	/* Fluid-solid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= pth0[ia][Z];
	}
      }
      else {
	/* Fluid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}
      }
      
      index1 = kernel_coords_index(ktx, ic, jc, kc-1);
      
      if (map->status[index1] != MAP_FLUID) {
	/* Fluid-solid */
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][Z];
	}
      }
      else {
	/* Fluid-fluid */
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}
      }
      
      /* Store the force on lattice */

      for (ia = 0; ia < 3; ia++) {
	hydro->f[addr_rank1(hydro->nsite, NHDIM, index, ia)] += force[ia];
      }
    }
    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  pth_force_wall_kernel
 *
 *  Tally contributions of the stress divergence transfered to
 *  the wall. This needs to agree with phi_force_kernel().
 *
 *  This is for accounting only; there's no dynamics.
 *
 *  TODO: expose wall->fnet and remove actual/dummy argument fw.
 *
 *****************************************************************************/

__global__ void pth_force_wall_kernel(kernel_ctxt_t * ktx, pth_t * pth,
				      map_t * map, wall_t * wall,
				      double fw[3]) {
  int kindex;
  int kiterations;

  int ic, jc, kc;
  int ia, ib;
  int index, index1;
  int tid;

  double pth0[3][3];
  double fxb, fyb, fzb;

  __shared__ double fx[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fy[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fz[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(pth);
  assert(map);
  assert(wall);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;

  fx[tid] = 0.0;
  fy[tid] = 0.0;
  fz[tid] = 0.0;

  targetdp_simt_for(kindex, kiterations, 1) {

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

    if (map->status[index] == MAP_FLUID) {

      /* Compute pth at current point */

      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  pth0[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)];
	}
      }
      
      /* Contributions to surface stress */
      
      index1 = kernel_coords_index(ktx, ic+1, jc, kc);
      
      if (map->status[index1] == MAP_BOUNDARY) {
	fx[tid] += -pth0[X][X];
	fy[tid] += -pth0[Y][X];
	fz[tid] += -pth0[Z][X];
      }
      
      index1 = kernel_coords_index(ktx, ic-1, jc, kc);
      
      if (map->status[index1] == MAP_BOUNDARY) {
	fx[tid] += pth0[X][X];
	fy[tid] += pth0[Y][X];
	fz[tid] += pth0[Z][X];
      }
      
      index1 = kernel_coords_index(ktx, ic, jc+1, kc);
      
      if (map->status[index1] == MAP_BOUNDARY) {
	fx[tid] += -pth0[X][Y];
	fy[tid] += -pth0[Y][Y];
	fz[tid] += -pth0[Z][Y];
      }
      
      index1 = kernel_coords_index(ktx, ic, jc-1, kc);
      
      if (map->status[index1] == MAP_BOUNDARY) {
	fx[tid] += pth0[X][Y];
	fy[tid] += pth0[Y][Y];
	fz[tid] += pth0[Z][Y];
      }
      
      index1 = kernel_coords_index(ktx, ic, jc, kc+1);
      
      if (map->status[index1] == MAP_BOUNDARY) {
	fx[tid] += -pth0[X][Z];
	fy[tid] += -pth0[Y][Z];
	fz[tid] += -pth0[Z][Z];
      }
      
      index1 = kernel_coords_index(ktx, ic, jc, kc-1);
      
      if (map->status[index1] == MAP_BOUNDARY) {
	fx[tid] += pth0[X][Z];
	fy[tid] += pth0[Y][Z];
	fz[tid] += pth0[Z][Z];
      }      
    }
    /* Next site */
  }

  /* Reduction */

  fxb = atomicBlockAddDouble(fx);
  fyb = atomicBlockAddDouble(fy);
  fzb = atomicBlockAddDouble(fz);

  if (tid == 0) {
    atomicAddDouble(fw+X, -fxb);
    atomicAddDouble(fw+Y, -fyb);
    atomicAddDouble(fw+Z, -fzb);
  }

  return;
}
