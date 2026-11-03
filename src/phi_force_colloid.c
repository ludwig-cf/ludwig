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
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk) provided device implementations.
 *
 *  (c) 2010-2026 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "wall.h"
#include "colloids.h"
#include "phi_force_stress.h"
#include "phi_force_colloid.h"
#include "timer.h"
#include "util_math_inline.h"

int pth_force_driver(pth_t * pth, colloids_info_t * cinfo,
		     hydro_t * hydro, map_t * map, wall_t * wall,
		     const lb_t * lb);
int phi_force_colloid_stress_driver(const pth_t * pth, colloids_info_t * info,
				    const lb_t * lb);

__global__ void pth_force_map_kernel(kernel_3d_t k3d, pth_t * pth,
				     hydro_t * hydro, map_t * map);
__global__ void pth_force_wall_kernel(kernel_3d_t k3d, pth_t * pth,
				      map_t * map, wall_t * wall,
				      double fw[3]);

__global__ void pth_force_fluid_kernel_v(kernel_3d_v_t k3v, pth_t * pth,
					 hydro_t * hydro);

/*****************************************************************************
 *
 *  pth_force_colloid
 *
 *  If no colloids, and no hydrodynamics, no action is required.
 *  The LB model is required (only the model, but need device pointer).
 *
 *****************************************************************************/

__host__ int pth_force_colloid(pth_t * pth, fe_t * fe, colloids_info_t * cinfo,
			       hydro_t * hydro, map_t * map, wall_t * wall,
			       const lb_t * lb) {
  int ncolloid;

  assert(pth);

  colloids_info_ntotal(cinfo, &ncolloid);

  if (hydro == NULL && ncolloid == 0) return 0;

  if (pth->method == FE_FORCE_METHOD_STRESS_DIVERGENCE) {
    pth_stress_compute(pth, fe);
    pth_force_driver(pth, cinfo, hydro, map, wall, lb);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pth_force_driver
 *
 *  Kernel driver. Allows fluid, colloids, walls.
 *
 *  Relaxational dynamics (hydro == NULL) are allowed.
 *
 *  TODO: if no wall, wall kernel not required!
 *  TODO: Fix up a kernel for the colloids.
 *
 *****************************************************************************/

__host__ int pth_force_driver(pth_t * pth, colloids_info_t * cinfo,
			      hydro_t * hydro, map_t * map, wall_t * wall,
			      const lb_t * lb) {
  int nlocal[3] = {};
  wall_t * wallt = NULL;

  /* Net momentum balance for wall */
  double * fwd = NULL;
  double fw[3] = {0.0, 0.0, 0.0};

  assert(pth);
  assert(cinfo);
  assert(map);

  /* Relaxational dynamics only are allowed: hydro = NULL */
  if (hydro == NULL) return 0;

  TIMER_start(TIMER_PHI_FORCE_CALC);

  cs_nlocal(pth->cs, nlocal);
  wall_target(wall, &wallt);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(pth->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpAssert( tdpMalloc((void **) &fwd, 3*sizeof(double)) );
    tdpAssert( tdpMemcpy(fwd, fw, 3*sizeof(double), tdpMemcpyHostToDevice) );

    tdpLaunchKernel(pth_force_map_kernel, nblk, ntpb, 0, 0,
		    k3d, pth->target, hydro->target, map->target);

    tdpLaunchKernel(pth_force_wall_kernel, nblk, ntpb, 0, 0,
		    k3d, pth->target, map->target, wallt, fwd);

    tdpAssert( tdpStreamSynchronize(0) );
  }

  tdpAssert( tdpMemcpy(fw, fwd, 3*sizeof(double), tdpMemcpyDeviceToHost) );
  wall_momentum_add(wall, fw);

  tdpAssert( tdpFree(fwd) );

  /* A separate kernel is required to allow reduction of the
   * force on each particle. A truly parallel version is
   pending... */

  /* COLLOID "KERNEL" */
#ifndef USE_LINK_ARRAY
  /* DRIVER */
  phi_force_colloid_stress_driver(pth, cinfo, lb);
#else
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
	cmod = lb->model->cv[p][X]*lb->model->cv[p][X]
	     + lb->model->cv[p][Y]*lb->model->cv[p][Y]
	     + lb->model->cv[p][Z]*lb->model->cv[p][Z];

	if (cmod != 1) continue;
	id = -1;
	if (lb->model->cv[p][X]) id = X;
	if (lb->model->cv[p][Y]) id = Y;
	if (lb->model->cv[p][Z]) id = Z;

	for (int ia = 0; ia < 3; ia++) {
	  pc->force[ia] += 1.0*lb->model->cv[p][id]
	    *pth->str[addr_rank2(pth->nsites, 3, 3, p_link->i, ia, id)];
	}
      }
    }
  }
#endif
  TIMER_stop(TIMER_PHI_FORCE_CALC);

  return 0;
}

/*****************************************************************************
 *
 *  pth_force_fluid_wall_driver
 *
 *  Kernel driver. Fluid in presence of walls.
 *
 *****************************************************************************/

__host__ int pth_force_fluid_wall_driver(pth_t * pth, hydro_t * hydro,
					 map_t * map, wall_t * wall) {
  int nlocal[3] = {0};
  wall_t * wallt = NULL;

  /* Net momentum balance for wall */
  double * fwd = NULL;
  double fw[3] = {0.0, 0.0, 0.0};

  assert(pth);
  assert(hydro);
  assert(map);

  cs_nlocal(pth->cs, nlocal);
  wall_target(wall, &wallt);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(pth->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpAssert( tdpMalloc((void **) &fwd, 3*sizeof(double)) );
    tdpAssert( tdpMemcpy(fwd, fw, 3*sizeof(double), tdpMemcpyHostToDevice) );

    tdpLaunchKernel(pth_force_map_kernel, nblk, ntpb, 0, 0,
		    k3d, pth->target, hydro->target, map->target);

    tdpLaunchKernel(pth_force_wall_kernel, nblk, ntpb, 0, 0,
		    k3d, pth->target, map->target, wallt, fwd);

    tdpAssert( tdpDeviceSynchronize() );
  }

  tdpAssert( tdpMemcpy(fw, fwd, 3*sizeof(double), tdpMemcpyDeviceToHost) );
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

  int nlocal[3] = {0};

  assert(pth);
  assert(hydro);

  cs_nlocal(pth->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_v_t k3v = kernel_3d_v(pth->cs, lim, NSIMDVL);

    kernel_3d_launch_param(k3v.kiterations, &nblk, &ntpb);

    TIMER_start(TIMER_PHI_FORCE_CALC);

    tdpLaunchKernel(pth_force_fluid_kernel_v, nblk, ntpb, 0, 0,
		    k3v, pth->target, hydro->target);
    tdpAssert( tdpDeviceSynchronize() );

    TIMER_stop(TIMER_PHI_FORCE_CALC);
  }

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

__global__ void pth_force_fluid_kernel_v(kernel_3d_v_t k3v, pth_t * pth,
					 hydro_t * hydro) {
  int kindex = 0;

  assert(pth);
  assert(hydro);

  for_simt_parallel(kindex, k3v.kiterations, NSIMDVL) {

    int iv;
    int ic[NSIMDVL];             /* ic for this iteration */
    int jc[NSIMDVL];             /* jc for this iteration */
    int kc[NSIMDVL];             /* kc ditto */
    int pm[NSIMDVL];             /* ordinate +/- 1 */
    int maskv[NSIMDVL];          /* = 0 if not kernel site, 1 otherwise */
    int index1[NSIMDVL];
    double pth0[3][3][NSIMDVL];
    double pth1[3][3][NSIMDVL];
    double force[3][NSIMDVL];


    int index = k3v.kindex0 + kindex;
    kernel_3d_v_coords(&k3v, kindex, ic, jc, kc);

    kernel_3d_v_mask(&k3v, ic, jc, kc, maskv);

    /* Compute pth at current point */
    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	for_simd_v(iv, NSIMDVL) {
	  pth0[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index+iv,ia,ib)];
	}
      }
    }

    /* Compute differences */

    for_simd_v(iv, NSIMDVL) pm[iv] = ic[iv] + maskv[iv];
    kernel_3d_v_cs_index(&k3v, pm, jc, kc, index1);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	for_simd_v(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (int ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	force[ia][iv] = -0.5*(pth1[ia][X][iv] + pth0[ia][X][iv]);
      }
    }


    for_simd_v(iv, NSIMDVL) pm[iv] = ic[iv] - maskv[iv];
    kernel_3d_v_cs_index(&k3v, pm, jc, kc, index1);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	for_simd_v(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (int ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	force[ia][iv] += 0.5*(pth1[ia][X][iv] + pth0[ia][X][iv]);
      }
    }

    for_simd_v(iv, NSIMDVL) pm[iv] = jc[iv] + maskv[iv];
    kernel_3d_v_cs_index(&k3v, ic, pm, kc, index1);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	for_simd_v(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (int ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	force[ia][iv] -= 0.5*(pth1[ia][Y][iv] + pth0[ia][Y][iv]);
      }
    }

    for_simd_v(iv, NSIMDVL) pm[iv] = jc[iv] - maskv[iv];
    kernel_3d_v_cs_index(&k3v, ic, pm, kc, index1);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	for_simd_v(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (int ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	force[ia][iv] += 0.5*(pth1[ia][Y][iv] + pth0[ia][Y][iv]);
      }
    }

    for_simd_v(iv, NSIMDVL) pm[iv] = kc[iv] + maskv[iv];
    kernel_3d_v_cs_index(&k3v, ic, jc, pm, index1);

    for (int ia = 0; ia < 3; ia++){
      for (int ib = 0; ib < 3; ib++){
	for_simd_v(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (int ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	force[ia][iv] -= 0.5*(pth1[ia][Z][iv] + pth0[ia][Z][iv]);
      }
    }

    for_simd_v(iv, NSIMDVL) pm[iv] = kc[iv] - maskv[iv];
    kernel_3d_v_cs_index(&k3v, ic, jc, pm, index1);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	for_simd_v(iv, NSIMDVL) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(pth->nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (int ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	force[ia][iv] += 0.5*(pth1[ia][Z][iv] + pth0[ia][Z][iv]);
      }
    }

    /* Store the force on lattice */

    for (int ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	hydro->force->data[addr_rank1(hydro->nsite, NHDIM, index+iv, ia)]
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

__global__ void pth_force_map_kernel(kernel_3d_t k3d, pth_t * pth,
				     hydro_t * hydro, map_t * map) {
  int kindex = 0;

  assert(pth);
  assert(hydro);
  assert(map);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic, jc, kc;
    int index, index1;

    double pth0[3][3];
    double pth1[3][3];
    double force[3];

    ic = kernel_3d_ic(&k3d, kindex);
    jc = kernel_3d_jc(&k3d, kindex);
    kc = kernel_3d_kc(&k3d, kindex);
    index = kernel_3d_cs_index(&k3d, ic, jc, kc);

    if (map->status[index] == MAP_FLUID) {

      /* Compute pth at current point */

      for (int ia = 0; ia < 3; ia++) {
	for (int ib = 0; ib < 3; ib++) {
	  pth0[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)];
	}
      }

      /* Compute differences */

      index1 = kernel_3d_cs_index(&k3d, ic+1, jc, kc);

      if (map->status[index1] != MAP_FLUID) {
	/* Compute the fluxes at solid/fluid boundary */
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] = -pth0[ia][X];
	}
      }
      else {
	/* This flux is fluid-fluid */
	for (int ia = 0; ia < 3; ia++) {
	  for (int ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	}
      }

      index1 = kernel_3d_cs_index(&k3d, ic-1, jc, kc);

      if (map->status[index1] != MAP_FLUID) {
	/* Solid-fluid */
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][X];
	}
      }
      else {
	/* Fluid - fluid */
	for (int ia = 0; ia < 3; ia++) {
	  for (int ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	}
      }

      index1 = kernel_3d_cs_index(&k3d, ic, jc+1, kc);

      if (map->status[index1] != MAP_FLUID) {
	/* Solid-fluid */
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] -= pth0[ia][Y];
	}
      }
      else {
	/* Fluid-fluid */
	for (int ia = 0; ia < 3; ia++) {
	  for (int ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
      }

      index1 = kernel_3d_cs_index(&k3d, ic, jc-1, kc);

      if (map->status[index1] != MAP_FLUID) {
	/* Solid-fluid */
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][Y];
	}
      }
      else {
	/* Fluid-fluid */
	for (int ia = 0; ia < 3; ia++) {
	  for (int ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
      }

      index1 = kernel_3d_cs_index(&k3d, ic, jc, kc+1);

      if (map->status[index1] != MAP_FLUID) {
	/* Fluid-solid */
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] -= pth0[ia][Z];
	}
      }
      else {
	/* Fluid-fluid */
	for (int ia = 0; ia < 3; ia++) {
	  for (int ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}
      }

      index1 = kernel_3d_cs_index(&k3d, ic, jc, kc-1);

      if (map->status[index1] != MAP_FLUID) {
	/* Fluid-solid */
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] += pth0[ia][Z];
	}
      }
      else {
	/* Fluid-fluid */
	for (int ia = 0; ia < 3; ia++) {
	  for (int ib = 0; ib < 3; ib++) {
	    pth1[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index1,ia,ib)];
	  }
	}
	for (int ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}
      }

      /* Accumulate the force on lattice */

      hydro_f_local_add(hydro, index, force);
    }
    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  pth_force_wall_kernel
 *
 *  Tally contributions of the stress divergence transferred to
 *  the wall. This needs to agree with phi_force_kernel().
 *
 *  This is for accounting only; there's no dynamics.
 *
 *  TODO: expose wall->fnet and remove actual/dummy argument fw.
 *
 *****************************************************************************/

__global__ void pth_force_wall_kernel(kernel_3d_t k3d, pth_t * pth,
				      map_t * map, wall_t * wall,
				      double fw[3]) {
  int kindex = 0;

  int tid = threadIdx.x;
  int pid = TARGET_PAD*tid;

  __shared__ double fx[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fy[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fz[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  assert(pth);
  assert(map);
  assert(wall);

  fx[pid] = 0.0;
  fy[pid] = 0.0;
  fz[pid] = 0.0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);

    if (map->status[index] == MAP_FLUID) {

      int index1 = 0;
      double pth0[3][3] = {0};

      /* Compute pth at current point */

      for (int ia = 0; ia < 3; ia++) {
	for (int ib = 0; ib < 3; ib++) {
	  pth0[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)];
	}
      }

      /* Contributions to surface stress */

      index1 = kernel_3d_cs_index(&k3d, ic+1, jc, kc);

      if (map->status[index1] == MAP_BOUNDARY) {
	fx[pid] += -pth0[X][X];
	fy[pid] += -pth0[Y][X];
	fz[pid] += -pth0[Z][X];
      }

      index1 = kernel_3d_cs_index(&k3d, ic-1, jc, kc);

      if (map->status[index1] == MAP_BOUNDARY) {
	fx[pid] += pth0[X][X];
	fy[pid] += pth0[Y][X];
	fz[pid] += pth0[Z][X];
      }

      index1 = kernel_3d_cs_index(&k3d, ic, jc+1, kc);

      if (map->status[index1] == MAP_BOUNDARY) {
	fx[pid] += -pth0[X][Y];
	fy[pid] += -pth0[Y][Y];
	fz[pid] += -pth0[Z][Y];
      }

      index1 = kernel_3d_cs_index(&k3d, ic, jc-1, kc);

      if (map->status[index1] == MAP_BOUNDARY) {
	fx[pid] += pth0[X][Y];
	fy[pid] += pth0[Y][Y];
	fz[pid] += pth0[Z][Y];
      }

      index1 = kernel_3d_cs_index(&k3d, ic, jc, kc+1);

      if (map->status[index1] == MAP_BOUNDARY) {
	fx[pid] += -pth0[X][Z];
	fy[pid] += -pth0[Y][Z];
	fz[pid] += -pth0[Z][Z];
      }

      index1 = kernel_3d_cs_index(&k3d, ic, jc, kc-1);

      if (map->status[index1] == MAP_BOUNDARY) {
	fx[pid] += pth0[X][Z];
	fy[pid] += pth0[Y][Z];
	fz[pid] += pth0[Z][Z];
      }
    }
    /* Next site */
  }

  __syncthreads();

  /* Reduction */

  if (tid == 0) {

    double fxb = 0.0;
    double fyb = 0.0;
    double fzb = 0.0;

    for (int it = 0; it < blockDim.x; it++) {
      pid = TARGET_PAD*it;
      fxb += fx[pid];
      fyb += fy[pid];
      fzb += fz[pid];
    }

    atomicAdd(fw + X, -fxb);
    atomicAdd(fw + Y, -fyb);
    atomicAdd(fw + Z, -fzb);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_colloid_stress_accumulate
 *
 *  Accumulate contributions to the force on a single colloid from the
 *  normal stress at (unit area) faces. This means links with c^2 = 1.
 *
 *  The stress at the relevant fluid site is used.
 *
 *****************************************************************************/

__host__ __device__ void
phi_force_colloid_stress_accumulate(const pth_t * pth,	const lb_model_t * model, colloid_t * pc) {

  double f[3] = {};

  for (int nl = 0; nl < pc->links->active_links; ++nl) {

    if (pc->links->status[nl] == LINK_FLUID) {

      int i  = pc->links->i[nl];
      int p  = pc->links->p[nl];
      int pp = util_square_modulus_int8(model->cv[p]);

      if (pp == 1) {
	int ib = -1;
	if (model->cv[p][X]) ib = X;
	if (model->cv[p][Y]) ib = Y;
	if (model->cv[p][Z]) ib = Z;

	for (int ia = 0; ia < 3; ia++) {
	  double s = pth->str[addr_rank2(pth->nsites, 3, 3, i, ia, ib)];
	  f[ia] += s*model->cv[p][ib];
	}
      }
    }
    /* Next link */
  }

  /* This is accumulated ... */
  pc->force[X] += f[X];
  pc->force[Y] += f[Y];
  pc->force[Z] += f[Z];

  return;
}

/*****************************************************************************
 *
 *  phi_force_colloid_stress_kernel
 *
 *  Colloid per block, single thread.
 *
 *****************************************************************************/

__global__ void phi_force_colloid_stress_kernel(const pth_t * pth,
						const lb_model_t * model,
						colloids_info_t * info) {
  int nc = blockIdx.x;
  colloid_t * pc = info->pointers->colloid[nc];

  phi_force_colloid_stress_accumulate(pth, model, pc);

  return;
}

/*****************************************************************************
 *
 *  phi_force_colloid_stress_driver
 *
 *****************************************************************************/

int phi_force_colloid_stress_driver(const pth_t * pth, colloids_info_t * info,
				    const lb_t * lb) {
  int ndevice = 0;

  tdpAssert(tdpGetDeviceCount(&ndevice));

  if (ndevice == 0) {
    /* Host */
    #pragma omp parallel for
    for (int n = 0; n < info->npall; ++n) {
      colloid_t * pc = info->pointers->colloid[n];
      phi_force_colloid_stress_accumulate(pth, &lb->model, pc);
    }
  }
  else {
    /* Kernel */
    dim3 blocks = {1, 1, 1};
    dim3 threads = {1, 1, 1};

    blocks.x = info->npall;

    tdpLaunchKernel(phi_force_colloid_stress_kernel, blocks, threads, 0, 0,
		    pth->target, &lb->target->model, info->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpStreamSynchronize(0));
  }

  return 0;
}
