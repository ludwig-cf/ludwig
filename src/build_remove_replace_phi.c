/*****************************************************************************
 *
 *  build_remove_replace_phi.c
 *
 *  Removal and replacement of compositional order parameter on colloid
 *  movement.
 *
 *  Also corrections to ensure global conservation.
 *
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Edinburgh Soft Matter an Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "build_remove_replace_phi.h"
#include "kernel_3d.h"

/*****************************************************************************
 *
 *  build_remove_order_parameter_kernel
 *
 *****************************************************************************/

__global__ void build_remove_order_parameter_kernel(kernel_3d_t       k3d,
                                                    field_t *         phi,
                                                    colloids_info_t * info,
                                                    double            phi0) {
  assert(phi);
  assert(info);

  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    /* Fluid site index and position (local) */
    int index = cs_index(info->cs, ic, jc, kc);

    colloid_t * pcold = NULL;
    colloid_t * pc    = NULL;

    colloids_info_map_old(info, index, &pcold);
    colloids_info_map(info, index, &pc);

    if (pcold == NULL && pc != NULL) {
      /* Here we always use the value from the scalar field. */
      double phiold = 0.0;
      field_scalar(phi, index, &phiold);
      atomicAdd(&pc->s.deltaphi, +(phiold - phi0));
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_replace_phi_distributions
 *
 *  Identify a replacement phinew for position (ic, jc, kc).
 *  At the same time, reset the lb distributuions.
 *
 *  If no nearby valid values are available, return 0 and phinew is
 *  undefined.
 *
 *****************************************************************************/

__host__ __device__
int build_replace_phi_distributions(lb_t * lb, field_t * phi,
                                    colloids_info_t * info, map_t * map,
                                    int ic, int jc, int kc,
                                    double * phinew) {
  int interpolate = 0;

  double gnew[27] = {};
  double weight   = 0.0;

  /* Reset the distribution  */

  for (int p = 1; p < lb->model.nvel; p++) {

    int index  = cs_index(lb->cs, ic + lb->model.cv[p][X],
                                  jc + lb->model.cv[p][Y],
                                  kc + lb->model.cv[p][Z]);
    int status = MAP_FLUID;

    colloid_t * pc = NULL;

    /* Site must have been fluid before position update */

    colloids_info_map_old(info, index, &pc);
    if (pc) continue;

    map_status(map, index, &status);
    if (status == MAP_BOUNDARY) continue;

    for (int pdash = 0; pdash < lb->model.nvel; pdash++) {
      double g = 0.0;
      lb_f(lb, index, pdash, LB_PHI, &g);
      gnew[pdash] += lb->model.wv[p]*g;
    }
    weight += lb->model.wv[p];
    interpolate += 1;
  }

  /* Set new fluid distributions */

  if (interpolate == 0) {
    int index = cs_index(lb->cs, ic, jc, kc);
    /* No neighbouring fluid: as there's no information, we
     * fall back to the value that is currently stored on the
     * lattice. This is not entirely unreasonable, as it may
     * reflect what is nearby, or initial conditions. It could
     * also be set as a contingency in a separate step. */
    field_scalar(phi, index, gnew + 0);
    weight = 1.0;
  }

  weight = 1.0 / weight;

  for (int p = 0; p < lb->model.nvel; p++) {
    int index = cs_index(lb->cs, ic, jc, kc);
    gnew[p] *= weight;
    lb_f_set(lb, index, p, LB_PHI, gnew[p]);
  }

  if (interpolate > 0) {
    double phia = 0.0;
    for (int p = 0; p < lb->model.nvel; p++) {
      phia += gnew[p];
    }
    *phinew = phia;
  }

  return interpolate;
}

/*****************************************************************************
 *
 *  build_replace_phi_scalar
 *
 *  Identify a replacement phinew for position (ic, jc, kc).
 *
 *  If no nearby valid values are available, return 0 and phinew is
 *  undefined.
 *
 *****************************************************************************/

__host__ __device__ int build_replace_phi_scalar(lb_t * lb, field_t * phi,
                                                 colloids_info_t * info,
                                                 map_t * map, int ic, int jc,
                                                 int kc, double * phinew) {
  int interpolate = 0;

  double phia   = 0.0;
  double weight = 0.0;

  /* Form weighted average ... */

  for (int p = 1; p < lb->model.nvel; p++) {

    int index  = cs_index(lb->cs, ic + lb->model.cv[p][X],
                                  jc + lb->model.cv[p][Y],
                                  kc + lb->model.cv[p][Z]);
    int status = MAP_FLUID;

    double      phiold = 0.0;
    colloid_t * pcold  = NULL;

    /* Site must have been fluid before position update */

    colloids_info_map_old(info, index, &pcold);
    if (pcold) continue;

    map_status(map, index, &status);
    if (status == MAP_BOUNDARY) continue;

    field_scalar(phi, index, &phiold);
    phia += lb->model.wv[p]*phiold;

    weight += lb->model.wv[p];
    interpolate += 1;
  }

  if (interpolate > 0) *phinew = phia / weight;

  return interpolate;
}

/*****************************************************************************
 *
 *  build_replace_order_parameter_kernel
 *
 *  Order parameter here refers to compositional order parameter for
 *  symmetric free energies.
 *
 *  If symmetric_lb, we need to replace the relevant distribution, LB_PHI.
 *  We always replace the scalar field phi.
 *
 *****************************************************************************/

__global__ void build_replace_order_parameter_kernel(kernel_3d_t k3d,
                                                     lb_t * lb, field_t * phi,
                                                     colloids_info_t * info,
                                                     map_t *           map,
                                                     double            phi0) {
  assert(lb);
  assert(phi);
  assert(info);
  assert(map);

  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    /* Fluid site index and position (local) */
    int index = cs_index(info->cs, ic, jc, kc);

    colloid_t * pc    = NULL;
    colloid_t * pcnew = NULL;

    colloids_info_map_old(info, index, &pc);
    colloids_info_map(info, index, &pcnew);

    if (pc != NULL && pcnew == NULL) {

      int    interpolate = 0;
      double phinew      = 0.0; /* Replacement value */

      if (lb->ndist == 2) {
        interpolate = build_replace_phi_distributions(lb, phi, info, map,
                                                      ic, jc, kc, &phinew);
      }
      else {
        interpolate =
            build_replace_phi_scalar(lb, phi, info, map, ic, jc, kc, &phinew);
      }

      if (interpolate == 0) {
        /* No information. For phinew, use existing (solid) value. */
        field_scalar(phi, index, &phinew);
      }
      else {
        field_scalar_set(phi, index, phinew);
      }

      /* Set correction arising from change in conserved order parameter. */

      atomicAdd(&pc->s.deltaphi, -(phinew - phi0));
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_remove_replace_order_parameter_driver
 *
 *****************************************************************************/

int build_remove_replace_order_parameter_driver(lb_t * lb,
                                                colloids_info_t * info,
                                                map_t * map, field_t * phi) {
  int ifail = 0;

  if (phi == NULL) {
    ifail = -1;
  }
  else {

    double      phi0 = 0.0;
    physics_t * p    = NULL;

    dim3 nblk = {};
    dim3 ntpb = {};

    cs_limits_t lim = cs_limits(info->cs->param->nlocal);
    kernel_3d_t k3d = kernel_3d(info->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    physics_ref(&p);
    physics_phi0(p, &phi0);

    /* Remove composition */
    tdpLaunchKernel(build_remove_order_parameter_kernel, nblk, ntpb, 0, 0, k3d,
                    phi->target, info->target, phi0);
    tdpAssert(tdpPeekAtLastError());

    /* Replace composition */
    tdpLaunchKernel(build_replace_order_parameter_kernel, nblk, ntpb, 0, 0,
                    k3d, lb->target, phi->target, info->target, map->target,
                    phi0);
    tdpAssert(tdpPeekAtLastError());

    tdpAssert(tdpStreamSynchronize(0));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  build_conservation_phi_kernel
 *
 *  For each fluid site, check for nearest neighbour colloids which
 *  may supply a correction to be added back to the fluid.
 *
 *****************************************************************************/

__global__ void build_conservation_phi_kernel(kernel_3d_t             k3d,
                                              const colloids_info_t * info,
                                              field_t *               phi) {
  int    kindex        = 0;
  int8_t stencil[6][3] = {{-1, 0, 0}, {1, 0, 0},  {0, -1, 0},
                          {0, 1, 0},  {0, 0, -1}, {0, 0, 1}};

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    /* Fluid site index and position (local) */
    /* Site must be fluid */
    int index = cs_index(info->cs, ic, jc, kc);

    colloid_t * pc = NULL;

    colloids_info_map(info, index, &pc);

    if (pc == NULL) {

      for (int p = 0; p < 6; p++) {

        int px = stencil[p][X];
        int py = stencil[p][Y];
        int pz = stencil[p][Z];

        int indexc = cs_index(info->cs, ic + px, jc + py, kc + pz);

        colloids_info_map(info, indexc, &pc);

        if (pc != NULL) {
          double phi0 = 0.0;
          double dphi = pc->s.deltaphi / pc->s.saf;

          field_scalar(phi, index, &phi0);
          field_scalar_set(phi, index, phi0 + dphi);
        }
      }
    }
  }

  return;
  ;
}

/*****************************************************************************
 *
 *  build_conservation_phi_driver
 *
 *  For conserved scalar order parameter field phi, add corrections
 *  arising from remove/replace back to the fluid to ensure global
 *  conservation.
 *
 *  - correction
 *  - set delta phi to zero for all colloids.
 *
 *****************************************************************************/

int build_conservation_phi_driver(const colloids_info_t * info,
                                  field_t *               phi) {
  dim3 nblk = {};
  dim3 ntpb = {};

  cs_limits_t lim = cs_limits(info->cs->param->nlocal);
  kernel_3d_t k3d = kernel_3d(info->cs, lim);

  kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

  /* Make sure the target copy is up-to-date ... */
  info->target->headall = info->headall;

  tdpLaunchKernel(build_conservation_phi_kernel, nblk, ntpb, 0, 0, k3d,
                  info->target, phi->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpStreamSynchronize(0));

  /* Set all deltaphi to zero. */

  for (colloid_t * pc = info->headall; pc != NULL; pc = pc->nextall) {
    pc->s.deltaphi = 0.0;
  }

  return 0;
}
