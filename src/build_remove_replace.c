/*****************************************************************************
 *
 *  build_remove_replace.c
 *
 *  Removal or replacement of fluid on change in discrete particle
 *  shape. Removal/replacement of order parameters could perhaps be
 *  separate, although the routines are of a similar form.
 *
 *  The overall driver is found in build,c
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

#include "build_remove_replace.h"
#include "kernel_3d.h"

/*****************************************************************************
 *
 *  build_bbl_rebuild_flags_kernel
 *
 *  Looks for changes in the status map and sets the rebuild flag (only).
 *
 *****************************************************************************/

__global__ void build_bbl_rebuild_flags_kernel(kernel_3d_t       k3d,
                                               colloids_info_t * info) {
  assert(info);

  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    /* Fluid site index and position (local) */
    int index = cs_index(info->cs, ic, jc, kc);

    colloid_t * pcold = NULL;
    colloid_t * pcnew = NULL;

    colloids_info_map_old(info, index, &pcold);
    colloids_info_map(info, index, &pcnew);

    /* Potential race (albeit benign) */
    if (pcold == NULL && pcnew != NULL) pcnew->s.rebuild = 1;
    if (pcold != NULL && pcnew == NULL) pcold->s.rebuild = 1;
  }

  return;
}

/*****************************************************************************
 *
 *  build_bbl_rebuild_flags_driver
 *
 *****************************************************************************/

int build_bbl_rebuild_flags_driver(colloids_info_t * info) {

  assert(info);

  int  nhalo = info->cs->param->nhalo;
  dim3 nblk  = {};
  dim3 ntpb  = {};

  cs_limits_t lim = cs_limits_with_halo(info->cs->param->nlocal, nhalo);
  kernel_3d_t k3d = kernel_3d(info->cs, lim);

  kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

  tdpLaunchKernel(build_bbl_rebuild_flags_kernel, nblk, ntpb, 0, 0, k3d,
                  info->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpStreamSynchronize(0));

  return 0;
}

/*****************************************************************************
 *
 *  build_remove_fluid_kernel
 *
 *****************************************************************************/

__global__ void build_remove_fluid_kernel(kernel_3d_t k3d, lb_t * lb,
                                          colloids_info_t * info,
					  double rho0) {
  assert(lb);
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
      /* Fluid is removed and corrections are added to particle now at this
       * site ... (can be one or more sites) */

      double rho        = 0.0;
      double rhou[3]    = {0};
      double rbxrhou[3] = {0};

      /* Get the properties of the old fluid at inode */

      lb_0th_moment(lb, index, LB_RHO, &rho);
      lb_1st_moment(lb, index, LB_RHO, rhou);

      /* Set the corrections for colloid motion. This requires
       * the local boundary vector rb for the torque */

      /* Mass (anomaly) */
      atomicAdd(&pc->deltam, -(rho - rho0));

      /* Force */
      atomicAdd(pc->f0 + X, rhou[X]);
      atomicAdd(pc->f0 + Y, rhou[Y]);
      atomicAdd(pc->f0 + Z, rhou[Z]);

      {
        double r0[3] = {1.0 * ic, 1.0 * jc, 1.0 * kc};
        double rb[3] = {0}; /* centre -> local site r0 (local coords) */

        rb[X] = r0[X] - (pc->s.r[X] - 1.0*lb->cs->param->noffset[X]);
        rb[Y] = r0[Y] - (pc->s.r[Y] - 1.0*lb->cs->param->noffset[Y]);
        rb[Z] = r0[Z] - (pc->s.r[Z] - 1.0*lb->cs->param->noffset[Z]);

        util_vector_cross_product(rbxrhou, rb, rhou);
      }

      /* Torque correction */
      atomicAdd(pc->t0 + X, rbxrhou[X]);
      atomicAdd(pc->t0 + Y, rbxrhou[Y]);
      atomicAdd(pc->t0 + Z, rbxrhou[Z]);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_replace_fluid_by_interpolation
 *
 *  At (ic, jc, kc), identify a replacement distribution fnew[] by
 *  forming an average af nearby points.
 *
 *  The number of points used in the interpolation is returned
 *  (may be zero).
 *
 *****************************************************************************/

__host__ __device__
int build_replace_fluid_by_interpolation(lb_t * lb, colloids_info_t * info,
                                         const map_t * map,
					 int ic, int jc, int kc,
                                         double fnew[27]) {
  int    iused  = 0;
  double weight = 0.0;

  /* Check the surrounding sites that were linked to inode,
   * and accumulate a (weighted) average distribution. */

  for (int p = 1; p < lb->model.nvel; p++) {

    int indexp = cs_index(lb->cs, ic + lb->model.cv[p][X],
                                  jc + lb->model.cv[p][Y],
			          kc + lb->model.cv[p][Z]);

    /* Need to exclude boundary sites, which do not appear in colloid
     * map, and must have been fluid at previous step */

    int         status = MAP_FLUID;
    colloid_t * pcold  = NULL;

    colloids_info_map_old(info, indexp, &pcold);
    if (pcold) continue;

    map_status(map, indexp, &status);
    if (status == MAP_BOUNDARY) continue;

    for (int pdash = 0; pdash < lb->model.nvel; pdash++) {
      double f = 0.0;
      lb_f(lb, indexp, pdash, LB_RHO, &f);
      fnew[pdash] += lb->model.wv[p]*f;
    }

    weight += lb->model.wv[p];
    iused += 1;
  }

  if (iused > 0) {
    /* Apply the weight ... */
    weight = 1.0 / weight;
    for (int p = 0; p < lb->model.nvel; p++) {
      fnew[p] *= weight;
    }
  }

  return iused;
}

/*****************************************************************************
 *
 *  build_replace_fluid_by_equilibrium
 *
 *****************************************************************************/

__host__ __device__ void build_replace_fluid_by_equilibrium(lb_t *       lb,
                                                            double       rho0,
                                                            const double ub[3],
                                                            double fnew[27]) {
  for (int p = 0; p < lb->model.nvel; p++) {

    int8_t * cv  = lb->model.cv[p];

    double cs2   = lb->model.cs2;
    double rcs2  = 1.0 / cs2;
    double udotc = cv[X]*ub[X] + cv[Y]*ub[Y] + cv[Z]*ub[Z];
    double sdotq = 0.0;

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
        double dab = (ia == ib);
        sdotq += (cv[ia]*cv[ib] - cs2*dab)*ub[ia]*ub[ib];
      }
    }
    fnew[p] = lb->model.wv[p] * (rho0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq);
  }

  return;
}

/*****************************************************************************
 *
 *  build_replace_fluid_kernel
 *
 *  We need to:
 *
 *    1. provide a new distribution at the relevant site;
 *    2. accumulate corrections to colloid mass, force, torque
 *
 *  Parallelism over the lattice means that the update to the distribution
 *  is thread safe. However, two or more sites may contribute to a single
 *  colloid correction, so updates to the colloids must be atomic.
 *
 *****************************************************************************/

__global__ void build_replace_fluid_kernel(kernel_3d_t k3d, lb_t * lb,
                                           colloids_info_t * info,
                                           const map_t * map, double rho0) {
  assert(lb);
  assert(info);
  assert(lb->model.nvel <= 27);

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

      /* Fluid needs to be replaced at newly exposed site with appropriate
       * quantities; corrections to the colloid mass/force/torque */

      int interpolate = 0;

      double rhonew   = 0.0;
      double rhou[3]  = {0};
      double fnew[27] = {0};

      double rb[3]      = {0}; /* colloid centre -> boundary site "r_b" */
      double rbxrhou[3] = {0};

      rb[X] = 1.0*ic - (pc->s.r[X] - 1.0*lb->cs->param->noffset[X]);
      rb[Y] = 1.0*jc - (pc->s.r[Y] - 1.0*lb->cs->param->noffset[Y]);
      rb[Z] = 1.0*kc - (pc->s.r[Z] - 1.0*lb->cs->param->noffset[Z]);

      /* Obtain a new distribution */

      interpolate = build_replace_fluid_by_interpolation(lb, info, map,
                                                         ic, jc, kc, fnew);
      if (interpolate == 0) {
        /* ... it was not possible to interpolate, so ... */
        double ub[3] = {};
        colloid_ub(pc, rb, ub);
        build_replace_fluid_by_equilibrium(lb, rho0, ub, fnew);
      }

      /* Compute the new rho, rhou (with a sign) from the fnew ... */

      for (int p = 0; p < lb->model.nvel; p++) {
        lb_f_set(lb, index, p, LB_RHO, fnew[p]);

        rhonew += fnew[p];
        rhou[X] -= fnew[p]*lb->model.cv[p][X];
        rhou[Y] -= fnew[p]*lb->model.cv[p][Y];
        rhou[Z] -= fnew[p]*lb->model.cv[p][Z];
      }

      /* Set corrections for excess mass and momentum, and for the
       * correction to the torque */

      atomicAdd(&pc->deltam, (rhonew - rho0));

      atomicAdd(pc->f0 + X, rhou[X]);
      atomicAdd(pc->f0 + Y, rhou[Y]);
      atomicAdd(pc->f0 + Z, rhou[Z]);

      util_vector_cross_product(rbxrhou, rb, rhou);

      atomicAdd(pc->t0 + X, rbxrhou[X]);
      atomicAdd(pc->t0 + Y, rbxrhou[Y]);
      atomicAdd(pc->t0 + Z, rbxrhou[Z]);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_remove_replace_fluid_driver
 *
 *  Remove/replace takes place only at local sites.
 *
 *****************************************************************************/

int build_remove_replace_fluid_driver(lb_t * lb, colloids_info_t * info,
                                      map_t * map) {
  assert(lb);
  assert(info);

  dim3 nblk = {};
  dim3 ntpb = {};

  cs_limits_t lim = cs_limits(info->cs->param->nlocal);
  kernel_3d_t k3d = kernel_3d(info->cs, lim);

  kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

  /* Remove */
  tdpLaunchKernel(build_remove_fluid_kernel, nblk, ntpb, 0, 0, k3d, lb->target,
                  info->target, lb->param->rho0);
  tdpAssert(tdpPeekAtLastError());

  /* Replace */
  tdpLaunchKernel(build_replace_fluid_kernel, nblk, ntpb, 0, 0, k3d,
                  lb->target, info->target, map->target, lb->param->rho0);
  tdpAssert(tdpPeekAtLastError());

  tdpAssert(tdpStreamSynchronize(0));

  return 0;
}

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

__host__ __device__ int build_replace_phi_distributions(lb_t * lb,
							field_t * phi,
							colloids_info_t * info,
							map_t * map,
							int ic, int jc, int kc,
							double * phinew) {
  int interpolate = 0;
  double gnew[27] = {0};
  double weight   = 0.0;

  /* Reset the distribution  */

  for (int p = 1; p < lb->model.nvel; p++) {

    int index = cs_index(lb->cs, ic + lb->model.cv[p][X],
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

  weight = 1.0/weight;

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
						     lb_t * lb,
                                                     field_t *  phi,
                                                     colloids_info_t * info,
                                                     map_t * map,
                                                     double phi0) {
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

int build_remove_replace_order_parameter_driver(lb_t *            lb,
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

    /* Remove */
    tdpLaunchKernel(build_remove_order_parameter_kernel, nblk, ntpb, 0, 0, k3d,
                    phi->target, info->target, phi0);
    tdpAssert(tdpPeekAtLastError());

    /* Replace */
    tdpLaunchKernel(build_replace_order_parameter_kernel, nblk, ntpb, 0, 0,
                   k3d, lb->target, phi->target, info->target, map->target,
                   phi0);
    tdpAssert(tdpPeekAtLastError());

    tdpAssert(tdpStreamSynchronize(0));
  }

  return ifail;
}
