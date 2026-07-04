/*****************************************************************************
 *
 *  build_remove_replqce_q.c
 *
 *  Remove / replace tensor order parameter for liquid crystals.
 *  As the order parameter is not conserved, the "remove" is a
 *  no-op. The replacement may depend on things like surface
 *  anchoring.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "build_remove_replace_q.h"
#include "kernel_3d.h"
#include "util_ellipsoid.h"
#include "util_vector.h"

__host__ __device__ int build_replace_q_interp(const lb_t * lb,
                                               const colloids_info_t * info,
                                               const map_t * map,
                                               const field_t * q, int ic,
                                               int jc, int kc,
                                               double qreplacement[NQAB]);
__host__ __device__ int build_replace_q_surface(const fe_lc_t * fe,
                                                const colloids_info_t * info,
                                                colloid_t * pc,
                                                int ic, int jc, int kc,
                                                double qreplacement[NQAB]);

/*****************************************************************************
 *
 *  build_replace_q_kernel
 *
 *  Replace the order parameter(s) at a newly exposed site (index).
 *
 *****************************************************************************/

__global__ void build_replace_q_kernel(kernel_3d_t k3d, const fe_lc_t * fe,
                                       const lb_t * lb,
                                       const colloids_info_t * info,
                                       const map_t * map, field_t * q) {
  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index = cs_index(info->cs, ic, jc, kc);

    colloid_t * pc    = NULL;
    colloid_t * pcnew = NULL;

    colloids_info_map_old(info, index, &pc);
    colloids_info_map(info, index, &pcnew);

    if (pc != NULL && pcnew == NULL) {

      /* Was solid and now fluid: Q_ab at index needs to be replaced. */

      double qnew[NQAB] = {};

      /* Try interpolation ... */
      int have_q = build_replace_q_interp(lb, info, map, q, ic, jc, kc, qnew);

      if (have_q == 0) {
        /* No fluid information. Use the anchoring ... */
        build_replace_q_surface(fe, info, pc, ic, jc, kc, qnew);
      }
      field_scalar_array_set(q, index, qnew);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_replace_q_interpolate
 *
 *  Interpolate between nearby fluid sites.
 *
 *  Returns 0 if no interpolation is available.
 *
 *****************************************************************************/

__host__ __device__ int build_replace_q_interp(const lb_t * lb,
                                               const colloids_info_t * info,
                                               const map_t * map,
                                               const field_t * q,
                                               int ic, int jc, int kc,
                                               double qreplacement[NQAB]) {
  int nweight = 0;

  double weight     = 0.0;
  double qnew[NQAB] = {};

  for (int p = 1; p < lb->model.nvel; p++) {

    int index = cs_index(lb->cs, ic + lb->model.cv[p][X],
                                 jc + lb->model.cv[p][Y],
                                 kc + lb->model.cv[p][Z]);
    int status = MAP_FLUID;

    double      qs[NQAB] = {};
    colloid_t * pcold    = NULL;

    /* Adjacent site must have been fluid before position update */
    /* Adjacent site must not be a boundary */

    colloids_info_map_old(info, index, &pcold);
    if (pcold) continue;

    map_status(map, index, &status);
    if (status == MAP_BOUNDARY) continue;

    field_scalar_array(q, index, qs);
    for (int n = 0; n < NQAB; n++) {
      qnew[n] += lb->model.wv[p]*qs[n];
    }
    weight  += lb->model.wv[p];
    nweight += 1;
  }

  if (nweight > 0) {
    weight = 1.0 / weight;
    for (int n = 0; n < NQAB; n++) {
      qreplacement[n] = weight*qnew[n];
    }
  }

  return nweight;
}

/*****************************************************************************
 *
 *  build_replace_q_surface
 *
 *  Construct a replacement Q_ab from the local surface anchoring.
 *
 *****************************************************************************/

__host__ __device__ int build_replace_q_surface(const fe_lc_t * fe,
                                                const colloids_info_t * info,
                                                colloid_t * pc,
                                                int ic, int jc, int kc,
                                                double qreplacement[NQAB]) {
  double rb[3]      = {};
  double qnew[3][3] = {};

  assert(fe);
  assert(info);
  assert(pc);

  double amplitude = 0.0;

  fe_lc_amplitude_compute(fe->param, &amplitude);

  /* For normal anchoring we determine the radial boundary vector rb */

  rb[X] = 1.0*ic - (pc->s.r[X] - 1.0*info->cs->param->noffset[X]);
  rb[Y] = 1.0*jc - (pc->s.r[Y] - 1.0*info->cs->param->noffset[Y]);
  rb[Z] = 1.0*kc - (pc->s.r[Z] - 1.0*info->cs->param->noffset[Z]);

  if (pc->s.shape == COLLOID_SHAPE_ELLIPSOID) {
    /* Compute correct spheroid normal and copy to rb ... */
    int isphere = util_ellipsoid_is_sphere(pc->s.elabc);
    if (!isphere) {
      double rnormal[3] = {};
      util_spheroid_surface_normal(pc->s.elabc, pc->s.m, rb, rnormal);
      util_vector_copy(3, rnormal, rb);
    }
  }

  /* Make sure we have a unit vector (both cases above) */
  {
    double rbmod = 1.0/sqrt(rb[X]*rb[X] + rb[Y]*rb[Y] + rb[Z]*rb[Z]);
    rb[X] *= rbmod;
    rb[Y] *= rbmod;
    rb[Z] *= rbmod;
  }

  /* For planar degenerate anchoring we subtract the projection of a
     randomly oriented unit vector on rb and renormalise the result   */

  if (fe->param->coll.type == LC_ANCHORING_PLANAR) {

    double rbp[3]  = {};
    double rhat[3] = {};
    double rbmod   = 0.0;
    double rhatrb  = 0.0;

    pc->s.rng = util_vector_random_unit_vector(pc->s.rng, rhat);

    rhatrb = util_vector_dot_product(rhat, rb);

    rbp[X] = rhat[X] - rhatrb*rb[X];
    rbp[Y] = rhat[Y] - rhatrb*rb[Y];
    rbp[Z] = rhat[Z] - rhatrb*rb[Z];

    rbmod = 1.0/sqrt(rbp[X]*rbp[X] + rbp[Y]*rbp[Y] + rbp[Z]*rbp[Z]);
    rb[X] = rbmod*rbp[X];
    rb[Y] = rbmod*rbp[Y];
    rb[Z] = rbmod*rbp[Z];
  }

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      double d_ab  = (ia == ib);
      qnew[ia][ib] = 0.5*amplitude*(3.0*rb[ia]*rb[ib] - d_ab);
    }
  }

  qreplacement[XX] = qnew[X][X];
  qreplacement[XY] = qnew[X][Y];
  qreplacement[XZ] = qnew[X][Z];
  qreplacement[YY] = qnew[Y][Y];
  qreplacement[YZ] = qnew[Y][Z];

  return 0;
}

/*****************************************************************************
 *
 *  build_remove_replace_q_driver
 *
 *  Only replace is significant. If a site becomes solid, the associated
 *  order parameter information is just ignored.
 *
 *****************************************************************************/

int build_remove_replace_q_driver(const lb_t * lb, const fe_lc_t * fe,
                                  const colloids_info_t * info,
                                  const map_t * map, field_t * q) {
  int ifail = 0;

  if (q == NULL) {
    ifail = -1;
  }
  else {
    dim3 nblk = {};
    dim3 ntpb = {};

    cs_limits_t lim = cs_limits(info->cs->param->nlocal);
    kernel_3d_t k3d = kernel_3d(info->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(build_replace_q_kernel, nblk, ntpb, 0, 0, k3d, fe->target,
                    lb->target, info->target, map->target, q->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpStreamSynchronize(0));
  }

  return ifail;
}
