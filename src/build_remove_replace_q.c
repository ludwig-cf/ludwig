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
 *  Edinburgh Soft Matter and Statisitical Physics Group and
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
#include "util_vector.h"

int build_replace_q_local(const fe_lc_t * fe,
			  const colloids_info_t * info,
			  const colloid_t * pc,
			  int ic, int jc, int kc, field_t * q);


/*****************************************************************************
 *
 *  build_replace_q_kernel
 *
 *  Replace the order parameter(s) at a newly exposed site (index).
 *
 *****************************************************************************/

__global__ void build_replace_q_kernel(kernel_3d_t k3d,
				       const fe_lc_t * fe,
				       const lb_t * lb,
				       const colloids_info_t * info,
				       const map_t * map,
				       field_t * q) {
  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index0 = cs_index(info->cs, ic, jc, kc);

    colloid_t * pc    = NULL;
    colloid_t * pcnew = NULL;

    colloids_info_map_old(info, index0, &pc);
    colloids_info_map_old(info, index0, &pcnew);

    if (pc != NULL && pcnew == NULL) {

      /* Was solid && now fluid: Q_ab at index needs to be replaced. */

      /* Check the surrounding sites that were linked to inode,
       * and accumulate a (weighted) average distribution. */

      int nweight       = 0;
      double weight     = 0.0;
      double qnew[NQAB] = {};

      for (int p = 1; p < lb->model.nvel; p++) {

	int index = cs_index(lb->cs, ic + lb->model.cv[p][X],
			             jc + lb->model.cv[p][Y],
		                     kc + lb->model.cv[p][Z]);
	int status = MAP_FLUID;	double qs[NQAB] = {};
	colloid_t * pcold = NULL;

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

      if (nweight == 0) {
	/* No fluid information. */
	build_replace_q_local(fe, info, pc, ic, jc, kc, q);
      }
      else {
	weight = 1.0 / weight;
	for (int n = 0; n < NQAB; n++) {
	  qnew[n] *= weight;
	}
      }
      field_scalar_array_set(q, index0, qnew);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_replace_q_local
 *
 *****************************************************************************/

__host__ __device__ int build_replace_q_local(const fe_lc_t * fe,
					      const colloids_info_t * info,
					      const colloid_t * pc,
					      int ic, int jc, int kc,
					      field_t * q) {
  double rb[3]      = {};
  double qnew[3][3] = {};

  assert(fe);
  assert(info);
  assert(pc);
  assert(q);

  double amplitude = 0.0;

  fe_lc_amplitude_compute(fe->param, &amplitude);

  /* For normal anchoring we determine the radial unit vector rb */

  rb[X] = 1.0*ic - (pc->s.r[X] - 1.0*info->cs->param->noffset[X]);
  rb[Y] = 1.0*jc - (pc->s.r[Y] - 1.0*info->cs->param->noffset[Y]);
  rb[Z] = 1.0*kc - (pc->s.r[Z] - 1.0*info->cs->param->noffset[Z]);

  if (pc->s.shape == COLLOID_SHAPE_ELLIPSOID) {
    /* Compute correct spheroid normal ... */
    int isphere = util_ellipsoid_is_sphere(pc->s.elabc);
    if (!isphere) {
      double posvector[3] = {};
      util_vector_copy(3, rb, posvector); /* FIXME What is this...? */
      util_spheroid_surface_normal(pc->s.elabc, pc->s.m, posvector, rb);
    }
  }

  /* Make sure we have a unit vector */
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

    /* FIXME device version required... */
    /* util_random_unit_vector(&pc->s.rng, rhat);*/
    assert(0);

    rhatrb = util_vector_dot_product(rhat, rb);

    rbp[X] = rhat[X] - rhatrb*rb[X];
    rbp[Y] = rhat[Y] - rhatrb*rb[Y];
    rbp[Z] = rhat[Z] - rhatrb*rb[Z];

    rbmod = 1.0/sqrt(rbp[X]*rbp[X] + rbp[Y]*rbp[Y] + rbp[Z]*rbp[Z]);
    rb[X] = rbmod * rbp[X];
    rb[Y] = rbmod * rbp[Y];
    rb[Z] = rbmod * rbp[Z];
  }

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      double d_ab = (ia == ib);
      qnew[ia][ib] = 0.5*amplitude*(3.0*rb[ia]*rb[ib] - d_ab);
    }
  }

  {
    int index = cs_index(info->cs, ic, jc, kc);
    field_tensor_set(q, index, qnew);
  }

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

int build_remove_replace_q_driver(const lb_t * lb,
				  const fe_lc_t * fe,
				  const colloids_info_t * info,
				  const map_t * map,
				  field_t * q) {
  int ifail = 0;

  if (q == NULL) {
    ifail = -1;
  }
  else {
    dim3 nblk = {};
    dim3 ntpb = {};

    cs_limits_t lim = cs_limits(info->cs->param->nlocal);
    kernel_3d_t k3d = kernel_3d(info->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, & ntpb);

    /* FIXME free energy target pointer ... */
    tdpLaunchKernel(build_replace_q_kernel, nblk, ntpb, 0, 0,
		    k3d, fe, lb->target, info->target, map->target, q->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpStreamSynchronize(0));
  }

  return ifail;
}
