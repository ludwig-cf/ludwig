/*****************************************************************************
 *
 *  build.c
 *
 *  Responsible for the construction of links for particles which
 *  do bounce back on links.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2006-2026 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "colloid_sums.h"
#include "psi_colloid.h"
#include "util.h"
#include "util_ellipsoid.h"
#include "util_math_inline.h"
#include "util_vector.h"
#include "wall.h"
#include "build.h"
#include "build_remove_replace.h"
#include "build_remove_replace_phi.h"
#include "build_remove_replace_q.h"


int build_replace_fluid_local(colloids_info_t * info, colloid_t * pc,
			      int index, lb_t * lb);

static int build_remove_fluid(lb_t * lb, int index, colloid_t * pc);

int build_conservation_psi(colloids_info_t * cinfo, psi_t * psi,
			   const lb_model_t * model);

int build_update_map_driver(map_t * map);
int build_update_map_colloids_driver(colloids_info_t * info, map_t * map);
int build_update_map_additional_driver(colloids_info_t * info, map_t * map);

/*****************************************************************************
 *
 *  build_update_map
 *
 *  This routine is responsible for setting the solid/fluid status
 *  of all nodes in the presence on colloids. This must be complete
 *  before attempting to build the colloid links.
 *
 ****************************************************************************/

int build_update_map(colloids_info_t * cinfo, map_t * map) {

  /* Reset the solid/fluid status map */
  build_update_map_driver(map);
  colloids_info_map_update(cinfo);

  /* Update the current colloid map */
  colloids_info_list_all_build(cinfo);
  build_update_map_additional_driver(cinfo, map);

  return 0;
}

/*****************************************************************************
 *
 *  build_remove_replace_fluid
 *
 *  Compare the current coll_map with the one from the previous time
 *  step and act on changes:
 *
 *    (1) newly occupied sites must have their fluid removed
 *    (2) newly vacated sites must have fluid replaced.
 *
 *  Correction terms are added for the appropriate colloids to be
 *  implemented at the next step.
 *
 *  The 'abstract' free energy fe may be NULL for single fluid.
 *
 *****************************************************************************/

int build_remove_replace(fe_t * fe, colloids_info_t * cinfo, lb_t * lb,
			 field_t * phi, field_t * q, psi_t * psi,
			 map_t * map) {

  int ic, jc, kc, index;
  int is_halo;
  int nlocal[3];
  int nhalo;
  colloid_t * pcold;
  colloid_t * pcnew;

  assert(lb);
  assert(cinfo);

  build_bbl_rebuild_flags_driver(cinfo);
  build_remove_replace_fluid_driver(lb, cinfo, map);
  build_remove_replace_order_parameter_driver(lb, cinfo, map, phi);
  build_remove_replace_q_driver(lb, (fe_lc_t *) fe, cinfo, map, q);

  if (psi == NULL) return 0;

  /* Charge is dealt with here. */

  /* Required until there is a device implementation for charge */
  map_memcpy(map, tdpMemcpyDeviceToHost);
  colloids_memcpy(cinfo, tdpMemcpyDeviceToHost);

  cs_nlocal(lb->cs, nlocal);
  cs_nhalo(lb->cs, &nhalo);

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = cs_index(lb->cs, ic, jc, kc);

	colloids_info_map_old(cinfo, index, &pcold);
	colloids_info_map(cinfo, index, &pcnew);

	is_halo = (ic < 1 || jc < 1 || kc < 1 ||
		   ic > nlocal[X] || jc > nlocal[Y] || kc > nlocal[Z]);

	if (pcold == NULL && pcnew != NULL) {

	  if (!is_halo) {
	    psi_colloid_remove_charge(psi, pcnew, index);
	  }
	}

	if (pcold != NULL && pcnew == NULL) {

	  if (!is_halo) {
	    psi_colloid_replace_charge(psi, cinfo, pcold, index);
	  }
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  Remove replace fluid (only)
 *  Not currently used.
 *
 *****************************************************************************/

int build_remove_replace_policy_local(cs_t * cs, colloids_info_t * cinfo,
				      lb_t * lb) {
  int ic, jc, kc, index;
  int nlocal[3];
  colloid_t * pcold;
  colloid_t * pcnew;

  assert(cs);
  assert(lb);
  assert(cinfo);

  cs_nlocal(cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);

	colloids_info_map_old(cinfo, index, &pcold);
	colloids_info_map(cinfo, index, &pcnew);

	if (pcold == NULL && pcnew != NULL) {
	  build_remove_fluid(lb, index, pcnew);
	}

	if (pcold != NULL && pcnew == NULL) {
	  build_replace_fluid_local(cinfo, pcold, index, lb);
	}
	/* Next site */
      }
    }
  }

  return 0;
}

/******************************************************************************
 *
 *  build_remove_fluid
 *
 *  Remove density, momentum at site inode.
 *
 *  Corrections to the mass, force, and torque updates to the relevant
 *  colloid are required.
 *
 *  We don't care about the 'swallowed' distribution information
 *  associated with the old fluid.
 *
 *****************************************************************************/

static int build_remove_fluid(lb_t * lb, int index, colloid_t * p_colloid) {

  int    ia;
  int    ib[3];
  int    noffset[3];

  double rho;             /* density of removed fluid */
  double g[3];            /* momentum of removed fluid */
  double r0[3];           /* Local coords of colloid centre */
  double rb[3];           /* Boundary vector at lattice site index */
  double rtmp[3];
  double rho0;
  physics_t  * phys = NULL;

  assert(lb);

  cs_nlocal_offset(lb->cs, noffset);
  cs_index_to_ijk(lb->cs, index, ib);

  physics_ref(&phys);
  physics_rho0(phys, &rho0);

  /* Get the properties of the old fluid at inode */

  lb_0th_moment(lb, index, LB_RHO, &rho);
  lb_1st_moment(lb, index, LB_RHO, g);

  /* Set the corrections for colloid motion. This requires
   * the local boundary vector rb for the torque */

  p_colloid->deltam -= (rho - rho0);

  for (ia = 0; ia < 3; ia++) {
    p_colloid->f0[ia] += g[ia];
    r0[ia] = p_colloid->s.r[ia] - 1.0*noffset[ia];
    rtmp[ia] = 1.0*ib[ia];
  }

  cs_minimum_distance(lb->cs, r0, rtmp, rb);
  cross_product(rb, g, rtmp);

  for (ia = 0; ia < 3; ia++) {
    p_colloid->t0[ia] += rtmp[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_replace_fluid_local
 *
 *  For COLLOID_REPLACE_POLICY_LOCAL, replace distributions
 *  by using a reprojection based on the local solid body
 *  velocity of the colloid that has just vacated the site.
 *
 *  This has the advantage (cf interpolation) of being local.
 *  [Test coverage?]
 *
 *****************************************************************************/

int build_replace_fluid_local(colloids_info_t * cinfo, colloid_t * pc,
			      int index, lb_t * lb) {

  int ia, ib, p;
  double rho0;
  double f, sdotq, udotc;
  double rb[3], ub[3];
  double gnew[3] = {0.0, 0.0, 0.0};
  double tnew[3] = {0.0, 0.0, 0.0};

  assert(cinfo);
  assert(pc);
  assert(lb);

  /* Compute new distribution */

  rho0 = lb->param->rho0; /* fluid density */
  colloid_rb_ub(cinfo, pc, index, rb, ub);

  for (p = 0; p < lb->model.nvel; p++) {
    double cs2 = lb->model.cs2;
    double rcs2 = 1.0/cs2;
    udotc = lb->model.cv[p][X]*ub[X]
          + lb->model.cv[p][Y]*ub[Y]
          + lb->model.cv[p][Z]*ub[Z];
    sdotq = 0.0;
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	double dab = (ia == ib);
	double q = lb->model.cv[p][ia]*lb->model.cv[p][ib] - cs2*dab;
	sdotq += q*ub[ia]*ub[ib];
      }
    }

    f = lb->model.wv[p]*(rho0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq);
    lb_f_set(lb, index, p, LB_RHO, f);

    /* Subtract momentum from colloid (contribution to) */
    gnew[X] -= f*lb->model.cv[p][X];
    gnew[Y] -= f*lb->model.cv[p][Y];
    gnew[Z] -= f*lb->model.cv[p][Z];
  }

  cross_product(rb, gnew, tnew);

  for (ia = 0; ia < 3; ia++) {
    pc->f0[ia] += gnew[ia];
    pc->t0[ia] += tnew[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_count_faces_local
 *
 *  Count number of faces (local) for this colloid. This is the 'surface
 *  area' on the finite difference grid.
 *
 *  Count both total, and those faces which have fluid neighbours.
 *
 *****************************************************************************/

int build_count_faces_local(colloid_t * colloid, const lb_model_t * model,
			    double * sa, double * saf) {

  assert(colloid);
  assert(sa);
  assert(saf);
  assert(model);

  *sa = 0.0;
  *saf = 0.0;

  for (int nl = 0; nl < colloid->links->active_links; ++nl) {
    if (colloid->links->status[nl] == LINK_UNUSED) continue;
    int p  = colloid->links->p[nl];
    int pp = util_square_modulus_int8(model->cv[p]);
    if (pp == 1) {
      *sa += 1.0;
      if (colloid->links->status[nl] == LINK_FLUID) *saf += 1.0;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_conservation
 *
 *  Restore conserved order parameters (phi, charge) after change of
 *  shape in the finite volume picture.
 *
 *  Either phi or psi are allowed to be NULL, in which case, they are
 *  ignored. If both are NULL, don't need to call this at all!
 *
 *****************************************************************************/

int build_conservation(colloids_info_t * cinfo, field_t * phi, psi_t * psi,
		       const lb_model_t * model) {

  assert(cinfo);

  if (phi) build_conservation_phi_driver(cinfo, phi);
  if (psi) build_conservation_psi(cinfo, psi, model);

  return 0;
}

/*****************************************************************************
 *
 *  build_conservation_psi
 *
 *  Ensure fluid charge is conserved following remove / replace.
 *
 *  Charge has the additional constraint that quantity of charge must
 *  not fall below zero. This means some correction may be carried
 *  forward to future steps.
 *
 *****************************************************************************/

int build_conservation_psi(colloids_info_t * cinfo, psi_t * psi,
			   const lb_model_t * model) {

  double value;
  double dq0, dq1;
  double sa_local, saf_local;

  colloid_t * colloid = NULL;

  assert(cinfo);
  assert(psi);

  colloids_info_all_head(cinfo, &colloid);

  for (; colloid != NULL; colloid = colloid->nextall) {

    /* Add any contribution form previous steps (all copies);
     * work out what should be put back. */

    colloid->dq[0] += colloid->s.deltaq0;
    colloid->dq[1] += colloid->s.deltaq1;

    dq0  = colloid->dq[0]  / colloid->s.saf;
    dq1  = colloid->dq[1]  / colloid->s.saf;

    if (dq0 == 0.0 && dq1 == 0.0) continue;

    /* Locally, the total we expect to put back is: */

    build_count_faces_local(colloid, model, &sa_local, &saf_local);

    assert(colloid->s.saf > 0.0);
    colloid->dq[0] *= saf_local/colloid->s.saf;
    colloid->dq[1] *= saf_local/colloid->s.saf;

  for (int nl = 0; nl < colloid->links->active_links; ++nl) {

      if (colloid->links->status[nl] != LINK_FLUID) continue;

      int pl= colloid->links->p[nl];
      int pp = util_square_modulus_int8(model->cv[pl]);

      if (pp == 1) {
	int index = colloid->links->i[nl];
	/* For charge, do not drop densities below zero. */
	psi_rho(psi, index, 0, &value);
	if ((value + dq0) >= 0.0) {
	  colloid->dq[0] -= dq0;
	  psi_rho_set(psi, index, 0, value + dq0);
	}
	psi_rho(psi, index, 1, &value);
	if ((value + dq1) >=  0.0) {
	  colloid->dq[1] -= dq1;
	  psi_rho_set(psi, index, 1, value + dq1);
	}
      }
    }
  }

  /* Now, repeat the sum of dq so that all copies have a copy
   * of any shortfall in what we have tried to put back.
   * Record this in the state so it is always retained. */

  colloid_sums_halo(cinfo, COLLOID_SUM_CONSERVATION);

  colloids_info_all_head(cinfo, &colloid);

  for (; colloid; colloid = colloid->nextall) {
    colloid->s.deltaq0 = colloid->dq[0];
    colloid->s.deltaq1 = colloid->dq[1];
    colloid->dq[0] = 0.0;
    colloid->dq[1] = 0.0;
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_update_map_kernel
 *
 *****************************************************************************/

__global__ void build_update_map_kernel(kernel_3d_t k3d, map_t * map, double c,
                                        double h) {
  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    int index  = cs_index(map->cs, ic, jc, kc);
    int status = MAP_FLUID;

    /* A check is required to ensure we do not, e.g., update boundary
     * sites */

    map_status(map, index, &status);

    if (status == MAP_COLLOID) {
      double wet[2] = {c, h};
      map_status_set(map, index, MAP_FLUID);
      map_data_set(map, index, wet);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_update_map_driver
 *
 *  We will set both wetting constants {c, h} equal zero.
 *
 *****************************************************************************/

int build_update_map_driver(map_t * map) {

  int ifail = 0;

  const double c = 0.0;
  const double h = 0.0;

  assert(map);

  if (map->ndata == 0) {
    ifail = -1;
  }
  else {

    int  nhalo = map->cs->param->nhalo;
    dim3 nblk  = {};
    dim3 ntpb  = {};

    cs_limits_t lim = cs_limits_with_halo(map->cs->param->nlocal, nhalo);
    kernel_3d_t k3d = kernel_3d(map->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(build_update_map_kernel, nblk, ntpb, 0, 0, k3d,
                    map->target, c, h);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpStreamSynchronize(0));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  build_update_map_colloids_kernel
 *
 *  At each lattice site, we check all possible colloids. There must
 *  be at most one particle at each site, or else something has
 *  failed earlier in "collision avoidance". Hence thread safe.
 *
 *****************************************************************************/

__global__ void build_update_map_colloids_kernel(kernel_3d_t       k3d,
                                                 colloids_info_t * info,
                                                 map_t *           map) {
  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    /* Fluid site index and position (local) */
    int    index = cs_index(map->cs, ic, jc, kc);
    double r0[3] = {1.0 * ic, 1.0 * jc, 1.0 * kc};

    /* All sites are by default to be fluid ... */

    colloids_info_map_set(info, index, NULL);

    /* Loop through all copies locally ... */

    for (colloid_t * pc = info->headall; pc; pc = pc->nextall) {

      double dr[3] = {}; /* colloid centre -> site */

      if (pc->s.bc != COLLOID_BC_BBL) {
        continue;
      }

      /* Not a minimum image separation as we are potentially checking more
       * than one copy and need to get the right pointer for this site ... */

      dr[X] = r0[X] - (pc->s.r[X] - 1.0 * map->cs->param->noffset[X]);
      dr[Y] = r0[Y] - (pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y]);
      dr[Z] = r0[Z] - (pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z]);

      /* Are we inside? Set status and wetting constants */

      if (colloid_r_inside(&pc->s, dr)) {

        double wet[2] = {pc->s.c, pc->s.h}; /* Wetting c, h */

        colloids_info_map_set(info, index, pc);
        map_status_set(map, index, MAP_COLLOID);

        /* Janus particles have h = h_0 cos (theta)
         * with s[3] pointing to the 'north pole' */

        if (pc->s.attr & COLLOID_ATTR_JANUS) {
          double mod = util_vector_modulus(dr);
          if (mod > 0.0) {
            double cosine = util_vector_dot_product(pc->s.s, dr) / mod;
            wet[1]        = cosine * wet[1]; /* h */
          }
        }

        map_data_set(map, index, wet);

        break; /* Can skip any further colloids in pc->nextall loop */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_update_map_colloids_driver
 *
 *****************************************************************************/

int build_update_map_colloids_driver(colloids_info_t * info, map_t * map) {

  int ifail = 0;

  assert(map);

  if (map->ndata == 0) {
    /* Wetting actually needs to be treated as a separate issue
     * which would obviate the need for wettting data here */
    ifail = -1;
  }
  else {

    int  nhalo = map->cs->param->nhalo;
    dim3 nblk  = {};
    dim3 ntpb  = {};

    cs_limits_t lim = cs_limits_with_halo(map->cs->param->nlocal, nhalo);
    kernel_3d_t k3d = kernel_3d(map->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    /* Make sure the target copy is up-to-date ... */
    info->target->headall = info->headall;

    tdpLaunchKernel(build_update_map_colloids_kernel, nblk, ntpb, 0, 0, k3d,
                    info->target, map->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpStreamSynchronize(0));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  build_additional_version
 *
 *  If number of colloids is small, use the method above, which is
 *  quicker in this limit.
 *  If the number of colloids is larger, the method above scales
 *  adversly (as nsites x ncolloids). Parallel must then be over
 *  colloids.
 *
 *****************************************************************************/

__host__ __device__ void build_additional_version(colloids_info_t * info,
                                                  map_t *           map,
                                                  colloid_t *       pc) {
  int nhalo     = 1;
  int nlocal[3] = {};
  int i_min, i_max;
  int j_min, j_max;
  int k_min, k_max;

  double amax = colloid_principal_radius(&pc->s);

  cs_nlocal(info->cs, nlocal);
  cs_nhalo(info->cs, &nhalo);

  {
    /* Local limits require colloid position minus offset */
    double rc[3] = {};

    rc[X] = pc->s.r[X] - 1.0 * map->cs->param->noffset[X];
    rc[Y] = pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y];
    rc[Z] = pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z];

    i_min = util_imax(1 - nhalo, (int) floor(rc[X] - amax));
    j_min = util_imax(1 - nhalo, (int) floor(rc[Y] - amax));
    k_min = util_imax(1 - nhalo, (int) floor(rc[Z] - amax));
    i_max = util_imin(nlocal[X] + nhalo, (int) ceil(rc[X] + amax));
    j_max = util_imin(nlocal[Y] + nhalo, (int) ceil(rc[Y] + amax));
    k_max = util_imin(nlocal[Z] + nhalo, (int) ceil(rc[Z] + amax));
  }

  /* Begin search ... */

  for (int ic = i_min; ic <= i_max; ic++) {
    for (int jc = j_min; jc <= j_max; jc++) {
      for (int kc = k_min; kc <= k_max; kc++) {

        int index = cs_index(info->cs, ic, jc, kc);

        double r0[3] = {1.0 * ic, 1.0 * jc, 1.0 * kc};
        double dr[3] = {}; /* colloid centre -> site */

        /* Not a minimum image separation as we are potentially checking more
         * than one copy and need to get the right pointer for this site ... */

        dr[X] = r0[X] - (pc->s.r[X] - 1.0 * map->cs->param->noffset[X]);
        dr[Y] = r0[Y] - (pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y]);
        dr[Z] = r0[Z] - (pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z]);

        /* Are we inside? Set status and wetting constants */

        if (colloid_r_inside(&pc->s, dr)) {

          double wet[2] = {pc->s.c, pc->s.h}; /* Wetting c, h */

          colloids_info_map_set(info, index, pc);
          map_status_set(map, index, MAP_COLLOID);

          /* Janus particles have h = h_0 cos (theta)
           * with s[3] pointing to the 'north pole' */

          if (pc->s.attr & COLLOID_ATTR_JANUS) {
            double mod = util_vector_modulus(dr);
            if (mod > 0.0) {
              double cosine = util_vector_dot_product(pc->s.s, dr) / mod;
              wet[1]        = cosine * wet[1]; /* h */
            }
          }
          map_data_set(map, index, wet);
        }

        /* Nest search site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_additional_kernel
 *
 *****************************************************************************/

__global__ void build_additional_kernel(colloids_info_t * info, map_t * map) {

  assert(info);

  int nc = blockIdx.x;

  if (nc < info->npall) {
    colloid_t * pc = info->pointers->colloid[nc];

    if (pc->s.bc == COLLOID_BC_BBL) {
      build_additional_version(info, map, pc);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_update_map_additional_driver
 *
 *  Reconstruct colloid map by either:
 *  1. "low density" method if the number of colloids is below the limit;
 *  2. "high density" method.
 *
 *****************************************************************************/

int build_update_map_additional_driver(colloids_info_t * info, map_t * map) {

  int ifail = 0;

  if (map->ndata == 0) {
    ifail = -1;
  }
  else {

    int nlimit = 1;

    if (info->nplocal <= nlimit) {
      build_update_map_colloids_driver(info, map);
    }
    else {

      int ndevice = 0;
      tdpAssert(tdpGetDeviceCount(&ndevice));

      /* Nullify the pointer map (device version; or host if alias) */
      tdpAssert(tdpMemset(info->target->map_new, 0,
                          info->nsites * sizeof(colloid_t *)));

      if (ndevice == 0) {
        /* Host only */

#pragma omp parallel for
        for (int n = 0; n < info->npall; n++) {
          colloid_t * pc = info->pointers->colloid[n];
          if (pc->s.bc == COLLOID_BC_BBL) {
            build_additional_version(info, map, pc);
          }
        }
      }
      else {
        /* Kernel */
        dim3 blocks  = {1, 1, 1};
        dim3 threads = {1, 1, 1};

        blocks.x = info->npall;

        tdpLaunchKernel(build_additional_kernel, blocks, threads, 0, 0,
                        info->target, map->target);

        tdpAssert(tdpPeekAtLastError());
        tdpAssert(tdpStreamSynchronize(0));
      }
    }
  }

  return ifail;
}
