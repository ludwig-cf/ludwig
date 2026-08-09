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
#include "util_vector.h"
#include "wall.h"
#include "build.h"
#include "build_remove_replace.h"
#include "build_remove_replace_phi.h"
#include "build_remove_replace_q.h"


int build_replace_fluid_local(colloids_info_t * info, colloid_t * pc,
			      int index, lb_t * lb);

static int build_remove_fluid(lb_t * lb, int index, colloid_t * pc);
static int build_reset_links(cs_t * cs, colloid_t * pc, map_t * map,
			     const lb_model_t * model);
static int build_reconstruct_links(cs_t * cs, colloids_info_t * cinfo,
				   colloid_t * pc, map_t * map,
				   const lb_model_t * model);
static void build_link_mean(colloid_t * pc, double wv, const int8_t cv[3],
			    const double rb[3]);
static int build_colloid_wall_links(cs_t * cs, colloids_info_t * cinfo,
				    colloid_t * pc, map_t * map,
				    const lb_model_t * model);

int build_conservation_psi(colloids_info_t * cinfo, psi_t * psi,
			   const lb_model_t * model);

int build_update_map_driver(map_t * map);
int build_update_map_colloids_driver(colloids_info_t * info, map_t * map);

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
  build_update_map_colloids_driver(cinfo, map);

  /* __NVCC__ temporary, we need to return the map status etc to host */
  /* ... before the link construction on the host can occur */

  map_memcpy(map, tdpMemcpyDeviceToHost);
  colloids_memcpy(cinfo, tdpMemcpyDeviceToHost);

  return 0;
}

/*****************************************************************************
 *
 *  build_update_links
 *
 *  Reconstruct or reset the boundary links for each colloid as necessary.
 *
 *****************************************************************************/

int build_update_links(cs_t * cs, colloids_info_t * cinfo, wall_t * wall,
		       map_t * map, const lb_model_t * model) {

  int ia;
  int ic, jc, kc;
  int ncell[3];
  int nhalo;
  colloid_t * pc;

  assert(cs);
  assert(cinfo);
  assert(map);
  assert(model);

  colloids_info_ncell(cinfo, ncell);
  colloids_info_nhalo(cinfo, &nhalo);

  for (ic = 1 - nhalo; ic <= ncell[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= ncell[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= ncell[Z] + nhalo; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	for (; pc; pc = pc->next) {

	  if (pc->s.bc != COLLOID_BC_BBL) continue;

	  pc->sumw   = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    pc->cbar[ia] = 0.0;
	    pc->rxcbar[ia] = 0.0;
	  }

	  if (pc->s.rebuild) {
	    /* The shape has changed, so need to reconstruct */
	    build_reconstruct_links(cs, cinfo, pc, map, model);
	    if (wall) build_colloid_wall_links(cs, cinfo, pc, map, model);
	  }
	  else {
	    /* Shape unchanged, so just reset existing links */
	    build_reset_links(cs, pc, map, model);
	  }

	  build_count_faces_local(pc, model, &pc->s.sa, &pc->s.saf);

	  /* Next colloid */

	  pc->s.rebuild = 0;
	}

	/* Next cell */
      }
    }
  }

  return 0;
}

/****************************************************************************
 *
 *  build_reconstruct_links
 *
 *  Rebuild the boundary links of a particle whose shape has just
 *  changed.
 *
 *  Check each lattice site in a cube around the particle to see
 *  whether it is inside or outside, and set appropriate links.
 *  The new links overwrite the existing ones, or new memory may
 *  be required if the new shape contains more links. The the
 *  new shape contains fewer links, then flag the excess links
 *  as solid.
 *
  ****************************************************************************/

int build_reconstruct_links(cs_t * cs, colloids_info_t * cinfo,
			    colloid_t * p_colloid,
			    map_t * map, const lb_model_t * model) {

  colloid_link_t * p_link;
  colloid_link_t * p_last;
  int i_min, i_max, j_min, j_max, k_min, k_max;
  int i, ic, ii, j, jc, jj, k, kc, kk;
  int index0, index1, p;
  int status1;

  double       lambda = 0.5;
  double      rsite1[3];
  double      rsep[3];
  double      r0[3];
  int ntotal[3];
  int offset[3];

  double   largestdimn;

  colloid_t * pc = NULL;

  assert(cs);
  assert(model);

  cs_nlocal(cs, ntotal);
  cs_nlocal_offset(cs, offset);

  p_link = p_colloid->lnk;

  /* Failsafe approach: set all links to unused status */

  while (p_link) {
    p_link->status = LINK_UNUSED;
    p_link = p_link->next;
  }

  p_link = p_colloid->lnk;
  p_last = p_link;
  /* ... end failsafe */

  /* Limits of the cube around the particle. Make sure these are
   * the appropriate lattice nodes, which extend to the penultimate
   * site in each direction (to include halos). */

  largestdimn = colloid_principal_radius(&p_colloid->s);

  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

  i_min = imax(1,         (int) floor(r0[X] - largestdimn));
  i_max = imin(ntotal[X], (int) ceil (r0[X] + largestdimn));
  j_min = imax(1,         (int) floor(r0[Y] - largestdimn));
  j_max = imin(ntotal[Y], (int) ceil (r0[Y] + largestdimn));
  k_min = imax(1,         (int) floor(r0[Z] - largestdimn));
  k_max = imin(ntotal[Z], (int) ceil (r0[Z] + largestdimn));

  for (i = i_min; i <= i_max; i++) {
    for (j = j_min; j <= j_max; j++) {
      for (k = k_min; k <= k_max; k++) {

	ic = i;
	jc = j;
	kc = k;

	index1 = cs_index(cs, ic, jc, kc);
	colloids_info_map(cinfo, index1, &pc);
	if (pc == p_colloid) continue;

	rsite1[X] = 1.0*i;
	rsite1[Y] = 1.0*j;
	rsite1[Z] = 1.0*k;
	cs_minimum_distance(cs, r0, rsite1, rsep);
	map_status(map, index1, &status1);

	/* Index 1 is outside, so cycle through the lattice vectors
	 * to determine if the end is inside, and so requires a link */

	for (p = 1; p < model->nvel; p++) {

	  /* Find the index of the inside site */

	  ii = ic + model->cv[p][X];
	  jj = jc + model->cv[p][Y];
	  kk = kc + model->cv[p][Z];

	  index0 = cs_index(cs, ii, jj, kk);
	  colloids_info_map(cinfo, index0, &pc);
	  if (pc != p_colloid) continue;

	  /* Index 0 is inside, so now add a link*/

	  if (p_link) {
	    /* Use existing link (lambda always 0.5 at moment) */

	    p_link->rb[X] = rsep[X] + lambda*model->cv[p][X];
	    p_link->rb[Y] = rsep[Y] + lambda*model->cv[p][Y];
	    p_link->rb[Z] = rsep[Z] + lambda*model->cv[p][Z];

	    p_link->i = index1;
	    p_link->j = index0;
	    p_link->p = p;

	    if (status1 == MAP_FLUID) {
	      p_link->status = LINK_FLUID;
	      build_link_mean(p_colloid, model->wv[p], model->cv[p],
			      p_link->rb);
	    }
	    else {
	      p_link->status = LINK_COLLOID;
	    }

	    /* Next link */
	    p_last = p_link;
	    p_link = p_link->next;

	  }
	  else {
	    /* Add a new link to the end of the list */

	    p_link = colloid_link_allocate();

	    p_link->rb[X] = rsep[X] + lambda*model->cv[p][X];
	    p_link->rb[Y] = rsep[Y] + lambda*model->cv[p][Y];
	    p_link->rb[Z] = rsep[Z] + lambda*model->cv[p][Z];

	    p_link->i = index1;
	    p_link->j = index0;
	    p_link->p = p;

	    if (status1 == MAP_FLUID) {
	      p_link->status = LINK_FLUID;
	      build_link_mean(p_colloid, model->wv[p], model->cv[p],
			      p_link->rb);
	    }
	    else {
	      p_link->status = LINK_COLLOID;
	    }

	    if (p_colloid->lnk == NULL) {
	      /* Remember to attach the head of the list */
	      p_colloid->lnk = p_link;
	    }
	    else {
	      assert(p_last);
	      p_last->next = p_link;
	    }

	    p_link->next = NULL;
	    p_last = p_link;
	    p_link = NULL;
	  }

	  /* Next lattice vector */
	}

	/* Next site in the cube */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_reset_links
 *
 *  Recompute the boundary link vectors and solid/fluid status
 *  of links for an existing particle.
 *
 *  Issues
 *    Non volumetric lambda = 0.5 at the moment.
 *
 *    There is no assumption here about the form of the position update,
 *    so the separation is recomputed. For Euler update, one could just
 *    subtract the current velocity to get the new boundary link vector
 *    from the old one; however, no assumption is preferred.
 *
 ****************************************************************************/

int build_reset_links(cs_t * cs, colloid_t * p_colloid, map_t * map,
		      const lb_model_t * model) {

  int ia;

  colloid_link_t * p_link;
  int         isite[3];
  double      rsite[3];
  double      rsep[3];
  double      r0[3];
  int         offset[3];
  int status;

  double      lambda = 0.5;

  assert(p_colloid);
  assert(map);
  assert(model);

  cs_nlocal_offset(cs, offset);

  p_link = p_colloid->lnk;

  while (p_link) {

    if (p_link->status == LINK_UNUSED) {
      /* Link is not active */
    }
    else {

      /* Compute the separation between the centre of the colloid
       * and the fluid site involved with this link. The position
       * of the outside site is rsite in local coordinates. */

      cs_index_to_ijk(cs, p_link->i, isite);
      for (ia = 0; ia < 3; ia++) {
	rsite[ia] = 1.0*isite[ia];
	r0[ia] = p_colloid->s.r[ia] - 1.0*offset[ia];
      }
      cs_minimum_distance(cs, r0, rsite, rsep);

      p_link->rb[X] = rsep[X] + lambda*model->cv[p_link->p][X];
      p_link->rb[Y] = rsep[Y] + lambda*model->cv[p_link->p][Y];
      p_link->rb[Z] = rsep[Z] + lambda*model->cv[p_link->p][Z];

      map_status(map, p_link->i, &status);

      if (status == MAP_FLUID) {
	int p = p_link->p;
	p_link->status = LINK_FLUID;
	build_link_mean(p_colloid, model->wv[p], model->cv[p], p_link->rb);
      }
      else {
	if (status == MAP_COLLOID) p_link->status = LINK_COLLOID;
	if (status == MAP_BOUNDARY) p_link->status = LINK_BOUNDARY;
      }
    }

    /* Next link */
    p_link = p_link->next;
  }

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
	    if (psi)  psi_colloid_remove_charge(psi, pcnew, index);
	  }
	}

	if (pcold != NULL && pcnew == NULL) {

	  if (!is_halo) {
	    if (psi) psi_colloid_replace_charge(psi, cinfo, pcold, index);
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
 *  build_link_mean
 *
 *  Add a contribution to cbar, rxcbar, and sumw from a given link.
 *
 *****************************************************************************/

static void build_link_mean(colloid_t * pc, double wv, const int8_t cv[3],
			    const double rb[3]) {

  int    ia;
  double c[3];
  double rbxc[3];

  for (ia = 0; ia < 3; ia++) {
    c[ia] = 1.0*cv[ia];
  }

  cross_product(rb, c, rbxc);

  pc->sumw += wv;

  for (ia = 0; ia < 3; ia++) {
    pc->cbar[ia]   += wv*c[ia];
    pc->rxcbar[ia] += wv*rbxc[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  build_colloid_wall_links
 *
 *  This constructs links between colloid and fixed wall.
 *
 *  Some notes.
 *
 *  This is intended for the inbuilt walls, which occupy the halo
 *  regions. Initialisation with coll_recontruct_links will not
 *  identify BOUNDARY links because it does not look into the
 *  halo region. This routine does.
 *
 *  coll_reset_links() examines existing links and sets the
 *  BOUNDARY status as appropriate. See issue 871.
 *
 *****************************************************************************/

int build_colloid_wall_links(cs_t * cs, colloids_info_t * cinfo,
			     colloid_t * p_colloid, map_t * map,
			     const lb_model_t * model) {

  int i_min, i_max, j_min, j_max, k_min, k_max;
  int i, ic, ii, j, jc, jj, k, kc, kk;
  int index0, index1, p;
  int status;
  int ntotal[3];
  int offset[3];

  double largestdimn;
  double lambda = 0.5;
  double r0[3];
  double rsite1[3];
  double rsep[3];

  colloid_t * pcmap = NULL;
  colloid_link_t * p_link;
  colloid_link_t * p_last;

  assert(p_colloid);
  assert(map);

  cs_nlocal(cs, ntotal);
  cs_nlocal_offset(cs, offset);

  p_link = p_colloid->lnk;
  p_last = p_colloid->lnk;
  largestdimn = colloid_principal_radius(&p_colloid->s);

  /* Work out the first unused link */

  while (p_link && p_link->status != LINK_UNUSED) {
    p_last = p_link;
    p_link = p_link->next;
  }

  /* Limits of the cube around the particle. Make sure these are
   * the appropriate lattice nodes... */

  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

  i_min = imax(1,         (int) floor(r0[X] - largestdimn));
  i_max = imin(ntotal[X], (int) ceil (r0[X] + largestdimn));
  j_min = imax(1,         (int) floor(r0[Y] - largestdimn));
  j_max = imin(ntotal[Y], (int) ceil (r0[Y] + largestdimn));
  k_min = imax(1,         (int) floor(r0[Z] - largestdimn));
  k_max = imin(ntotal[Z], (int) ceil (r0[Z] + largestdimn));

  for (i = i_min; i <= i_max; i++) {
    for (j = j_min; j <= j_max; j++) {
      for (k = k_min; k <= k_max; k++) {

	ic = i;
	jc = j;
	kc = k;

	index1 = cs_index(cs, ic, jc, kc);
	colloids_info_map(cinfo, index1, &pcmap);
	if (pcmap != p_colloid) continue;

	rsite1[X] = 1.0*i;
	rsite1[Y] = 1.0*j;
	rsite1[Z] = 1.0*k;
	cs_minimum_distance(cs, r0, rsite1, rsep);

	for (p = 1; p < model->nvel; p++) {

	  /* Find the index of the outside site */

	  ii = ic + model->cv[p][X];
	  jj = jc + model->cv[p][Y];
	  kk = kc + model->cv[p][Z];

	  index0 = cs_index(cs, ii, jj, kk);
	  map_status(map, index0, &status);
	  if (status != MAP_BOUNDARY) continue;

	  /* Add a link */

	  if (p_link) {
	    /* Use existing link (lambda always 0.5 at moment) */

	    p_link->rb[X] = rsep[X] + lambda*model->cv[p][0];
	    p_link->rb[Y] = rsep[Y] + lambda*model->cv[p][1];
	    p_link->rb[Z] = rsep[Z] + lambda*model->cv[p][2];

	    p_link->i = index0;
	    p_link->j = index1;
	    p_link->p = model->nvel - p;
	    p_link->status = LINK_BOUNDARY;

	    /* Next link */
	    p_last = p_link;
	    p_link = p_link->next;
	  }
	  else {
	    /* Add a new link to the end of the list */

	    p_link = colloid_link_allocate();

	    p_link->rb[X] = rsep[X] + lambda*model->cv[p][X];
	    p_link->rb[Y] = rsep[Y] + lambda*model->cv[p][Y];
	    p_link->rb[Z] = rsep[Z] + lambda*model->cv[p][Z];

	    p_link->i = index0;
	    p_link->j = index1;
	    p_link->p = model->nvel - p;
	    p_link->status = LINK_BOUNDARY;

	    /* There must be at least one link in the list. */
	    assert(p_link);

	    p_last->next = p_link;
	    p_link->next = NULL;
	    p_last = p_link;
	    p_link = NULL;
	  }

	  /* Next lattice vector */
	}

	/* Next site in the cube */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_count_links_local
 *
 *  A utility.
 *
 *****************************************************************************/

int build_count_links_local(colloid_t * colloid, int * nlinks) {

  colloid_link_t * pl = NULL;
  int nlink = 0;

  assert(colloid);
  assert(nlinks);

  for (pl = colloid->lnk; pl != NULL; pl = pl->next) {
    nlink += 1;
  }

  *nlinks = nlink;

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

  int p;
  colloid_link_t * pl = NULL;

  assert(colloid);
  assert(sa);
  assert(saf);
  assert(model);

  *sa = 0.0;
  *saf = 0.0;

  for (pl = colloid->lnk; pl != NULL; pl = pl->next) {
    if (pl->status == LINK_UNUSED) continue;
    p = pl->p;
    p = model->cv[p][X]*model->cv[p][X]
      + model->cv[p][Y]*model->cv[p][Y]
      + model->cv[p][Z]*model->cv[p][Z];
    if (p == 1) {
      *sa += 1.0;
      if (pl->status == LINK_FLUID) *saf += 1.0;
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

  int p;

  double value;
  double dq0, dq1;
  double sa_local, saf_local;

  colloid_t * colloid = NULL;
  colloid_link_t * pl = NULL;

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

    for (pl = colloid->lnk; pl != NULL; pl = pl->next) {

      if (pl->status != LINK_FLUID) continue;

      p = pl->p;
      p = model->cv[p][X]*model->cv[p][X]
	+ model->cv[p][Y]*model->cv[p][Y]
	+ model->cv[p][Z]*model->cv[p][Z];

      if (p == 1) {
	/* For charge, do not drop densities below zero. */
	psi_rho(psi, pl->i, 0, &value);
	if ((value + dq0) >= 0.0) {
	  colloid->dq[0] -= dq0;
	  psi_rho_set(psi, pl->i, 0, value + dq0);
	}
	psi_rho(psi, pl->i, 1, &value);
	if ((value + dq1) >=  0.0) {
	  colloid->dq[1] -= dq1;
	  psi_rho_set(psi, pl->i, 1, value + dq1);
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

    cs_limits_t lim = cs_limits_with_halo(map->cs->param->nlocal,  nhalo);
    kernel_3d_t k3d = kernel_3d(map->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(build_update_map_kernel, nblk, ntpb, 0, 0,
                    k3d, map->target, c, h);

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

      double dr[3] = {0}; /* colloid centre -> site */

      if (pc->s.bc != COLLOID_BC_BBL) continue;

      /* Not a minimum image separation as we are potentially checking more
       * than one copy and need to get the right pointer for this site ... */

      dr[X] = r0[X] - (pc->s.r[X] - 1.0*map->cs->param->noffset[X]);
      dr[Y] = r0[Y] - (pc->s.r[Y] - 1.0*map->cs->param->noffset[Y]);
      dr[Z] = r0[Z] - (pc->s.r[Z] - 1.0*map->cs->param->noffset[Z]);

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
            wet[1]        = cosine*wet[1]; /* h */
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
