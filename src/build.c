/*****************************************************************************
 *
 *  build.c
 *
 *  Responsible for the construction of links for particles which
 *  do bounce back on links.
 *
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2006-2023 The University of Edinburgh
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
#include "blue_phase.h"


int build_replace_fluid_local(colloids_info_t * info, colloid_t * pc,
			      int index, lb_t * lb);

int build_replace_q_local(fe_t * fe, colloids_info_t * info, colloid_t * pc, int index,
			  field_t * q);

static int build_remove_fluid(lb_t * lb, int index, colloid_t * pc);
static int build_replace_fluid(lb_t * lb, colloids_info_t * info, int index,
			       colloid_t * pc, map_t * map);
static int build_remove_order_parameter(lb_t * lb, field_t * f, int index,
					colloid_t * pc);
static int build_replace_order_parameter(fe_t * fe, lb_t * lb, colloids_info_t * cinfo,
					 field_t * f, int index,
					 colloid_t * pc, map_t * map);
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

int build_conservation_phi(colloids_info_t * cinfo, field_t * phi,
			   const lb_model_t * model);
int build_conservation_psi(colloids_info_t * cinfo, psi_t * psi,
			   const lb_model_t * model);

/*****************************************************************************
 *
 *  build_update_map
 *
 *  This routine is responsible for setting the solid/fluid status
 *  of all nodes in the presence on colloids. This must be complete
 *  before attempting to build the colloid links.
 *
 ****************************************************************************/

int build_update_map(cs_t * cs, colloids_info_t * cinfo, map_t * map) {

  int nlocal[3];
  int noffset[3];
  int ncell[3];
  int ic, jc, kc;

  int i, j, k;
  int i_min, i_max, j_min, j_max, k_min, k_max;
  int index;
  int nhalo;
  int status;

  colloid_t * p_colloid = NULL;

  double  r0[3];
  double  rsite0[3];
  double  rsep[3];

  double   largestdimn;
  double   cosine, mod;

  /* To set the wetting data in the map, we assume C, H zero at moment */
  int ndata;
  double wet[2];

  assert(cs);
  assert(cinfo);
  assert(map);

  map_ndata(map, &ndata);
  assert(ndata <= 2);

  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);
  cs_nhalo(cs, &nhalo);

  colloids_info_ncell(cinfo, ncell);

  /* First, set any existing colloid sites to fluid */

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	/* This avoids setting BOUNDARY to FLUID */
	index = cs_index(cs, ic, jc, kc);
	map_status(map, index, &status);
	if (status == MAP_COLLOID) {
	  /* Set wetting properties to zero. */
	  map_status_set(map, index, MAP_FLUID);
	  wet[0] = 0.0;
	  wet[1] = 0.0;
	  map_data_set(map, index, wet);
	}

      }
    }
  }

  colloids_info_map_update(cinfo);

  /* Loop through all cells (including the halo cells) */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	/* Set the cell index */

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	/* For each colloid in this cell, check solid/fluid status */

	for ( ; p_colloid; p_colloid = p_colloid->next) {

	  if (p_colloid->s.bc != COLLOID_BC_BBL) continue;

	  /* Set actual position and size of the cube to be checked */

	  largestdimn = colloid_principal_radius(&p_colloid->s);

	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */

	  r0[X] = p_colloid->s.r[X] - 1.0*noffset[X];
	  r0[Y] = p_colloid->s.r[Y] - 1.0*noffset[Y];
	  r0[Z] = p_colloid->s.r[Z] - 1.0*noffset[Z];

	  /* Compute appropriate range of sites that require checks, i.e.,
	   * a cubic box around the centre of the colloid. However, this
	   * should not extend beyond the boundary of the current domain
	   * (but include halos). */

	  i_min = imax(1 - nhalo,         (int) floor(r0[X] - largestdimn));
	  i_max = imin(nlocal[X] + nhalo, (int) ceil (r0[X] + largestdimn));
	  j_min = imax(1 - nhalo,         (int) floor(r0[Y] - largestdimn));
	  j_max = imin(nlocal[Y] + nhalo, (int) ceil (r0[Y] + largestdimn));
	  k_min = imax(1 - nhalo,         (int) floor(r0[Z] - largestdimn));
	  k_max = imin(nlocal[Z] + nhalo, (int) ceil (r0[Z] + largestdimn));

	  /* Check each site to see whether it is inside or not */

	  for (i = i_min; i <= i_max; i++)
	    for (j = j_min; j <= j_max; j++)
	      for (k = k_min; k <= k_max; k++) {

		/* rsite0 is the coordinate position of the site */

		rsite0[X] = 1.0*i;
		rsite0[Y] = 1.0*j;
		rsite0[Z] = 1.0*k;
		cs_minimum_distance(cs, rsite0, r0, rsep);

		/* Are we inside? */

		if (colloid_r_inside(&p_colloid->s, rsep)) {

		  /* Set index */
		  index = cs_index(cs, i, j, k);

		  colloids_info_map_set(cinfo, index, p_colloid);
		  map_status_set(map, index, MAP_COLLOID);

		  /* Janus particles have h = h_0 cos (theta)
		   * with s[3] pointing to the 'north pole' */

		  cosine = 1.0;

		  if (p_colloid->s.attr & COLLOID_ATTR_JANUS) {
		    mod = modulus(rsep);
		    if (mod > 0.0) {
		      cosine = dot_product(p_colloid->s.s, rsep)/mod;
		    }
		  }

		  wet[0] = p_colloid->s.c;
		  wet[1] = cosine*p_colloid->s.h;

		  map_data_set(map, index, wet);
		}
		/* Next site */
	      }

	  /* Next colloid */
	}

	/* Next cell */
      }
    }
  }

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
			 field_t * phi,
			 field_t * p, field_t * q, psi_t * psi, map_t * map) {

  int ic, jc, kc, index;
  int is_halo;
  int nlocal[3];
  int nhalo;
  colloid_t * pcold;
  colloid_t * pcnew;

  assert(lb);
  assert(cinfo);

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

	  pcnew->s.rebuild = 1;

	  if (!is_halo) {
	    build_remove_fluid(lb, index, pcnew);
	    if (phi) build_remove_order_parameter(lb, phi, index, pcnew);
	    if (psi)  psi_colloid_remove_charge(psi, pcnew, index);
	  }
	}

	if (pcold != NULL && pcnew == NULL) {

	  pcold->s.rebuild = 1;

	  if (!is_halo) {
	    build_replace_fluid(lb, cinfo, index, pcold, map);
	    if (phi) build_replace_order_parameter(fe, lb, cinfo, phi, index, pcold, map);
	    if (p) build_replace_order_parameter(fe, lb, cinfo, p, index, pcold, map);
	    if (q) build_replace_order_parameter(fe, lb, cinfo, q, index, pcold, map);
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
 *  build_bbl_rebuild_flag
 *
 *  Looks for changes in the status map and sets the rebuild flag (only).
 *
 *  Not currently used.
 *
 *****************************************************************************/

int build_bbl_rebuild_flag(cs_t * cs, colloids_info_t * cinfo) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nhalo;
  colloid_t * pcold;
  colloid_t * pcnew;

  assert(cs);
  assert(cinfo);

  cs_nhalo(cs, &nhalo);
  cs_nlocal(cs, nlocal);

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = cs_index(cs, ic, jc, kc);

	colloids_info_map_old(cinfo, index, &pcold);
	colloids_info_map(cinfo, index, &pcnew);

	if (pcold == NULL && pcnew != NULL) pcnew->s.rebuild = 1;
	if (pcold != NULL && pcnew == NULL) pcold->s.rebuild = 1;
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
 *  build_remove_order_parameter
 *
 *  Conserved order parameters only.
 *
 *  Remove order parameter(s) at the site inode. The old site information
 *  can be lost inside the particle, but we must record the correction.
 *
 *  A rather cross-cutting routine.
 *
 *****************************************************************************/

static int build_remove_order_parameter(lb_t * lb, field_t * f, int index,
					colloid_t * pc) {
  int ndist;
  double phi;
  double phi0;
  physics_t * phys = NULL;

  assert(f);
  assert(lb);
  assert(pc);

  physics_ref(&phys);
  physics_phi0(phys, &phi0);
  lb_ndist(lb, &ndist);

  if (ndist == 2) {
    lb_0th_moment(lb, index, LB_PHI, &phi);
  }
  else {
    field_scalar(f, index, &phi);
  }

  pc->s.deltaphi += (phi - phi0);

  return 0;
}

/*****************************************************************************
 *
 *  build_replace_fluid
 *
 *  Replace the distributions when a fluid site (index) is exposed.
 *  This gives rise to corrections on the particle force and torque.
 *
 *****************************************************************************/

static int build_replace_fluid(lb_t * lb, colloids_info_t * cinfo, int index,
			       colloid_t * p_colloid, map_t * map) {

  int indexn, p, pdash;
  int ia;
  int status;
  int nweight;
  int ib[3];
  int noffset[3];

  double newrho;
  double weight;
  double g[3];                /* Change in momentum */
  double r0[3];               /* Centre of colloid in local coordinates */
  double rb[3];               /* Boundary vector at site index */
  double rtmp[3];
  double newf[NVEL];          /* Replacement distributions */
  double rho0;

  physics_t * phys = NULL;
  colloid_t * pc = NULL;

  assert(lb);
  assert(p_colloid);
  assert(map);

  cs_nlocal_offset(lb->cs, noffset);
  cs_index_to_ijk(lb->cs, index, ib);

  physics_ref(&phys);
  physics_rho0(phys, &rho0);

  newrho = 0.0;
  weight = 0.0;
  nweight = 0;

  for (ia = 0; ia < 3; ia++) {
    g[ia] = 0.0;
  }

  /* Check the surrounding sites that were linked to inode,
   * and accumulate a (weighted) average distribution. */

  for (p = 0; p < lb->model.nvel; p++) {
    newf[p] = 0.0;
  }

  for (p = 1; p < lb->model.nvel; p++) {

    indexn = cs_index(lb->cs, ib[X] + lb->model.cv[p][X],
		              ib[Y] + lb->model.cv[p][Y],
		              ib[Z] + lb->model.cv[p][Z]);

    /* Site must have been fluid before position update */

    colloids_info_map_old(cinfo, indexn, &pc);
    if (pc) continue;
    map_status(map, indexn, &status);
    if (status == MAP_BOUNDARY) continue;

    for (pdash = 0; pdash < lb->model.nvel; pdash++) {
      lb_f(lb, indexn, pdash, 0, rtmp);
      newf[pdash] += lb->model.wv[p]*rtmp[0];
    }
    weight += lb->model.wv[p];
    nweight += 1;
  }

  /* Set new fluid distributions */

  if (nweight == 0) {
    /* Cannot interpolate: fall back to local replacement */
    build_replace_fluid_local(cinfo, p_colloid, index, lb);
  }
  else {

    weight = 1.0/weight;

    for (p = 0; p < lb->model.nvel; p++) {
      newf[p] *= weight;
      lb_f_set(lb, index, p, 0, newf[p]);

      /* ... and remember the new fluid properties */
      newrho += newf[p];

      /* minus sign is appropriate for upcoming ...
	 ... correction to colloid momentum */

      for (ia = 0; ia < 3; ia++) {
	g[ia] -= newf[p]*lb->model.cv[p][ia];
      }
    }

    /* Set corrections for excess mass and momentum. For the
     * correction to the torque, we need the appropriate
     * boundary vector rb */

    p_colloid->deltam += (newrho - rho0);

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
 *  build_replace_order_parameter
 *
 *  Replace the order parameter(s) at a newly exposed site (index).
 *
 *****************************************************************************/

static int build_replace_order_parameter(fe_t * fe, lb_t * lb,
					 colloids_info_t * cinfo,
					 field_t * f, int index,
					 colloid_t * pc, map_t * map) {
  int indexn, n, p, pdash;
  int status;
  int ri[3];
  int nf;
  int ndist;
  int nweight;

  double g;
  double weight = 0.0;
  double newg[NVEL];
  double phi[NQAB];
  double qs[NQAB];
  double phi0;

  physics_t * phys = NULL;
  colloid_t * pcmap = NULL;

  assert(map);
  assert(lb);
  lb_ndist(lb, &ndist);

  field_nf(f, &nf);
  assert(nf <= NQAB);

  cs_index_to_ijk(lb->cs, index, ri);
  physics_ref(&phys);
  physics_phi0(phys, &phi0);

  /* Check the surrounding sites that were linked to inode,
   * and accumulate a (weighted) average distribution. */

  for (p = 0; p < lb->model.nvel; p++) {
    newg[p] = 0.0;
  }

  if (ndist == 2) {

    /* Reset the distribution (distribution index 1) */

    for (p = 1; p < lb->model.nvel; p++) {

      indexn = cs_index(lb->cs, ri[X] + lb->model.cv[p][X],
			        ri[Y] + lb->model.cv[p][Y],
			        ri[Z] + lb->model.cv[p][Z]);

      /* Site must have been fluid before position update */

      /* TODO Could be done with MAP_STATUS ? */
      colloids_info_map_old(cinfo, indexn, &pcmap);
      if (pcmap) continue;
      map_status(map, indexn, &status);
      if (status == MAP_BOUNDARY) continue;

      for (pdash = 0; pdash < lb->model.nvel; pdash++) {
	lb_f(lb, indexn, pdash, LB_PHI, &g);
	newg[pdash] += lb->model.wv[p]*g;
      }
      weight += lb->model.wv[p];
    }

    /* Set new fluid distributions */

    if (weight == 0.0) {
      /* No neighbouring fluid: as there's no information, we
       * fall back to the value that is currently stored on the
       * lattice. This is not entirely unreasonable, as it may
       * reflect what is nearby, or initial conditions. It could
       * also be set as a contingency in a separate step. */
      field_scalar(f, index, newg);
      weight = 1.0;
    }

    weight = 1.0/weight;
    phi[0] = 0.0;

    for (p = 0; p < lb->model.nvel; p++) {
      newg[p] *= weight;
      lb_f_set(lb, index, p, LB_PHI, newg[p]);

      /* ... and remember the new fluid properties */
      phi[0] += newg[p];
    }
  }
  else {

    /* Replace field value(s), based on same average */

    nweight = 0.0;
    for (n = 0; n < nf; n++) {
      phi[n] = 0.0;
    }

    for (p = 1; p < lb->model.nvel; p++) {

      indexn = cs_index(lb->cs, ri[X] + lb->model.cv[p][X],
			        ri[Y] + lb->model.cv[p][Y],
		                ri[Z] + lb->model.cv[p][Z]);

      /* Site must have been fluid before position update */

      colloids_info_map_old(cinfo, indexn, &pcmap);
      if (pcmap) continue;
      map_status(map, indexn, &status);
      if (status == MAP_BOUNDARY) continue;

      field_scalar_array(f, indexn, qs);
      for (n = 0; n < nf; n++) {
	phi[n] += lb->model.wv[p]*qs[n];
      }
      weight += lb->model.wv[p];
      nweight += 1;
    }
    if (nweight == 0) {
      /* No information. For phi, use existing (solid) value. */
      if (fe->id == FE_LC) build_replace_q_local(fe, cinfo, pc, index, f);
      if (fe->id == FE_SYMMETRIC) field_scalar(f, index, phi);
    }
    else {
      weight = 1.0 / weight;
      for (n = 0; n < nf; n++) {
	phi[n] *= weight;
      }
      field_scalar_array_set(f, index, phi);
    }
  }

  /* Set corrections arising from change in conserved order parameter,
   * which we assume means nf == 1 */

  pc->s.deltaphi -= (phi[0] - phi0);

  return 0;
}

/*****************************************************************************
 *
 *  build_replace_q_local
 *
 *  ASSUME NORMAL ANCHORING AMPLITUDE = 1/3
 *
 *****************************************************************************/

int build_replace_q_local(fe_t * fe, colloids_info_t * info, colloid_t * pc,
			  int index, field_t * q) {

  int ia, ib;
  double rb[3], rbp[3], rhat[3];
  double rbmod, rhat_dot_rb;
  double qnew[3][3];

  double amplitude = (1.0/3.0);

  fe_lc_t * fe_lc = (fe_lc_t *) fe;
  fe_lc_param_t * lc_param = fe_lc->param;

  KRONECKER_DELTA_CHAR(d);

  assert(fe);
  assert(info);
  assert(pc);
  assert(q);

  fe_lc_amplitude_compute(lc_param, &amplitude);

  /* For normal anchoring we determine the radial unit vector rb */

  colloid_rb(info, pc, index, rb);

  if (pc->s.shape == COLLOID_SHAPE_ELLIPSOID) {
    /* Compute correct spheroid normal ... */
    int isphere = util_ellipsoid_is_sphere(pc->s.elabc);
    if (!isphere) {
      double posvector[3] = {0};
      util_vector_copy(3, rb, posvector);
      util_spheroid_surface_normal(pc->s.elabc, pc->s.m, posvector, rb);
    }
  }

  /* Make sure we have a unit vector */
  rbmod = 1.0/sqrt(rb[X]*rb[X] + rb[Y]*rb[Y] + rb[Z]*rb[Z]);
  rb[0] *= rbmod;
  rb[1] *= rbmod;
  rb[2] *= rbmod;


  /* For planar degenerate anchoring we subtract the projection of a
     randomly oriented unit vector on rb and renormalise the result   */

  if (lc_param->coll.type == LC_ANCHORING_PLANAR) {

    util_random_unit_vector(&pc->s.rng, rhat);

    rhat_dot_rb = dot_product(rhat,rb);
    rbp[0] = rhat[0] - rhat_dot_rb*rb[0];
    rbp[1] = rhat[1] - rhat_dot_rb*rb[1];
    rbp[2] = rhat[2] - rhat_dot_rb*rb[2];

    rbmod = 1.0/sqrt(rbp[X]*rbp[X] + rbp[Y]*rbp[Y] + rbp[Z]*rbp[Z]);
    rb[0] = rbmod * rbp[0];
    rb[1] = rbmod * rbp[1];
    rb[2] = rbmod * rbp[2];

  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      qnew[ia][ib] = 0.5*amplitude*(3.0*rb[ia]*rb[ib] - d[ia][ib]);
    }
  }

  field_tensor_set(q, index, qnew);

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

  if (phi) build_conservation_phi(cinfo, phi, model);
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
 *  build_conservation_phi
 *
 *  To be run immediately following remove/replace so that there is no
 *  change in mean composition.
 *
 *  A call to colloid_sums_halo(cinfo, COLLOID_SUM_CONSERVATION) before
 *  we reach this point is required so that all parts of distributed
 *  colloids see the same deltaphi.
 *
 *****************************************************************************/

int build_conservation_phi(colloids_info_t * cinfo, field_t * phi,
			   const lb_model_t * model) {

  int p;

  double value;
  double dphi;

  colloid_t * colloid = NULL;
  colloid_link_t * pl = NULL;

  assert(cinfo);
  assert(phi);

  colloids_info_all_head(cinfo, &colloid);

  for (; colloid != NULL; colloid = colloid->nextall) {

    /* Add any contribution form previous steps (all copies);
     * work out what should be put back. */

    dphi = colloid->s.deltaphi / colloid->s.saf;
    if (dphi == 0.0) continue;

    for (pl = colloid->lnk; pl != NULL; pl = pl->next) {

      if (pl->status != LINK_FLUID) continue;

      p = pl->p;
      p = model->cv[p][X]*model->cv[p][X]
	+ model->cv[p][Y]*model->cv[p][Y]
	+ model->cv[p][Z]*model->cv[p][Z];

      if (p == 1) {
	/* Replace */
	field_scalar(phi, pl->i, &value);
	field_scalar_set(phi, pl->i, value + dphi);
      }
    }

    /* We may now reset deltaphi to zero. */
    colloid->s.deltaphi = 0.0;
  }

  return 0;
}
