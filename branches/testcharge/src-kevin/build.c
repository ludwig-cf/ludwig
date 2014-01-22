/*****************************************************************************
 *
 *  build.c
 *
 *  Responsible for the construction of links for particles which
 *  do bounce back on links.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
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
#include "model.h"

#include "psi_colloid.h"
#include "util.h"
#include "wall.h"
#include "build.h"


static int build_remove_order_parameter(field_t * f, int index,
					colloid_t * pc);
static int build_reconstruct_links(colloids_info_t * cinfo, colloid_t * pc,
				   map_t * map);
static int build_replace_fluid(colloids_info_t * info, int index, colloid_t *);
static int build_replace_order_parameter(colloids_info_t * cinfo, field_t * f,
					 int index,
					 colloid_t * pc);
static int build_colloid_wall_links(colloids_info_t * cinfo, colloid_t * pc,
				    map_t * map);

static void build_link_mean(colloid_t * pc, int p, const double rb[3]);
static void build_virtual_distribution_set(int index, int p,
					   const double u[3]);
static int build_reset_links(colloid_t * pc, map_t * map);
static int build_remove_fluid(int index, colloid_t *);

/*****************************************************************************
 *
 *  build_update_map
 *
 *  This routine is responsible for setting the solid/fluid status
 *  of all nodes in the presence on colloids. This must be complete
 *  before attempting to build the colloid links.
 *
 *  Issues:
 *
 ****************************************************************************/

int build_update_map(colloids_info_t * cinfo, map_t * map) {

  int nlocal[3];
  int noffset[3];
  int ncell[3];
  int ic, jc, kc;

  int nsites;
  int i, j, k;
  int i_min, i_max, j_min, j_max, k_min, k_max;
  int index;
  int nhalo;
  int status;

  colloid_t * p_colloid;

  double  r0[3];
  double  rsite0[3];
  double  rsep[3];

  double   radius, rsq;
  double   cosine, mod;

  /* To set the wetting data in the map, we assume C, H zero at moment */
  int ndata;
  double wet[2];

  assert(cinfo);
  assert(map);

  map_ndata(map, &ndata);
  assert(ndata <= 2);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  nhalo = coords_nhalo();

  colloids_info_ncell(cinfo, ncell);

  /* First, set any existing colloid sites to fluid */

  nsites = coords_nsites();

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	/* This avoids setting BOUNDARY to FLUID */
	index = coords_index(ic, jc, kc);
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

	while (p_colloid != NULL) {

	  /* Set actual position and radius */

	  radius = p_colloid->s.a0;
	  rsq    = radius*radius;

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

	  i_min = imax(1 - nhalo,         (int) floor(r0[X] - radius));
	  i_max = imin(nlocal[X] + nhalo, (int) ceil (r0[X] + radius));
	  j_min = imax(1 - nhalo,         (int) floor(r0[Y] - radius));
	  j_max = imin(nlocal[Y] + nhalo, (int) ceil (r0[Y] + radius));
	  k_min = imax(1 - nhalo,         (int) floor(r0[Z] - radius));
	  k_max = imin(nlocal[Z] + nhalo, (int) ceil (r0[Z] + radius));

	  /* Check each site to see whether it is inside or not */

	  for (i = i_min; i <= i_max; i++)
	    for (j = j_min; j <= j_max; j++)
	      for (k = k_min; k <= k_max; k++) {

		/* rsite0 is the coordinate position of the site */

		rsite0[X] = 1.0*i;
		rsite0[Y] = 1.0*j;
		rsite0[Z] = 1.0*k;
		coords_minimum_distance(rsite0, r0, rsep);

		/* Are we inside? */

		if (dot_product(rsep, rsep) < rsq) {

		  /* Set index */
		  index = coords_index(i, j, k);

		  colloids_info_map_set(cinfo, index, p_colloid);
		  map_status_set(map, index, MAP_COLLOID);

		  /* Janus particles have h = h_0 cos (theta)
		   * with s[3] pointing to the 'north pole' */

		  cosine = 1.0;
		  if (p_colloid->s.type == COLLOID_TYPE_JANUS) {
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
	  p_colloid = p_colloid->next;
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

int build_update_links(colloids_info_t * cinfo, map_t * map) {

  int ia;
  int ic, jc, kc;
  int ncell[3];
  int nhalo;
  colloid_t * pc;

  assert(cinfo);
  assert(map);

  colloids_info_ncell(cinfo, ncell);
  colloids_info_nhalo(cinfo, &nhalo);

  for (ic = 1 - nhalo; ic <= ncell[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= ncell[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= ncell[Z] + nhalo; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {

	  pc->sumw   = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    pc->cbar[ia] = 0.0;
	    pc->rxcbar[ia] = 0.0;
	  }

	  if (pc->s.rebuild) {
	    /* The shape has changed, so need to reconstruct */
	    build_reconstruct_links(cinfo, pc, map);
	    if (wall_present()) build_colloid_wall_links(cinfo, pc, map);
	  }
	  else {
	    /* Shape unchanged, so just reset existing links */
	    build_reset_links(pc, map);
	  }

	  build_count_faces_local(pc, &pc->s.sa, &pc->s.saf);

	  /* Next colloid */

	  pc->s.rebuild = 0;
	  pc = pc->next;
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
 *  Issues
 *
 ****************************************************************************/

int build_reconstruct_links(colloids_info_t * cinfo, colloid_t * p_colloid,
			    map_t * map) {

  colloid_link_t * p_link;
  colloid_link_t * p_last;
  int i_min, i_max, j_min, j_max, k_min, k_max;
  int i, ic, ii, j, jc, jj, k, kc, kk;
  int index0, index1, p;
  int status1;

  double       radius;
  double       lambda = 0.5;
  double      rsite1[3];
  double      rsep[3];
  double      r0[3];
  int ntotal[3];
  int offset[3];

  colloid_t * pc = NULL;

  coords_nlocal(ntotal);
  coords_nlocal_offset(offset);

  p_link = p_colloid->lnk;
  p_last = p_link;
  radius = p_colloid->s.a0;

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

  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

  i_min = imax(1,         (int) floor(r0[X] - radius));
  i_max = imin(ntotal[X], (int) ceil (r0[X] + radius));
  j_min = imax(1,         (int) floor(r0[Y] - radius));
  j_max = imin(ntotal[Y], (int) ceil (r0[Y] + radius));
  k_min = imax(1,         (int) floor(r0[Z] - radius));
  k_max = imin(ntotal[Z], (int) ceil (r0[Z] + radius));

  for (i = i_min; i <= i_max; i++) {
    for (j = j_min; j <= j_max; j++) {
      for (k = k_min; k <= k_max; k++) {

	ic = i;
	jc = j;
	kc = k;

	index1 = coords_index(ic, jc, kc);
	colloids_info_map(cinfo, index1, &pc);
	if (pc == p_colloid) continue;

	rsite1[X] = 1.0*i;
	rsite1[Y] = 1.0*j;
	rsite1[Z] = 1.0*k;
	coords_minimum_distance(r0, rsite1, rsep);
	map_status(map, index1, &status1);

	/* Index 1 is outside, so cycle through the lattice vectors
	 * to determine if the end is inside, and so requires a link */

	for (p = 1; p < NVEL; p++) {

	  /* Find the index of the inside site */

	  ii = ic + cv[p][X];
	  jj = jc + cv[p][Y];
	  kk = kc + cv[p][Z];

	  index0 = coords_index(ii, jj, kk);
	  colloids_info_map(cinfo, index0, &pc);
	  if (pc != p_colloid) continue;

	  /* Index 0 is inside, so now add a link*/

	  if (p_link) {
	    /* Use existing link (lambda always 0.5 at moment) */

	    p_link->rb[X] = rsep[X] + lambda*cv[p][X];
	    p_link->rb[Y] = rsep[Y] + lambda*cv[p][Y];
	    p_link->rb[Z] = rsep[Z] + lambda*cv[p][Z];

	    p_link->i = index1;
	    p_link->j = index0;
	    p_link->p = p;

	    if (status1 == MAP_FLUID) {
	      p_link->status = LINK_FLUID;
	      build_link_mean(p_colloid, p, p_link->rb);
	    }
	    else {
	      double ub[3];
	      double wxrb[3];
	      p_link->status = LINK_COLLOID;

	      cross_product(p_colloid->s.w, p_link->rb, wxrb);
	      ub[X] = p_colloid->s.v[X] + wxrb[X];
	      ub[Y] = p_colloid->s.v[Y] + wxrb[Y];
	      ub[Z] = p_colloid->s.v[Z] + wxrb[Z];
	      build_virtual_distribution_set(p_link->j, p_link->p, ub);
	    }

	    /* Next link */
	    p_last = p_link;
	    p_link = p_link->next;

	  }
	  else {
	    /* Add a new link to the end of the list */

	    p_link = colloid_link_allocate();

	    p_link->rb[X] = rsep[X] + lambda*cv[p][X];
	    p_link->rb[Y] = rsep[Y] + lambda*cv[p][Y];
	    p_link->rb[Z] = rsep[Z] + lambda*cv[p][Z];

	    p_link->i = index1;
	    p_link->j = index0;
	    p_link->p = p;

	    if (status1 == MAP_FLUID) {
	      p_link->status = LINK_FLUID;
	      build_link_mean(p_colloid, p, p_link->rb);
	    }
	    else {
	      double ub[3];
	      double wxrb[3];
	      p_link->status = LINK_COLLOID;

	      cross_product(p_colloid->s.w, p_link->rb, wxrb);
	      ub[X] = p_colloid->s.v[X] + wxrb[X];
	      ub[Y] = p_colloid->s.v[Y] + wxrb[Y];
	      ub[Z] = p_colloid->s.v[Z] + wxrb[Z];
	      build_virtual_distribution_set(p_link->j, p_link->p, ub);
	    }

	    if (p_colloid->lnk == NULL) {
	      /* Remember to attach the head of the list */
	      p_colloid->lnk = p_link;
	      p_last = p_link;
	    }
	    else {
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
 *    from the old one; however, no assumption is prefered.
 *
 *    Note that setting virtual fluid properties for boundary sites
 *    is done elseehere.
 *
 ****************************************************************************/

int build_reset_links(colloid_t * p_colloid, map_t * map) {

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

  coords_nlocal_offset(offset);

  p_link = p_colloid->lnk;

  while (p_link) {

    if (p_link->status == LINK_UNUSED) {
      /* Link is not active */
    }
    else {

      /* Compute the separation between the centre of the colloid
       * and the fluid site involved with this link. The position
       * of the outside site is rsite in local coordinates. */

      coords_index_to_ijk(p_link->i, isite);
      for (ia = 0; ia < 3; ia++) {
	rsite[ia] = 1.0*isite[ia];
	r0[ia] = p_colloid->s.r[ia] - 1.0*offset[ia];
      }
      coords_minimum_distance(r0, rsite, rsep);

      p_link->rb[X] = rsep[X] + lambda*cv[p_link->p][X];
      p_link->rb[Y] = rsep[Y] + lambda*cv[p_link->p][Y];
      p_link->rb[Z] = rsep[Z] + lambda*cv[p_link->p][Z];

      map_status(map, p_link->i, &status);

      if (status == MAP_FLUID) {
	p_link->status = LINK_FLUID;
	build_link_mean(p_colloid, p_link->p, p_link->rb);
      }
      else {
	double ub[3];
	double wxrb[3];

	if (status == MAP_COLLOID) p_link->status = LINK_COLLOID;
	if (status == MAP_BOUNDARY) p_link->status = LINK_BOUNDARY;

	cross_product(p_colloid->s.w, p_link->rb, wxrb);
	ub[X] = p_colloid->s.v[X] + wxrb[X];
	ub[Y] = p_colloid->s.v[Y] + wxrb[Y];
	ub[Z] = p_colloid->s.v[Z] + wxrb[Z];
	build_virtual_distribution_set(p_link->j, p_link->p, ub);
      }
    }

    /* Next link */
    p_link = p_link->next;
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_remove_or_replace_fluid
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
 *****************************************************************************/

int build_remove_replace(colloids_info_t * cinfo, field_t * phi, field_t * p,
			 field_t * q, psi_t * psi) {

  int ic, jc, kc, index;
  int is_halo;
  int nlocal[3];
  int nhalo;
  colloid_t * pcold;
  colloid_t * pcnew;

  assert(cinfo);

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = coords_index(ic, jc, kc);

	colloids_info_map_old(cinfo, index, &pcold);
	colloids_info_map(cinfo, index, &pcnew);

	is_halo = (ic < 1 || jc < 1 || kc < 1 ||
		   ic > nlocal[X] || jc > nlocal[Y] || kc > nlocal[Z]);

	if (pcold == NULL && pcnew != NULL) {

	  pcnew->s.rebuild = 1;

	  if (!is_halo) {
	    build_remove_fluid(index, pcnew);
	    if (phi) build_remove_order_parameter(phi, index, pcnew);
	    if (psi)  psi_colloid_remove_charge(psi, pcnew, index);
	  }
	}

	if (pcold != NULL && pcnew == NULL) {

	  pcold->s.rebuild = 1;

	  if (!is_halo) {
	    build_replace_fluid(cinfo, index, pcold);
	    if (phi) build_replace_order_parameter(cinfo, phi, index, pcold);
	    if (p) build_replace_order_parameter(cinfo, p, index, pcold);
	    if (q) build_replace_order_parameter(cinfo, q, index, pcold);
	    if (psi) psi_colloid_replace_charge(psi, cinfo, pcold, index);
	  }
	}

      }
    }
  }

  return 0;
}

/******************************************************************************
 *
 *  build_remove_fluid
 *
 *  Remove denisty, momentum at site inode.
 *
 *  Corrections to the mass, force, and torque updates to the relevant
 *  colloid are required.
 *
 *  We don't care about the 'swallowed' distribution information
 *  associated with the old fluid.
 *
 *****************************************************************************/

static int build_remove_fluid(int index, colloid_t * p_colloid) {

  int    ia;
  int    ib[3];
  int    noffset[3];

  double rho;             /* density of removed fluid */
  double g[3];            /* momentum of removed fluid */
  double r0[3];           /* Local coords of colloid centre */
  double rb[3];           /* Boundary vector at lattice site index */
  double rtmp[3];
  double rho0;

  coords_nlocal_offset(noffset);
  coords_index_to_ijk(index, ib);

  physics_rho0(&rho0);

  /* Get the properties of the old fluid at inode */

  rho = distribution_zeroth_moment(index, 0);
  distribution_first_moment(index, 0, g);

  /* Set the corrections for colloid motion. This requires
   * the local boundary vector rb for the torque */

  p_colloid->deltam -= (rho - rho0);

  for (ia = 0; ia < 3; ia++) {
    p_colloid->f0[ia] += g[ia];
    r0[ia] = p_colloid->s.r[ia] - 1.0*noffset[ia];
    rtmp[ia] = 1.0*ib[ia];
  }

  coords_minimum_distance(r0, rtmp, rb);
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
 *****************************************************************************/

static int build_remove_order_parameter(field_t * f, int index,
					colloid_t * pc) {
  double phi;
  double phi0;

  assert(f);
  assert(pc);

  physics_phi0(&phi0);

  if (distribution_ndist() == 2) {
    phi = distribution_zeroth_moment(index, 1);
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

static int build_replace_fluid(colloids_info_t * cinfo, int index,
			       colloid_t * p_colloid) {

  int indexn, p, pdash;
  int ia;
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

  colloid_t * pc = NULL;

  assert(p_colloid);

  coords_nlocal_offset(noffset);
  coords_index_to_ijk(index, ib);

  physics_rho0(&rho0);

  newrho = 0.0;
  weight = 0.0;

  for (ia = 0; ia < 3; ia++) {
    g[ia] = 0.0;
  }

  /* Check the surrounding sites that were linked to inode,
   * and accumulate a (weighted) average distribution. */

  for (p = 0; p < NVEL; p++) {
    newf[p] = 0.0;
  }

  for (p = 1; p < NVEL; p++) {

    indexn = coords_index(ib[X] + cv[p][X], ib[Y] + cv[p][Y],
			  ib[Z] + cv[p][Z]);

    /* Site must have been fluid before position update */

    colloids_info_map_old(cinfo, indexn, &pc);
    if (pc) continue;

    for (pdash = 0; pdash < NVEL; pdash++) {
      newf[pdash] += wv[p]*distribution_f(indexn, pdash, 0);
    }
    weight += wv[p];
  }

  /* Set new fluid distributions */

  weight = 1.0/weight;

  for (p = 0; p < NVEL; p++) {
    newf[p] *= weight;
    distribution_f_set(index, p, 0, newf[p]);

    /* ... and remember the new fluid properties */
    newrho += newf[p];

    /* minus sign is approprite for upcoming ...
       ... correction to colloid momentum */

    for (ia = 0; ia < 3; ia++) {
      g[ia] -= newf[p]*cv[p][ia];
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

  coords_minimum_distance(r0, rtmp, rb);
  cross_product(rb, g, rtmp);

  for (ia = 0; ia < 3; ia++) {
    p_colloid->t0[ia] += rtmp[ia];
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

static int build_replace_order_parameter(colloids_info_t * cinfo,
					 field_t * f, int index,
					 colloid_t * pc) {
  int indexn, n, p, pdash;
  int ri[3];
  int nf;

  double weight = 0.0;
  double newg[NVEL];
  double phi[NQAB];
  double qs[NQAB];
  double phi0;

  colloid_t * pcmap = NULL;

  field_nf(f, &nf);
  assert(nf <= NQAB);

  coords_index_to_ijk(index, ri);
  physics_phi0(&phi0);

  /* Check the surrounding sites that were linked to inode,
   * and accumulate a (weighted) average distribution. */

  for (p = 0; p < NVEL; p++) {
    newg[p] = 0.0;
  }

  if (distribution_ndist() == 2) {

    /* Reset the distribution (distribution index 1) */

    for (p = 1; p < NVEL; p++) {

      indexn = coords_index(ri[X] + cv[p][X], ri[Y] + cv[p][Y],
			      ri[Z] + cv[p][Z]);

      /* Site must have been fluid before position update */

      /* TODO Could be done with MAP_STATUS ? */
      colloids_info_map_old(cinfo, indexn, &pcmap);
      if (pcmap) continue;

      for (pdash = 0; pdash < NVEL; pdash++) {
	newg[pdash] += wv[p]*distribution_f(indexn, pdash, 1);
      }
      weight += wv[p];
    }

    /* Set new fluid distributions */

    weight = 1.0/weight;
    phi[0] = 0.0;

    for (p = 0; p < NVEL; p++) {
      newg[p] *= weight;
      distribution_f_set(index, p, 1, newg[p]);

      /* ... and remember the new fluid properties */
      phi[0] += newg[p];
    }
  }
  else {

    /* Replace field value(s), based on same average */

    for (n = 0; n < nf; n++) {
      phi[n] = 0.0;
    }

    for (p = 1; p < NVEL; p++) {

      indexn = coords_index(ri[X] + cv[p][X], ri[Y] + cv[p][Y],
			      ri[Z] + cv[p][Z]);

      /* Site must have been fluid before position update */

      colloids_info_map_old(cinfo, indexn, &pcmap);
      if (pcmap) continue;

      field_scalar_array(f, indexn, qs);
      for (n = 0; n < nf; n++) {
	phi[n] += wv[p]*qs[n];
      }
      weight += wv[p];
    }

    weight = 1.0 / weight;
    for (n = 0; n < nf; n++) {
      phi[n] *= weight;
    }
    field_scalar_array_set(f, index, phi);
  }

  /* Set corrections arising from change in conserved order parameter,
   * which we assume means nf == 1 */

  pc->s.deltaphi -= (phi[0] - phi0);

  return 0;
}

/****************************************************************************
 *
 *  build_virtual_distribution_set
 *
 *  Set f_p at inode to an equilibrium value for a given velocity.
 *  rho = 1.
 *
 ****************************************************************************/

static void build_virtual_distribution_set(int index, int p,
					   const double u[3]) {
  double udotc;

  udotc = u[X]*cv[p][X] + u[Y]*cv[p][Y] + u[Z]*cv[p][Z];
  distribution_f_set(index, p, 0, wv[p]*(1.0 + 3.0*udotc));

  return;
}

/*****************************************************************************
 *
 *  build_link_mean
 *
 *  Add a contribution to cbar, rxcbar, and sumw from a given link.
 *
 *****************************************************************************/

static void build_link_mean(colloid_t * p_colloid, int p, const double rb[3]) {

  int    ia;
  double c[3];
  double rbxc[3];

  for (ia = 0; ia < 3; ia++) {
    c[ia] = 1.0*cv[p][ia];
  }

  cross_product(rb, c, rbxc);

  p_colloid->sumw += wv[p];

  for (ia = 0; ia < 3; ia++) {
    p_colloid->cbar[ia]   += wv[p]*c[ia];
    p_colloid->rxcbar[ia] += wv[p]*rbxc[ia];
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
 *  indentify BOUNDARY links because it does not look into the
 *  halo region. This routine does.
 *
 *  coll_reset_links() examines exsiting links and sets the
 *  BOUNDARY status as appropriate. See issue 871.
 *
 *****************************************************************************/

int build_colloid_wall_links(colloids_info_t * cinfo, colloid_t * p_colloid,
			     map_t * map) {

  int i_min, i_max, j_min, j_max, k_min, k_max;
  int i, ic, ii, j, jc, jj, k, kc, kk;
  int index0, index1, p;
  int status;
  int ntotal[3];
  int offset[3];

  double radius;
  double lambda = 0.5;
  double r0[3];
  double rsite1[3];
  double rsep[3];

  colloid_t * pcmap = NULL;
  colloid_link_t * p_link;
  colloid_link_t * p_last;

  assert(p_colloid);
  assert(map);

  coords_nlocal(ntotal);
  coords_nlocal_offset(offset);

  p_link = p_colloid->lnk;
  p_last = p_colloid->lnk;
  radius = p_colloid->s.a0;

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

  i_min = imax(1,         (int) floor(r0[X] - radius));
  i_max = imin(ntotal[X], (int) ceil (r0[X] + radius));
  j_min = imax(1,         (int) floor(r0[Y] - radius));
  j_max = imin(ntotal[Y], (int) ceil (r0[Y] + radius));
  k_min = imax(1,         (int) floor(r0[Z] - radius));
  k_max = imin(ntotal[Z], (int) ceil (r0[Z] + radius));

  for (i = i_min; i <= i_max; i++) { 
    for (j = j_min; j <= j_max; j++) {
      for (k = k_min; k <= k_max; k++) {

	ic = i;
	jc = j;
	kc = k;

	index1 = coords_index(ic, jc, kc);
	colloids_info_map(cinfo, index1, &pcmap);
	if (pcmap != p_colloid) continue;

	rsite1[X] = 1.0*i;
	rsite1[Y] = 1.0*j;
	rsite1[Z] = 1.0*k;
	coords_minimum_distance(r0, rsite1, rsep);

	for (p = 1; p < NVEL; p++) {

	  /* Find the index of the outside site */

	  ii = ic + cv[p][X];
	  jj = jc + cv[p][Y];
	  kk = kc + cv[p][Z];

	  index0 = coords_index(ii, jj, kk);
	  map_status(map, index0, &status);
	  if (status != MAP_BOUNDARY) continue;

	  /* Add a link */

	  if (p_link) {
	    /* Use existing link (lambda always 0.5 at moment) */

	    p_link->rb[X] = rsep[X] + lambda*cv[p][0];
	    p_link->rb[Y] = rsep[Y] + lambda*cv[p][1];
	    p_link->rb[Z] = rsep[Z] + lambda*cv[p][2];

	    p_link->i = index0;
	    p_link->j = index1;
	    p_link->p = NVEL - p;
	    p_link->status = LINK_BOUNDARY;

	    /* Next link */
	    p_last = p_link;
	    p_link = p_link->next;
	  }
	  else {
	    /* Add a new link to the end of the list */

	    p_link = colloid_link_allocate();

	    p_link->rb[X] = rsep[X] + lambda*cv[p][0];
	    p_link->rb[Y] = rsep[Y] + lambda*cv[p][1];
	    p_link->rb[Z] = rsep[Z] + lambda*cv[p][2];

	    p_link->i = index0;
	    p_link->j = index1;
	    p_link->p = NVEL - p;
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

int build_count_faces_local(colloid_t * colloid, double * sa, double * saf) {

  int p;
  colloid_link_t * pl = NULL;

  assert(colloid);
  assert(sa);
  assert(saf);

  *sa = 0.0;
  *saf = 0.0;

  for (pl = colloid->lnk; pl != NULL; pl = pl->next) {
    if (pl->status == LINK_UNUSED) continue;
    p = pl->p;
    p = cv[p][X]*cv[p][X] + cv[p][Y]*cv[p][Y] + cv[p][Z]*cv[p][Z];
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
 *  For each particle, we replace a proportion of the conservered
 *  surplus or deficit at each fluid face.
 *
 *****************************************************************************/

int build_conservation(colloids_info_t * cinfo, field_t * phi, psi_t * psi) {

  int ic, jc, kc;
  int ncell[3];
  int p;

  double value;
  double dphi, dq0, dq1;

  colloid_t * colloid = NULL;
  colloid_link_t * pl = NULL;

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &colloid);

        /* For each colloid in this cell, check solid/fluid status */

        for (; colloid != NULL; colloid = colloid->next) {

          dphi = colloid->s.deltaphi / colloid->s.saf;
          dq0  = colloid->s.deltaq0  / colloid->s.saf;
          dq1  = colloid->s.deltaq1  / colloid->s.saf;

	  if (dq0 == 0.0 && dq1 == 0.0) break;

          for (pl = colloid->lnk; pl != NULL; pl = pl->next) {

	    if (pl->status == LINK_UNUSED) continue;

            p = pl->p;
            p = cv[p][X]*cv[p][X] + cv[p][Y]*cv[p][Y] + cv[p][Z]*cv[p][Z];

            if (p == 1) {
              /* Replace */
              if (phi) {
                field_scalar(phi, pl->i, &value);
                field_scalar_set(phi, pl->i, value + dphi);
              }
	      /* For charge, do not drop densities below zero. */
              if (psi) {
                psi_rho(psi, pl->i, 0, &value);
                if ((value + dq0) >= 0.0) {
		  colloid->s.deltaq0 -= dq0;
		  psi_rho_set(psi, pl->i, 0, value + dq0);
		}
                psi_rho(psi, pl->i, 1, &value);
                if ((value + dq1) >=  0.0) {
		  colloid->s.deltaq1 -= dq1;
		  psi_rho_set(psi, pl->i, 1, value + dq1);
		}
              }
            }
          }
        }

        /* Next cell */
      }
    }
  }

  return 0;
}
