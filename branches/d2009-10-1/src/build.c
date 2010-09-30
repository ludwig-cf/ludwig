/*****************************************************************************
 *
 *  build.c
 *
 *  Responsible for the construction of links for particles which
 *  do bounce back on links.
 *
 *  $Id: build.c,v 1.5.4.13 2010-09-30 18:02:35 kevin Exp $
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
#include "phi.h"
#include "timer.h"
#include "colloids.h"
#include "site_map.h"
#include "util.h"
#include "wall.h"
#include "build.h"

static colloid_t ** coll_map;        /* colloid_t map. */
static colloid_t ** coll_old;        /* Map at the previous time step */

static void build_link_mean(colloid_t * pc, int p, const double rb[3]);
static void build_virtual_distribution_set(int index, int p,
					   const double u[3]);
static void    COLL_reconstruct_links(colloid_t *);
static void    reconstruct_wall_links(colloid_t *);
static void    COLL_reset_links(colloid_t *);
static void    build_remove_fluid(int index, colloid_t *);
static void    build_remove_order_parameter(int index, colloid_t *);
static void    build_replace_fluid(int index, colloid_t *);
static void    build_replace_order_parameter(int indeex, colloid_t *);

/*****************************************************************************
 *
 *  COLL_init_coordinates
 *
 *  Allocate colloid map.
 *
 *****************************************************************************/

void COLL_init_coordinates() {

  int nsites;

  /* Allocate space for the local colloid map (2 of them) */

  nsites = coords_nsites();

  coll_map = (colloid_t **) malloc(nsites*sizeof(colloid_t *));
  coll_old = (colloid_t **) malloc(nsites*sizeof(colloid_t *));

  if (coll_map == (colloid_t **) NULL) fatal("malloc (coll_map) failed");
  if (coll_old == (colloid_t **) NULL) fatal("malloc (coll_old) failed");

  return;
}

/*****************************************************************************
 *
 *  COLL_update_map
 *
 *  This routine is responsible for setting the solid/fluid status
 *  of all nodes in the presence on colloids. This must be complete
 *  before attempting to build the colloid links.
 *
 *  Issues:
 *
 ****************************************************************************/

void COLL_update_map() {

  int     n, nsites;
  int     i, j, k;
  int     i_min, i_max, j_min, j_max, k_min, k_max;
  int     index;
  int     nhalo;

  colloid_t * p_colloid;

  double  r0[3];
  double  rsite0[3];
  double  rsep[3];

  double   radius, rsq;

  int     N[3];
  int     offset[3];
  int     ic, jc, kc;

  coords_nlocal(N);
  coords_nlocal_offset(offset);
  nhalo = coords_nhalo();

  /* First, set any existing colloid sites to fluid */

  nsites = coords_nsites();

  for (ic = 1 - nhalo; ic <= N[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= N[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= N[Z] + nhalo; kc++) {
	/* This avoids setting BOUNDARY to FLUID */
	index = coords_index(ic, jc, kc);

	if (site_map_get_status_index(index) == COLLOID) {
	  /* No wetting properties required */
	  site_map_set(index, FLUID, 0.0, 0.0);
	}
      }
    }
  }

  for (n = 0; n < nsites; n++) {
    coll_old[n] = coll_map[n];
    coll_map[n] = NULL;
  }

  /* Loop through all cells (including the halo cells) */

  for (ic = 0; ic <= Ncell(X) + 1; ic++)
    for (jc = 0; jc <= Ncell(Y) + 1; jc++)
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	/* Set the cell index */

	p_colloid = colloids_cell_list(ic, jc, kc);

	/* For each colloid in this cell, check solid/fluid status */

	while (p_colloid != NULL) {

	  /* Set actual position and radius */

	  radius = p_colloid->s.a0;
	  rsq    = radius*radius;

	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */

	  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
	  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
	  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

	  /* Compute appropriate range of sites that require checks, i.e.,
	   * a cubic box around the centre of the colloid. However, this
	   * should not extend beyond the boundary of the current domain
	   * (but include halos). */

	  i_min = imax(1-nhalo,    (int) floor(r0[X] - radius));
	  i_max = imin(N[X]+nhalo, (int) ceil (r0[X] + radius));
	  j_min = imax(1-nhalo,    (int) floor(r0[Y] - radius));
	  j_max = imin(N[Y]+nhalo, (int) ceil (r0[Y] + radius));
	  k_min = imax(1-nhalo,    (int) floor(r0[Z] - radius));
	  k_max = imin(N[Z]+nhalo, (int) ceil (r0[Z] + radius));

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

		  coll_map[index] = p_colloid;
		  site_map_set(index, COLLOID, p_colloid->s.c, p_colloid->s.h);
		}
		/* Next site */
	      }

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }

  return;
}


/*****************************************************************************
 *
 *  COLL_update_links
 *
 *  Reconstruct or reset the boundary links for each colloid as necessary.
 *
 *****************************************************************************/

void COLL_update_links() {

  colloid_t   * p_colloid;

  int         ia;
  int         ic, jc, kc;

  for (ic = 0; ic <= Ncell(X) + 1; ic++)
    for (jc = 0; jc <= Ncell(Y) + 1; jc++)
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {

	  p_colloid->sumw   = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    p_colloid->cbar[ia] = 0.0;
	    p_colloid->rxcbar[ia] = 0.0;
	  }

	  if (p_colloid->s.rebuild) {
	    /* The shape has changed, so need to reconstruct */
	    COLL_reconstruct_links(p_colloid);
	    if (wall_present()) reconstruct_wall_links(p_colloid);
	  }
	  else {
	    /* Shape unchanged, so just reset existing links */
	    COLL_reset_links(p_colloid);
	  }

	  /* Next colloid */

	  p_colloid->s.rebuild = 0;
	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }

  return;
}


/****************************************************************************
 *
 *  COLL_reconstruct_links
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

void COLL_reconstruct_links(colloid_t * p_colloid) {

  colloid_link_t * p_link;
  colloid_link_t * p_last;
  int         i_min, i_max, j_min, j_max, k_min, k_max;
  int         i, ic, ii, j, jc, jj, k, kc, kk;
  int         index0, index1, p;

  double       radius;
  double       lambda = 0.5;
  double      rsite1[3];
  double      rsep[3];
  double      r0[3];
  int         N[3];
  int         offset[3];

  coords_nlocal(N);
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

  i_min = imax(1,    (int) floor(r0[X] - radius));
  i_max = imin(N[X], (int) ceil (r0[X] + radius));
  j_min = imax(1,    (int) floor(r0[Y] - radius));
  j_max = imin(N[Y], (int) ceil (r0[Y] + radius));
  k_min = imax(1,    (int) floor(r0[Z] - radius));
  k_max = imin(N[Z], (int) ceil (r0[Z] + radius));

  for (i = i_min; i <= i_max; i++)
    for (j = j_min; j <= j_max; j++)
      for (k = k_min; k <= k_max; k++) {

	ic = i;
	jc = j;
	kc = k;

	index1 = coords_index(ic, jc, kc);

	if (coll_map[index1] == p_colloid) continue;

	rsite1[X] = 1.0*i;
	rsite1[Y] = 1.0*j;
	rsite1[Z] = 1.0*k;
	coords_minimum_distance(r0, rsite1, rsep);

	/* Index 1 is outside, so cycle through the lattice vectors
	 * to determine if the end is inside, and so requires a link */

	for (p = 1; p < NVEL; p++) {

	  /* Find the index of the inside site */

	  ii = ic + cv[p][X];
	  jj = jc + cv[p][Y];
	  kk = kc + cv[p][Z];

	  index0 = coords_index(ii, jj, kk);

	  if (coll_map[index0] != p_colloid) continue;

	  /* Index 0 is inside, so now add a link*/

	  if (p_link) {
	    /* Use existing link (lambda always 0.5 at moment) */

	    p_link->rb[X] = rsep[X] + lambda*cv[p][X];
	    p_link->rb[Y] = rsep[Y] + lambda*cv[p][Y];
	    p_link->rb[Z] = rsep[Z] + lambda*cv[p][Z];

	    p_link->i = index1;
	    p_link->j = index0;
	    p_link->p = p;

	    if (site_map_get_status_index(index1) == FLUID) {
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

	    if (site_map_get_status_index(index1) == FLUID) {
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

  /* Finally, flag any remaining (unused) links as inactive */
  /* All links may have to be flagged as unused... */

  /* ... is the non-failsafe method ...
  if (p_last) p_link = p_last->next;
  if (p_last == p_colloid->lnk) p_link = p_last;

  while (p_link) {
    p_link->solid = -1;
    p_link = p_link->next;
  }
  */


  return;
}

/*****************************************************************************
 *
 *  COLL_reset_links
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

void COLL_reset_links(colloid_t * p_colloid) {

  int ia;

  colloid_link_t * p_link;
  int         isite[3];
  double      rsite[3];
  double      rsep[3];
  double      r0[3];
  int         offset[3];

  double      lambda = 0.5;

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

      if (site_map_get_status_index(p_link->i) == FLUID) {
	p_link->status = LINK_FLUID;
	build_link_mean(p_colloid, p_link->p, p_link->rb);
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
    }

    /* Next link */
    p_link = p_link->next;
  }

  return;
}

/*****************************************************************************
 *
 *  COLL_remove_or_replace_fluid
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

void COLL_remove_or_replace_fluid() {

  colloid_t * p_colloid;

  int     i, j, k;
  int     index;
  int     sold, snew;
  int     halo;
  int     N[3];
  int     nhalo;

  coords_nlocal(N);
  nhalo = coords_nhalo();

  for (i = 1 - nhalo; i <= N[X] + nhalo; i++) {
    for (j = 1 - nhalo; j <= N[Y] + nhalo; j++) {
      for (k = 1 - nhalo; k <= N[Z] + nhalo; k++) {

	index = coords_index(i, j, k);

	sold = (coll_old[index] != (colloid_t *) NULL);
	snew = (coll_map[index] != (colloid_t *) NULL);

	halo = (i < 1 || j < 1 || k < 1 ||
		i > N[X] || j > N[Y] || k > N[Z]);

	if (sold == 0 && snew == 1) {
	  p_colloid = coll_map[index];
	  p_colloid->s.rebuild = 1;

	  if (!halo) {
	    build_remove_fluid(index, p_colloid);
	    build_remove_order_parameter(index, p_colloid);
	  }
	}

	if (sold == 1 && snew == 0) {
	  p_colloid = coll_old[index];
	  p_colloid->s.rebuild = 1;

	  if (!halo) {
	    build_replace_fluid(index, p_colloid);
	    build_replace_order_parameter(index, p_colloid);
	  }
	}
      }
    }
  }

  return;
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

static void build_remove_fluid(int index, colloid_t * p_colloid) {

  int    ia;
  int    ib[3];
  int    noffset[3];

  double rho;             /* density of removed fluid */
  double g[3];            /* momentum of removed fluid */
  double r0[3];           /* Local coords of colloid centre */
  double rb[3];           /* Boundary vector at lattice site index */
  double rtmp[3];

  coords_nlocal_offset(noffset);
  coords_index_to_ijk(index, ib);

  /* Get the properties of the old fluid at inode */

  rho = distribution_zeroth_moment(index, 0);
  distribution_first_moment(index, 0, g);

  /* Set the corrections for colloid motion. This requires
   * the local boundary vector rb for the torque */

  p_colloid->deltam -= (rho - get_rho0());

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

  return;
}

/*****************************************************************************
 *
 *  build_remove_order_parameter
 *
 *  Remove order parameter(s) at the site inode. The old site information
 *  can be lost inside the particle, but we must record the correction.
 *
 *****************************************************************************/

static void build_remove_order_parameter(int index, colloid_t * p_colloid) {

  double phi;

  if (phi_is_finite_difference()) {
    phi = phi_get_phi_site(index);
  }
  else {
    phi = distribution_zeroth_moment(index, 1);
  }

  p_colloid->s.deltaphi += (phi - get_phi0());

  return;
}

/*****************************************************************************
 *
 *  build_replace_fluid
 *
 *  Replace the distributions when a fluid site (index) is exposed.
 *  This gives rise to corrections on the particle force and torque.
 *
 *****************************************************************************/

static void build_replace_fluid(int index, colloid_t * p_colloid) {

  int    indexn, p, pdash;
  int    ia;
  int    ib[3];
  int    noffset[3];

  double newrho;
  double weight;
  double g[3];                /* Change in momentum */
  double r0[3];               /* Centre of colloid in local coordinates */
  double rb[3];               /* Boundary vector at site index */
  double rtmp[3];
  double newf[NVEL];          /* Replacement distributions */

  coords_nlocal_offset(noffset);
  coords_index_to_ijk(index, ib);

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
    if (coll_old[indexn] || site_map_get_status_index(indexn)==SOLID) continue;

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

  p_colloid->deltam += (newrho - get_rho0());

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

  return;
}

/*****************************************************************************
 *
 *  build_replace_order_parameter
 *
 *  Replace the order parameter(s) at a newly exposed site (index).
 *
 *****************************************************************************/

static void build_replace_order_parameter(int index, colloid_t * p_colloid) {

  int indexn, n, p, pdash;
  int nop;
  int ri[3];

  double newphi = 0.0;
  double weight = 0.0;
  double newg[NVEL];
  double * phi;

  nop = phi_nop();
  coords_index_to_ijk(index, ri);

  /* Check the surrounding sites that were linked to inode,
   * and accumulate a (weighted) average distribution. */

  for (p = 0; p < NVEL; p++) {
    newg[p] = 0.0;
  }

  if (phi_is_finite_difference()) {

    phi = (double *) malloc(nop*sizeof(double));
    if (phi == NULL) fatal("malloc(phi) failed\n");

    for (n = 0; n < nop; n++) {
      phi[n] = 0.0;
    }

    for (p = 1; p < NVEL; p++) {

      indexn = coords_index(ri[X] + cv[p][X], ri[Y] + cv[p][Y],
			      ri[Z] + cv[p][Z]);

      /* Site must have been fluid before position update */
      if (coll_old[indexn] || site_map_get_status_index(indexn)==SOLID)
	continue;
      for (n = 0; n < nop; n++) {
	phi[n] += wv[p]*phi_op_get_phi_site(index, n);
      }
      weight += wv[p];
    }

    weight = 1.0/weight;
    for (n = 0; n < nop; n++) {
      phi_op_set_phi_site(index, n, phi[n]*weight);
    }
    free(phi);
  }
  else {

    /* Reset the distribution (distribution index 1) */

    for (p = 1; p < NVEL; p++) {

      indexn = coords_index(ri[X] + cv[p][X], ri[Y] + cv[p][Y],
			      ri[Z] + cv[p][Z]);

      /* Site must have been fluid before position update */
      if (coll_old[indexn] || site_map_get_status_index(indexn)==SOLID)
	continue;

      for (pdash = 0; pdash < NVEL; pdash++) {
	newg[pdash] += wv[p]*distribution_f(indexn, pdash, 1);
      }
      weight += wv[p];
    }

    /* Set new fluid distributions */

    weight = 1.0/weight;

    for (p = 0; p < NVEL; p++) {
      newg[p] *= weight;
      distribution_f_set(index, p, 1, newg[p]);

      /* ... and remember the new fluid properties */
      newphi += newg[p];
    }
  }

  /* Set corrections arising from change in order parameter */

  p_colloid->s.deltaphi -= (newphi - get_phi0());

  return;
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
 *  COLL_fcoords_from_ijk
 *
 *  Return the physical coordinates (x,y,z) of the lattice site with
 *  index (i,j,k) as an FVector.
 *
 *  So the convention is:
 *         i = 1 => x = 1.0 etc. so the 'control volume' for lattice
 *         site i extends from x(i)-1/2 to x(i)+1/2.
 *         Halo points at i = 0 and i = N.x+1 are images of i = N.x
 *         and i = 1, respectively. At the moment, the halo points
 *         retain 'unphysical' coordinates 0 and N.x+1.
 *
 *****************************************************************************/


/*****************************************************************************
 *
 *  build_wall_links
 *
 *  This constructs links between colloid and fixed wall.
 *
 *****************************************************************************/

void reconstruct_wall_links(colloid_t * p_colloid) {

  colloid_link_t * p_link;
  colloid_link_t * p_last;
  int         i_min, i_max, j_min, j_max, k_min, k_max;
  int         i, ic, ii, j, jc, jj, k, kc, kk;
  int         index0, index1, p;

  double       radius;
  double       lambda = 0.5;
  double     r0[3];
  double     rsite1[3];
  double     rsep[3];
  int         N[3];
  int         offset[3];

  coords_nlocal(N);
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

  i_min = imax(1,    (int) floor(r0[X] - radius));
  i_max = imin(N[X], (int) ceil (r0[X] + radius));
  j_min = imax(1,    (int) floor(r0[Y] - radius));
  j_max = imin(N[Y], (int) ceil (r0[Y] + radius));
  k_min = imax(1,    (int) floor(r0[Z] - radius));
  k_max = imin(N[Z], (int) ceil (r0[Z] + radius));

  for (i = i_min; i <= i_max; i++) { 
    for (j = j_min; j <= j_max; j++) {
      for (k = k_min; k <= k_max; k++) {

	ic = i;
	jc = j;
	kc = k;

	index1 = coords_index(ic, jc, kc);

	if (coll_map[index1] != p_colloid) continue;

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

	  if (site_map_get_status_index(index0) != BOUNDARY) continue;

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

	    if (p_colloid->lnk == NULL) {
	      /* There should be links in this list. */
	      fatal("No links in list\n");
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

  return;
}

/*****************************************************************************
 *
 *  colloid_at_site_index
 *
 *  Return a pointer to the colloid occupying this site index,
 *  or NULL if none.
 *
 *****************************************************************************/

colloid_t * colloid_at_site_index(int index) {

  if (coll_map == NULL) return NULL;
  return coll_map[index];
}
