/*****************************************************************************
 *
 *  build.c
 *
 *  Responsible for the construction of links for particles which
 *  do bounce back on links.
 *
 *  $Id: build.c,v 1.3 2008-08-24 17:30:44 kevin Exp $
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
#include "timer.h"
#include "colloids.h"
#include "site_map.h"
#include "build.h"

extern Site * site;
extern int     boundaries_present(void);


static Colloid ** coll_map;        /* Colloid map. */
static Colloid ** coll_old;        /* Map at the previous time step */

static void    COLL_link_mean_contrib(Colloid *, int, FVector);
static void    COLL_remove_binary_fluid(int inode, Colloid *);
static void    COLL_replace_binary_fluid(int inode, Colloid *);
static void    COLL_reconstruct_links(Colloid *);
static void    reconstruct_wall_links(Colloid *);
static void    COLL_reset_links(Colloid *);
static void    COLL_set_virtual_velocity(int inode, int p, FVector u);
static IVector COM_index2coord( int index );
static FVector COLL_fcoords_from_ijk(int, int, int);

FVector   COLL_fvector_separation(FVector, FVector);

/*****************************************************************************
 *
 *  COLL_init_coordinates
 *
 *  Allocate colloid map.
 *
 *****************************************************************************/

void COLL_init_coordinates() {

  int n;
  int N[3];

  /* Allocate space for the local colloid map */

  get_N_local(N);
  assert(nhalo_ == 1); /* all following require refactoring */
  n = (N[X] + 2)*(N[Y] + 2)*(N[Z] + 2);

  info("Requesting %d bytes for colloid maps\n", 2*n*sizeof(Colloid*));

  coll_map = (Colloid **) malloc(n*sizeof(Colloid *));
  coll_old = (Colloid **) malloc(n*sizeof(Colloid *));

  if (coll_map == (Colloid **) NULL) fatal("malloc (coll_map) failed");
  if (coll_old == (Colloid **) NULL) fatal("malloc (coll_old) failed");

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
  int     index, xfac, yfac;

  Colloid * p_colloid;

  FVector r0;
  FVector rsite0;
  FVector rsep;
  double   radius, rsq;

  int     N[3];
  int     offset[3];
  int     ic, jc, kc;

  get_N_local(N);
  get_N_offset(offset);

  xfac = (N[Y] + 2)*(N[Z] + 2);
  yfac = (N[Z] + 2);

  /* First, set any existing colloid sites to fluid */

  nsites = (N[X] + 2)*(N[Y] + 2)*(N[Z] + 2);

  for (ic = 1 - nhalo_; ic <= N[X] + nhalo_; ic++) {
    for (jc = 1 - nhalo_; jc <= N[Y] + nhalo_; jc++) {
      for (kc = 1 - nhalo_; kc <= N[Z] + nhalo_; kc++) {
	/* This avoids setting BOUNDARY to FLUID */
	if (site_map_get_status(ic, jc, kc) == COLLOID) {
	  site_map_set_status(ic, jc, kc, FLUID);
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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	/* For each colloid in this cell, check solid/fluid status */

	while (p_colloid != NULL) {

	  /* Set actual position and radius */

	  r0     = p_colloid->r;
	  radius = p_colloid->a0;
	  rsq    = radius*radius;

	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */

	  r0.x -= (double) offset[X];
	  r0.y -= (double) offset[Y];
	  r0.z -= (double) offset[Z];

	  /* Compute appropriate range of sites that require checks, i.e.,
	   * a cubic box around the centre of the colloid. However, this
	   * should not extend beyond the boundary of the current domain
	   * (but include halos). */

	  i_min = imax(0,      (int) floor(r0.x - radius));
	  i_max = imin(N[X]+1, (int) ceil (r0.x + radius));
	  j_min = imax(0,      (int) floor(r0.y - radius));
	  j_max = imin(N[Y]+1, (int) ceil (r0.y + radius));
	  k_min = imax(0,      (int) floor(r0.z - radius));
	  k_max = imin(N[Z]+1, (int) ceil (r0.z + radius));

	  /* Check each site to see whether it is inside or not */

	  for (i = i_min; i <= i_max; i++)
	    for (j = j_min; j <= j_max; j++)
	      for (k = k_min; k <= k_max; k++) {

		/* rsite0 is the coordinate position of the site */

		rsite0 = COLL_fcoords_from_ijk(i, j, k);
		rsep = COLL_fvector_separation(rsite0, r0);

		/* Are we inside? */

		if (UTIL_dot_product(rsep, rsep) < rsq) {

		  /* Set index */
		  index = i*xfac + j*yfac + k;

		  coll_map[index] = p_colloid;
		  site_map_set_status(i, j, k, COLLOID);
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

  Colloid   * p_colloid;

  int         ic, jc, kc;

  for (ic = 0; ic <= Ncell(X) + 1; ic++)
    for (jc = 0; jc <= Ncell(Y) + 1; jc++)
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  p_colloid->sumw   = 0.0;
	  p_colloid->cbar   = UTIL_fvector_zero();
	  p_colloid->rxcbar = UTIL_fvector_zero();

	  if (p_colloid->rebuild) {
	    /* The shape has changed, so need to reconstruct */
	    COLL_reconstruct_links(p_colloid);
	    if (boundaries_present()) reconstruct_wall_links(p_colloid);
	  }
	  else {
	    /* Shape unchanged, so just reset existing links */
	    COLL_reset_links(p_colloid);
	  }

	  /* Next colloid */
	  p_colloid->rebuild = 0;
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

void COLL_reconstruct_links(Colloid * p_colloid) {

  COLL_Link * p_link;
  COLL_Link * p_last;
  int         i_min, i_max, j_min, j_max, k_min, k_max;
  int         i, ic, ii, j, jc, jj, k, kc, kk;
  int         index0, index1, p;

  double       radius;
  double       lambda = 0.5;
  FVector     rsite1, rsep;
  FVector     r0;
  int         xfac, yfac;
  int         N[3];
  int         offset[3];

  get_N_local(N);
  get_N_offset(offset);

  yfac = (N[Z] + 2);
  xfac = (N[Y] + 2)*yfac;

  p_link = p_colloid->lnk;
  p_last = p_link;
  radius = p_colloid->a0;
  r0     = p_colloid->r;

  /* Failsafe approach: set all links to unused status */

  while (p_link) {
    p_link->status = LINK_UNUSED;
    p_link = p_link->next;
  }

  p_link = p_colloid->lnk;
  p_last = p_link;
  /* ... end failsafe */

  /* Limits of the cube around the particle. Make sure these are
   * the appropriate lattice nodes... */

  r0.x -= (double) offset[X];
  r0.y -= (double) offset[Y];
  r0.z -= (double) offset[Z];

  i_min = imax(1,    (int) floor(r0.x - radius));
  i_max = imin(N[X], (int) ceil (r0.x + radius));
  j_min = imax(1,    (int) floor(r0.y - radius));
  j_max = imin(N[Y], (int) ceil (r0.y + radius));
  k_min = imax(1,    (int) floor(r0.z - radius));
  k_max = imin(N[Z], (int) ceil (r0.z + radius));


  for (i = i_min; i <= i_max; i++)
    for (j = j_min; j <= j_max; j++)
      for (k = k_min; k <= k_max; k++) {

	ic = i;
	jc = j;
	kc = k;

	index1 = ic*xfac + jc*yfac + kc;

	if (coll_map[index1] == p_colloid) continue;

	rsite1 = COLL_fcoords_from_ijk(ic, jc, kc);
	rsep = COLL_fvector_separation(r0, rsite1);

	/* Index 1 is outside, so cycle through the lattice vectors
	 * to determine if the end is inside, and so requires a link */

	for (p = 1; p < NVEL; p++) {

	  /* Find the index of the inside site */

	  ii = ic + cv[p][0];
	  jj = jc + cv[p][1];
	  kk = kc + cv[p][2];

	  index0 = ii*xfac + jj*yfac + kk;

	  if (coll_map[index0] != p_colloid) continue;

	  /* Index 0 is inside, so now add a link*/

	  if (p_link) {
	    /* Use existing link (lambda always 0.5 at moment) */

	    p_link->rb.x = rsep.x + lambda*cv[p][0];
	    p_link->rb.y = rsep.y + lambda*cv[p][1];
	    p_link->rb.z = rsep.z + lambda*cv[p][2];

	    p_link->i = index1;
	    p_link->j = index0;
	    p_link->v = p;

	    if (site_map_get_status_index(index1) == FLUID) {
	      p_link->status = LINK_FLUID;
	      COLL_link_mean_contrib(p_colloid, p, p_link->rb);
	    }
	    else {
	      FVector ub;
	      p_link->status = LINK_COLLOID;
	      ub = UTIL_cross_product(p_colloid->omega, p_link->rb);
	      ub = UTIL_fvector_add(ub, p_colloid->v);
	      COLL_set_virtual_velocity(p_link->j, p_link->v, ub);
	    }


	    /* Next link */
	    p_last = p_link;
	    p_link = p_link->next;

	  }
	  else {
	    /* Add a new link to the end of the list */

	    p_link = allocate_boundary_link();

	    p_link->rb.x = rsep.x + lambda*cv[p][0];
	    p_link->rb.y = rsep.y + lambda*cv[p][1];
	    p_link->rb.z = rsep.z + lambda*cv[p][2];

	    p_link->i = index1;
	    p_link->j = index0;
	    p_link->v = p;

	    if (site_map_get_status_index(index1) == FLUID) {
	      p_link->status = LINK_FLUID;
	      COLL_link_mean_contrib(p_colloid, p, p_link->rb);
	    }
	    else {
	      FVector ub;
	      p_link->status = LINK_COLLOID;
	      ub = UTIL_cross_product(p_colloid->omega, p_link->rb);
	      ub = UTIL_fvector_add(ub, p_colloid->v);
	      COLL_set_virtual_velocity(p_link->j, p_link->v, ub);
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

void COLL_reset_links(Colloid * p_colloid) {

  COLL_Link * p_link;
  FVector     rsite, rsep;
  FVector     r0;
  IVector     isite;
  int         offset[3];

  double      lambda = 0.5;

  get_N_offset(offset);

  p_link = p_colloid->lnk;

  while (p_link) {

    if (p_link->status == LINK_UNUSED) {
      /* Link is not active */
    }
    else {

      /* Compute the separation between the centre of the colloid
       * and the fluid site involved with this link. The position
       * of the outside site is rsite in local coordinates. */

      isite = COM_index2coord(p_link->i);
      rsite = COLL_fcoords_from_ijk(isite.x, isite.y, isite.z);
      r0    = p_colloid->r;
      r0.x -= offset[X];
      r0.y -= offset[Y];
      r0.z -= offset[Z];
      rsep  = COLL_fvector_separation(r0, rsite);

      p_link->rb.x = rsep.x + lambda*cv[p_link->v][X];
      p_link->rb.y = rsep.y + lambda*cv[p_link->v][Y];
      p_link->rb.z = rsep.z + lambda*cv[p_link->v][Z];

      if (site_map_get_status_index(p_link->i) == FLUID) {
	p_link->status = LINK_FLUID;
	COLL_link_mean_contrib(p_colloid, p_link->v, p_link->rb);
      }
      else {
	FVector ub;

	p_link->status = LINK_COLLOID;
	ub = UTIL_cross_product(p_colloid->omega, p_link->rb);
	ub = UTIL_fvector_add(ub, p_colloid->v);
	COLL_set_virtual_velocity(p_link->j, p_link->v, ub);
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

  Colloid * p_colloid;

  int     i, j, k;
  int     xfac, yfac, index;
  int     sold, snew;
  int     halo;
  int     N[3];

  get_N_local(N);

  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  for (i = 0; i <= N[X] + 1; i++) {
    for (j = 0; j <= N[Y] + 1; j++) {
      for (k = 0; k <= N[Z] + 1; k++) {

	index = i*xfac + j*yfac + k;

	sold = (coll_old[index] != (Colloid *) NULL);
	snew = (coll_map[index] != (Colloid *) NULL);

	halo = (i == 0 || j == 0 || k == 0 ||
		i > N[X] || j > N[Y] || k > N[Z]);

	if (sold == 0 && snew == 1) {
	  p_colloid = coll_map[index];
	  p_colloid->rebuild = 1;
	  if (!halo) COLL_remove_binary_fluid(index, p_colloid);
	}

	if (sold == 1 && snew == 0) {
	  p_colloid = coll_old[index];
	  p_colloid->rebuild = 1;
	  if (!halo) COLL_replace_binary_fluid(index, p_colloid);
	}
      }
    }
  }

  return;
}


/*****************************************************************************
 *
 *  COLL_remove_binary_fluid
 *
 *  Remove fluid at site inode, which has just been occupied by
 *  colloid p_colloid.
 *
 *  This requires corrections to the bounce-back for this colloid
 *  at the following time-step.
 *
 *  As the old fluid has disappeared inside the colloid, the
 *  distributions (f, g) at inode do not actually need to be reset.
 *
 ****************************************************************************/

void COLL_remove_binary_fluid(int inode, Colloid * p_colloid) {

  double   oldrho;
  double   oldphi;
  double   tmp[ND];
  FVector oldu;

  IVector ri;
  FVector rb;
  FVector r0;

  int offset[3];

  get_N_offset(offset);

  /* Get the properties of the old fluid at inode */

  oldrho = get_rho_at_site(inode);
  oldphi = get_phi_at_site(inode);
  get_momentum_at_site(inode, tmp);
  oldu.x = tmp[X];
  oldu.y = tmp[Y];
  oldu.z = tmp[Z];

  /* Set the corrections for colloid motion. This requires
   * the local boundary vector rb */

  p_colloid->deltam -= (oldrho - get_rho0());
  p_colloid->f0      = UTIL_fvector_add(p_colloid->f0, oldu);

  ri    = COM_index2coord(inode);
  rb    = COLL_fcoords_from_ijk(ri.x, ri.y, ri.z);
  r0    = p_colloid->r;
  r0.x -= offset[X];
  r0.y -= offset[Y];
  r0.z -= offset[Z];
  rb    = COLL_fvector_separation(r0, rb);

  oldu               = UTIL_cross_product(rb, oldu);
  p_colloid->t0      = UTIL_fvector_add(p_colloid->t0, oldu);

  /* Set the corrections for order parameter */

  p_colloid->deltaphi += (oldphi - get_phi0());

  return;
}


/*****************************************************************************
 *
 *  COLL_replace_binary_fluid
 *
 *  Replace fluid at site inode, which has just been vacated by
 *  colloid p_colloid.
 *
 *  The new fluid properties must be set according to what
 *  fluid distributions (f, g) are
 *  present in the neighbourhood, and corrections added to the
 *  bounce-back for p_colloid at the next step.
 *
 *  Issues
 *    A naive weighted average at the moment.
 *
 ****************************************************************************/

void COLL_replace_binary_fluid(int inode, Colloid * p_colloid) {

  int      indexn, p, pdash;
  int      xfac, yfac;
  int      N[3];
  int      offset[3];
  IVector  ri;

  FVector  rb;

  double    newrho = 0.0;
  double    newphi = 0.0;
  FVector  newu, newt;
  FVector  r0;

  double    weight = 0.0;

  double    newf[NVEL];
  double    newg[NVEL];

  get_N_local(N);
  get_N_offset(offset);

  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  /* Check the surrounding sites that were linked to inode,
   * and accumulate a (weighted) average distribution. */

  for (p = 0; p < NVEL; p++) {
    newf[p] = 0.0;
    newg[p] = 0.0;
  }

  for (p = 1; p < NVEL; p++) {

    indexn = inode + xfac*cv[p][0] + yfac*cv[p][1] + cv[p][2];

    /* Site must have been fluid before position update */
    if (coll_old[indexn] || site_map_get_status_index(indexn)==SOLID) continue;

    for (pdash = 0; pdash < NVEL; pdash++) {
      newf[pdash] += wv[p]*site[indexn].f[pdash];
      newg[pdash] += wv[p]*site[indexn].g[pdash];
    }
    weight += wv[p];
  }

  /* Set new fluid distributions */

  newu = UTIL_fvector_zero();

  weight = 1.0/weight;

  for (p = 0; p < NVEL; p++) {
    newf[p] *= weight;
    newg[p] *= weight;
    site[inode].f[p] = newf[p];
    site[inode].g[p] = newg[p];

    /* ... and remember the new fluid properties */
    newrho += newf[p];
    newphi += newg[p];
    newu.x -= newf[p]*cv[p][0]; /* minus sign is approprite for upcoming ... */
    newu.y -= newf[p]*cv[p][1]; /* ... correction to colloid momentum */
    newu.z -= newf[p]*cv[p][2];
  }

  /* Set corrections for excess mass and momentum. For the
   * correction to the torque, we need the appropriate
   * boundary vector rb */

  p_colloid->deltam += (newrho - get_rho0());
  p_colloid->f0      = UTIL_fvector_add(p_colloid->f0, newu);

  ri    = COM_index2coord(inode);
  rb    = COLL_fcoords_from_ijk(ri.x, ri.y, ri.z);
  r0    = p_colloid->r;
  r0.x -= offset[X];
  r0.y -= offset[Y];
  r0.z -= offset[Z];
  rb    = COLL_fvector_separation(r0, rb);

  newt               = UTIL_cross_product(rb, newu);
  p_colloid->t0      = UTIL_fvector_add(p_colloid->t0, newt);

  /* Set corrections arising from change in order parameter */

  p_colloid->deltaphi -= (newphi - get_phi0());

  return;
}

/****************************************************************************
 *
 *  COLL_set_virtual_velocity
 *
 *  Set f_p at inode to an equilibrium value for a given velocity.
 *
 ****************************************************************************/

void COLL_set_virtual_velocity(int inode, int p, FVector u) {

  double uc;

  uc = u.x*cv[p][0] + u.y*cv[p][1] + u.z*cv[p][2];
  site[inode].f[p] = wv[p]*(1.0 + 3.0*uc);

  return;
}

/*****************************************************************************
 *
 *  COLL_link_mean_contrib
 *
 *  Add a contribution to cbar, rxcbar, and rsunw from
 *  a given link.
 *
 *****************************************************************************/

void COLL_link_mean_contrib(Colloid * p_colloid, int p, FVector rb) {

  FVector rxc;

  rxc.x = cv[p][0];
  rxc.y = cv[p][1];
  rxc.z = cv[p][2];
  rxc   = UTIL_cross_product(rb, rxc);

  p_colloid->cbar.x   += wv[p]*cv[p][0];
  p_colloid->cbar.y   += wv[p]*cv[p][1];
  p_colloid->cbar.z   += wv[p]*cv[p][2];

  p_colloid->rxcbar.x += wv[p]*rxc.x;
  p_colloid->rxcbar.y += wv[p]*rxc.y;
  p_colloid->rxcbar.z += wv[p]*rxc.z;

  p_colloid->sumw     += wv[p];
  return;
}

/*---------------------------------------------------------------------------*\
 * IVector COM_index2coord( int index )                                      *
 *                                                                           *
 * Translates a local index index to local coordinates (x,y,z)               *
 * Returns the local co-ordinates                                            *
\*---------------------------------------------------------------------------*/

IVector COM_index2coord( int index )
{
  IVector coord;
  int N[3];
  int xfac,yfac;

  get_N_local(N);

  yfac = N[Z]+2;
  xfac = (N[Y]+2)*yfac;
  
  coord.x = index/xfac;
  coord.y = (index%xfac)/yfac;
  coord.z = index%yfac;
  return(coord);
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

FVector COLL_fcoords_from_ijk(int i, int j, int k) {

  FVector coord;

  coord.x = (double) i;
  coord.y = (double) j;
  coord.z = (double) k;

  return coord;
}

/*****************************************************************************
 *
 *  COLL_fvector_separation
 *
 *  Returns the vector which joins the centre of two colloids. The
 *  vector starts at position r1, and finishes at r2.
 *
 *  This is a minimum image separation in case of periodic boundaries.
 *
 *****************************************************************************/

FVector COLL_fvector_separation(FVector r1, FVector r2) {

  FVector rsep;

  rsep.x = r2.x - r1.x;
  rsep.y = r2.y - r1.y;
  rsep.z = r2.z - r1.z;

  /* Correct if boundaries are periodic. */

  if (is_periodic(X)) {
    if (rsep.x >  L(X)/2.0) rsep.x -= L(X);
    if (rsep.x < -L(X)/2.0) rsep.x += L(X);
  }

  if (is_periodic(Y)) {
    if (rsep.y >  L(Y)/2.0) rsep.y -= L(Y);
    if (rsep.y < -L(Y)/2.0) rsep.y += L(Y);
  }

  if (is_periodic(Z)) {
    if (rsep.z >  L(Z)/2.0) rsep.z -= L(Z);
    if (rsep.z < -L(Z)/2.0) rsep.z += L(Z);
  }

  return rsep;
}




void reconstruct_wall_links(Colloid * p_colloid) {

  COLL_Link * p_link;
  COLL_Link * p_last;
  int         i_min, i_max, j_min, j_max, k_min, k_max;
  int         i, ic, ii, j, jc, jj, k, kc, kk;
  int         index0, index1, p;

  double       radius;
  double       lambda = 0.5;
  FVector     rsite1, rsep;
  FVector     r0;
  int         xfac, yfac;
  int         N[3];
  int         offset[3];

  get_N_local(N);
  get_N_offset(offset);

  yfac = (N[Z] + 2);
  xfac = (N[Y] + 2)*yfac;

  p_link = p_colloid->lnk;
  p_last = p_colloid->lnk;
  radius = p_colloid->a0;
  r0     = p_colloid->r;

  /* Work out the first unused link */

  while (p_link && p_link->status != LINK_UNUSED) {
    p_last = p_link;
    p_link = p_link->next;
  }

  /* Limits of the cube around the particle. Make sure these are
   * the appropriate lattice nodes... */

  r0.x -= (double) offset[X];
  r0.y -= (double) offset[Y];
  r0.z -= (double) offset[Z];

  i_min = imax(1,    (int) floor(r0.x - radius));
  i_max = imin(N[X], (int) ceil (r0.x + radius));
  j_min = imax(1,    (int) floor(r0.y - radius));
  j_max = imin(N[Y], (int) ceil (r0.y + radius));
  k_min = imax(1,    (int) floor(r0.z - radius));
  k_max = imin(N[Z], (int) ceil (r0.z + radius));

  for (i = i_min; i <= i_max; i++) { 
    for (j = j_min; j <= j_max; j++) {
      for (k = k_min; k <= k_max; k++) {

	ic = i;
	jc = j;
	kc = k;

	index1 = ic*xfac + jc*yfac + kc;

	if (coll_map[index1] != p_colloid) continue;

	rsite1 = COLL_fcoords_from_ijk(ic, jc, kc);
	rsep = COLL_fvector_separation(r0, rsite1);

	for (p = 1; p < NVEL; p++) {

	  /* Find the index of the outside site */

	  ii = ic + cv[p][0];
	  jj = jc + cv[p][1];
	  kk = kc + cv[p][2];

	  index0 = ii*xfac + jj*yfac + kk;

	  if (site_map_get_status_index(index0) != BOUNDARY) continue;

	  /* Add a link */

	  if (p_link) {
	    /* Use existing link (lambda always 0.5 at moment) */

	    p_link->rb.x = rsep.x + lambda*cv[p][0];
	    p_link->rb.y = rsep.y + lambda*cv[p][1];
	    p_link->rb.z = rsep.z + lambda*cv[p][2];

	    p_link->i = index0;
	    p_link->j = index1;
	    p_link->v = NVEL - p;
	    p_link->status = LINK_BOUNDARY;

	    /* Next link */
	    p_last = p_link;
	    p_link = p_link->next;

	  }
	  else {
	    /* Add a new link to the end of the list */

	    p_link = allocate_boundary_link();

	    p_link->rb.x = rsep.x + lambda*cv[p][0];
	    p_link->rb.y = rsep.y + lambda*cv[p][1];
	    p_link->rb.z = rsep.z + lambda*cv[p][2];

	    p_link->i = index0;
	    p_link->j = index1;
	    p_link->v = NVEL - p;
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
