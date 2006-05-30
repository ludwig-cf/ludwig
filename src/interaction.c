/*****************************************************************************
 *
 *  interaction.c
 *
 *  Here, broadly, is where the lattice and the colloids interact.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "globals.h"
#include "ccomms.h"
#include "cells.h"
#include "cmem.h"

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "cartesian.h"

static void    COLL_link_mean_contrib(Colloid *, int, FVector);
static void    COLL_remove_binary_fluid(int inode, Colloid *);
static void    COLL_replace_binary_fluid(int inode, Colloid *);
static void    COLL_reconstruct_links(Colloid *);
static void    COLL_reset_links(Colloid *);
static void    COLL_set_virtual_velocity(int inode, int p, FVector u);

static void    COLL_compute_phi_missing(void);


extern Colloid ** coll_map; /* From colloids.c */
extern Colloid ** coll_old;


enum { NGRAD = 27 };

static int bs_cv[NGRAD][3] = {{ 0, 0, 0},
			      { 1,-1,-1},
			      { 1,-1, 1},
			      { 1, 1,-1},
			      { 1, 1, 1},
			      { 0, 1, 0},
			      { 1, 0, 0},
			      { 0, 0, 1},
			      {-1, 0, 0},
			      { 0,-1, 0},
			      { 0, 0,-1},
			      {-1,-1,-1},
			      {-1,-1, 1},
			      {-1, 1,-1},
			      {-1, 1, 1},
			      { 1, 1, 0},
			      { 1,-1, 0},
			      {-1, 1, 0},
			      {-1,-1, 0},
			      { 1, 0, 1},
			      { 1, 0,-1},
			      {-1, 0, 1},
			      {-1, 0,-1},
			      { 0, 1, 1},
			      { 0, 1,-1},
			      { 0,-1, 1},
			      { 0,-1,-1}};


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
  Float   radius, rsq;

  IVector ncell;
  int     N[3];
  int     offset[3];
  int     ic, jc, kc;
  int     cifac, cjfac;

  get_N_local(N);
  get_N_offset(offset);

  xfac = (N[Y] + 2)*(N[Z] + 2);
  yfac = (N[Z] + 2);

  ncell = Global_Colloid.Ncell;
  cjfac = (ncell.z + 2);
  cifac = (ncell.y + 2)*cjfac;

  /* First, set any existing colloid sites to fluid */

  nsites = (N[X] + 2)*(N[Y] + 2)*(N[Z] + 2);

  for (n = 0; n < nsites; n++) {
    if (site_map[n] == COLLOID) site_map[n] = FLUID;
    coll_old[n] = coll_map[n];
    coll_map[n] = NULL;
  }

  /* Loop through all cells (including the halo cells) */

  for (ic = 0; ic <= ncell.x + 1; ic++)
    for (jc = 0; jc <= ncell.y + 1; jc++)
      for (kc = 0; kc <= ncell.z + 1; kc++) {

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
		  site_map[index] = COLLOID;
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
  IVector     ncell = Global_Colloid.Ncell;

  for (ic = 0; ic <= ncell.x + 1; ic++)
    for (jc = 0; jc <= ncell.y + 1; jc++)
      for (kc = 0; kc <= ncell.z + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  p_colloid->sumw   = 0.0;
	  p_colloid->cbar   = UTIL_fvector_zero();
	  p_colloid->rxcbar = UTIL_fvector_zero();

	  if (p_colloid->rebuild) {
	    /* The shape has changed, so need to reconstruct */
	    COLL_reconstruct_links(p_colloid);
	  }
	  else {
	    /* Shape unchanged, so just reset existing links */
	    COLL_reset_links(p_colloid);
	  }

#ifdef _VERY_VERBOSE_
	  VERBOSE(("(partial [%d]) position (%g,%g,%g)\n", p_colloid->index,
		   p_colloid->r));
	  VERBOSE(("(partial [%d]) cbar     (%g,%g,%g)\n", p_colloid->index,
		   p_colloid->cbar));
	  VERBOSE(("(partial [%d]) f0       (%g,%g,%g)\n", p_colloid->index,
		   p_colloid->f0));
	  VERBOSE(("(partial [%d]) delta    (%g,%g)\n", p_colloid->index,
		   p_colloid->deltam, p_colloid->deltaphi));
#endif

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
 *    The way that "unused status" (-1) is set is a failsafe at the
 *    moment.
 *
 ****************************************************************************/

void COLL_reconstruct_links(Colloid * p_colloid) {

  COLL_Link * p_link;
  COLL_Link * p_last;
  int         i_min, i_max, j_min, j_max, k_min, k_max;
  int         i, ic, ii, j, jc, jj, k, kc, kk;
  int         index0, index1, p;

  Float       radius;
  Float       lambda = 0.5;
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
    p_link->solid = -1;
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

  VERBOSE(("Reconstruct: particle (%d) at (%f,%f,%f)\n", p_colloid->index,
	   p_colloid->r.x, p_colloid->r.y, p_colloid->r.z));
  VERBOSE(("[%d, %d][%d, %d][%d, %d]\n", i_min, i_max, j_min, j_max,
	   k_min, k_max));

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

	    if (site_map[index1] == FLUID) {
	      p_link->solid = 0;
	      COLL_link_mean_contrib(p_colloid, p, p_link->rb);
	    }
	    else {
	      FVector ub;
	      p_link->solid = 1;
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

	    p_link = CMEM_allocate_boundary_link();

	    p_link->rb.x = rsep.x + lambda*cv[p][0];
	    p_link->rb.y = rsep.y + lambda*cv[p][1];
	    p_link->rb.z = rsep.z + lambda*cv[p][2];

	    p_link->i = index1;
	    p_link->j = index0;
	    p_link->v = p;

	    if (site_map[index1] == FLUID) {
	      p_link->solid = 0;
	      COLL_link_mean_contrib(p_colloid, p, p_link->rb);
	    }
	    else {
	      FVector ub;
	      p_link->solid = 1;
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

    if (p_link->solid == -1) {
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

      p_link->rb.x = rsep.x + lambda*cv[p_link->v][0];
      p_link->rb.y = rsep.y + lambda*cv[p_link->v][1];
      p_link->rb.z = rsep.z + lambda*cv[p_link->v][2];

      if (site_map[p_link->i] == FLUID) {
	p_link->solid = 0;
	COLL_link_mean_contrib(p_colloid, p_link->v, p_link->rb);
      }
      else {
	FVector ub;

	p_link->solid = 1;
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
 *  COLL_bounce_back_pass1
 *
 *  Issues
 *    The computation of the drag matrix elements zeta might
 *    be moved away from here in the event of position
 *    updates occuring at frequency less than every time step.
 *
 *****************************************************************************/

void COLL_bounce_back_pass1() {

  Colloid   * p_colloid;
  COLL_Link * p_link;

  FVector   ci;
  FVector   vb;

  int       i, j, ij, ji;
  Float     rho = Global_Colloid.rho;
  Float     dm;
  Float     delta;
  Float     rsumw;

  IVector   ncell;
  int       ic, jc, kc;
  extern const double c2rcs2;

  TIMER_start(TIMER_BBL);

  ncell = Global_Colloid.Ncell;

  for (ic = 0; ic <= ncell.x + 1; ic++)
    for (jc = 0; jc <= ncell.y + 1; jc++)
      for (kc = 0; kc <= ncell.z + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	/* For each colloid in the list */


	while (p_colloid != NULL) {

	  p_link = p_colloid->lnk;

	  for (i = 0; i < 21; i++) {
	    p_colloid->zeta[i] = 0.0;
	  }

	  /* We need to normalise link quantities by the sum of weights
	   * over the particle. Note that sumw cannot be zero here during
	   * correct operation (implies the particle has no links). */

	  rsumw = 1.0 / p_colloid->sumw;
	  p_colloid->cbar.x   *= rsumw;
	  p_colloid->cbar.y   *= rsumw;
	  p_colloid->cbar.z   *= rsumw;
	  p_colloid->rxcbar.x *= rsumw;
	  p_colloid->rxcbar.y *= rsumw;
	  p_colloid->rxcbar.z *= rsumw;
	  p_colloid->deltam   *= rsumw;
	  p_colloid->deltaphi *= rsumw;

	  /* Sum over the links */ 

	  while (p_link != NULL) {

#ifdef _VERY_VERBOSE_
	    VERBOSE(("* bbl pass 1 p_link (%p) next (%p)\n",
		     p_link, p_link->next));
#endif
	    if (p_link->solid == -1) {
	      /* ignore */
	    }
	    else {
	      i = p_link->i;        /* index site i (inside) */
	      j = p_link->j;        /* index site j (outside) */
	      ij = p_link->v;       /* link velocity index i->j */
	      ji = NVEL - ij;      /* link velocity index j->i */

	      ci.x = (Float) cv[ij][0];
	      ci.y = (Float) cv[ij][1];
	      ci.z = (Float) cv[ij][2];

	      /* For stationary link, the momentum transfer from the
	       * fluid to the colloid is "dm" */

	      if (p_link->solid == 1) {
		/* Virtual momentum transfer for solid->solid links,
		 * but no contribution to drag maxtrix */
		dm = site[i].f[ij] + site[j].f[ji];
		delta = 0.0;
	      }
	      else {
		/* Bounce back of fluid on outside plus correction
		 * arising from changes in shape at previous step. */
		dm =  2.0*site[i].f[ij]
		  - wv[ij]*p_colloid->deltam; /* minus */
		delta = c2rcs2*wv[ij]*rho;
	      }

	      vb = UTIL_cross_product(p_link->rb, ci);

	      /* Now add contribution to the sums required for 
	       * self-consistent evaluation of new velocities. */

	      p_colloid->f0.x += dm*ci.x;
	      p_colloid->f0.y += dm*ci.y;
	      p_colloid->f0.z += dm*ci.z;

	      p_colloid->t0.x += dm*vb.x;
	      p_colloid->t0.y += dm*vb.y;
	      p_colloid->t0.z += dm*vb.z;

	      /* Corrections when links are missing (close to contact) */

	      ci = UTIL_fvector_subtract(ci, p_colloid->cbar);
	      vb = UTIL_fvector_subtract(vb, p_colloid->rxcbar);

	      /* Drag matrix elements */

	      p_colloid->zeta[ 0] += delta*ci.x*ci.x;
	      p_colloid->zeta[ 1] += delta*ci.x*ci.y;
	      p_colloid->zeta[ 2] += delta*ci.x*ci.z;
	      p_colloid->zeta[ 3] += delta*ci.x*vb.x;
	      p_colloid->zeta[ 4] += delta*ci.x*vb.y;
	      p_colloid->zeta[ 5] += delta*ci.x*vb.z;

	      p_colloid->zeta[ 6] += delta*ci.y*ci.y;
	      p_colloid->zeta[ 7] += delta*ci.y*ci.z;
	      p_colloid->zeta[ 8] += delta*ci.y*vb.x;
	      p_colloid->zeta[ 9] += delta*ci.y*vb.y;
	      p_colloid->zeta[10] += delta*ci.y*vb.z;

	      p_colloid->zeta[11] += delta*ci.z*ci.z;
	      p_colloid->zeta[12] += delta*ci.z*vb.x;
	      p_colloid->zeta[13] += delta*ci.z*vb.y;
	      p_colloid->zeta[14] += delta*ci.z*vb.z;

	      p_colloid->zeta[15] += delta*vb.x*vb.x;
	      p_colloid->zeta[16] += delta*vb.x*vb.y;
	      p_colloid->zeta[17] += delta*vb.x*vb.z;

	      p_colloid->zeta[18] += delta*vb.y*vb.y;
	      p_colloid->zeta[19] += delta*vb.y*vb.z;

	      p_colloid->zeta[20] += delta*vb.z*vb.z;
	    }

	    /* Next link */
	    p_link = p_link->next;
	  }

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }

  TIMER_stop(TIMER_BBL);

  return;
}


/*****************************************************************************
 *
 *  COLL_bounce_back_pass2
 *
 *  Implement bounce-back on links having updated the colloid
 *  velocities via the implicit method.
 *
 *****************************************************************************/

void COLL_bounce_back_pass2() {

  Colloid   * p_colloid;
  COLL_Link * p_link;

  FVector   vb;
  FVector   ci;

  int       i, j, ij, ji;
  Float     dm;
  Float     vdotc;
  Float     rho = Global_Colloid.rho;
  Float     dms;
  Float     df, dg;

  Float     dgtm1;

  IVector   ncell;
  int       ic, jc, kc;

  extern double * phi_site;
  extern const double c2rcs2;

  TIMER_start(TIMER_BBL);

  ncell = Global_Colloid.Ncell;

  /* Account the current phi deficit */
  Global_Colloid.deltag = 0.0;

  for (ic = 0; ic <= ncell.x + 1; ic++)
    for (jc = 0; jc <= ncell.y + 1; jc++)
      for (kc = 0; kc <= ncell.z + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	/* Update solid -> fluid links for each colloid in the list */

	while (p_colloid != NULL) {

	  /* Set correction for phi arising from previous step */

	  dgtm1 = p_colloid->deltaphi;

	  p_colloid->deltaphi = 0.0;

	  /* Correction to the bounce-back for this particle if it is
	   * without full complement of links */

	  dms = c2rcs2*rho*
	    (UTIL_dot_product(p_colloid->v, p_colloid->cbar)
	     + UTIL_dot_product(p_colloid->omega, p_colloid->rxcbar));


	  /* Run through the links */

	  p_link = p_colloid->lnk;

	  while (p_link != NULL) {

	    if (p_link->solid != 0) {
	      /* This is a colloid -> colloid link (or unused). */
	    }
	    else {

	      i = p_link->i;        /* index site i (outside) */
	      j = p_link->j;        /* index site j (inside) */
	      ij = p_link->v;       /* link velocity index i->j */
	      ji = NVEL - ij;      /* link velocity index j->i */

	      ci.x = (Float) cv[ij][0];
	      ci.y = (Float) cv[ij][1];
	      ci.z = (Float) cv[ij][2];

	      dm =  2.0*site[i].f[ij]
		- wv[ij]*p_colloid->deltam; /* minus */

	      /* Compute the self-consistent boundary velocity,
	       * and add the correction term for changes in shape. */

	      vb = UTIL_cross_product(p_colloid->omega, p_link->rb);
	      vb = UTIL_fvector_add(vb, p_colloid->v);

	      vdotc = c2rcs2*wv[ij]*UTIL_dot_product(vb, ci);

	      df = rho*vdotc + wv[ij]*p_colloid->deltam;

	      dg = phi_site[i]*vdotc;
	      p_colloid->deltaphi += dg;
	      dg -= wv[ij]*dgtm1;

	      /* Correction owing to missing links "squeeze term" */

	      df -= wv[ij]*dms;

	      /* The outside site actually undergoes BBL. However,
	       * the inside site also gets treated to fool the
	       * propagation stage (not really necessary). */

	      site[j].f[ji] = site[i].f[ij] - df;

#ifndef _SINGLE_FLUID_
	      site[j].g[ji] = site[i].g[ij] - dg;
#endif
	    }

	    /* Next link */
	    p_link = p_link->next;
	  }

	  /* Reset factors required for change of shape */
	  p_colloid->deltam = 0.0;
	  p_colloid->f0 = UTIL_fvector_zero();
	  p_colloid->t0 = UTIL_fvector_zero();
	  Global_Colloid.deltag += p_colloid->deltaphi;

	  /* Next colloid */
	  p_colloid = p_colloid->next;

	}

	/* Next cell */
      }

  TIMER_stop(TIMER_BBL);

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

  Float   oldrho;
  Float   oldphi;
  FVector oldu;

  IVector ri;
  FVector rb;
  FVector r0;

  int offset[3];

  get_N_offset(offset);

  /* Get the properties of the old fluid at inode */

  oldrho = MODEL_get_rho_at_site(inode);
  oldphi = MODEL_get_phi_at_site(inode);
  oldu   = MODEL_get_momentum_at_site(inode);

  /* Set the corrections for colloid motion. This requires
   * the local boundary vector rb */

  p_colloid->deltam -= (oldrho - gbl.rho);
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

  p_colloid->deltaphi += (oldphi - gbl.phi);

#ifdef _DEVEL_
  printf("*** Remove fluid at (%d, %d, %d)\n", ri.x, ri.y, ri.z);
  printf("Addtional rc   : (%g, %g, %g)\n", rb.x,rb.y, rb.z);
  printf("Addtional force: (%g, %g, %g)\n", p_colloid->f0.x,
	 p_colloid->f0.y, p_colloid->f0.z);
  printf("Addtional torqe: (%g, %g, %g)\n", p_colloid->t0.x,
	 p_colloid->t0.y, p_colloid->t0.z);
  printf("*** Removed rho:  %f\n", oldrho);
  printf("*** Removed phi:  %f\n\n", oldphi);
#endif

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

  Float    newrho = 0.0;
  Float    newphi = 0.0;
  FVector  newu, newt;
  FVector  r0;

  Float    weight = 0.0;

  Float    newf[NVEL];
  Float    newg[NVEL];

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
    if (coll_old[indexn] || site_map[indexn] == SOLID) continue;

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

  p_colloid->deltam += (newrho - gbl.rho);
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

  p_colloid->deltaphi -= (newphi - gbl.phi);

#ifdef _DEVEL_
  /* Report */

  printf("*** Replace fluid at (%d, %d, %d)\n", ri.x, ri.y, ri.z);
  printf("*** New rho,phi: %g, %g\n", newrho, newphi);
  printf("Addtional rc   : (%g, %g, %g)\n", rb.x,rb.y, rb.z);
  printf("Addtional force: (%g, %g, %g)\n", p_colloid->f0.x,
	 p_colloid->f0.y, p_colloid->f0.z);
  printf("Addtional torqe: (%g, %g, %g)\n", p_colloid->t0.x,
	 p_colloid->t0.y, p_colloid->t0.z);
  printf("New fluid velocity: (%g, %g, %g)\n", newu.x, newu.y, newu.z);
  printf("*** excess mass:  %g\n\n", p_colloid->deltam);
#endif

  return;
}

/****************************************************************************
 *
 *  COLL_compute_phi_gradients
 *
 *  Compute gradients of the order parameter phi (\nabla\phi and
 *  \nabla^2\phi) at fluid nodes only (not halo sites).
 *
 *  Neutral wetting properties are always assumed at the moment.
 *
 *  This takes account of the presence of local solid sites
 *  and uses the full 26 (NGRAD) neighbours. This function uses
 *  exactly the same method as the original gradient method
 *  in BS_fix_gradients but doesn't require boundary sites and
 *  is designed to be more efficient for high solid fractions.
 *
 ****************************************************************************/

void COLL_compute_phi_gradients() {

  int     i, j, k, index, indexn;
  int     p;
  int     xfac, yfac;

  int     isite[NGRAD];

  Float   r9  = 1.0/9.0;
  Float   r18 = 1.0/18.0;
  Float   dphi, phi_b, delsq;
  Float   gradt[NGRAD];

  FVector gradn;

  IVector count;
  int     N[3];

  extern double * phi_site;
  extern double * delsq_phi;
  extern FVector * grad_phi;

  get_N_local(N);

  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  for (i = 1; i <= N[X]; i++)
    for (j = 1; j <= N[Y]; j++)
      for (k = 1; k <= N[Z]; k++) {

	index = i*xfac + j*yfac + k;

	/* Skip solid sites */
	if (site_map[index] != FLUID) continue;

	/* Set solid/fluid flag to index neighbours */

	for (p = 1; p < NGRAD; p++) {
	  indexn = index + xfac*bs_cv[p][0] + yfac*bs_cv[p][1] + bs_cv[p][2];
	  isite[p] = -1;
	  if (site_map[indexn] == FLUID) isite[p] = indexn;
	}


	/* Estimate gradients between fluid sites */

	gradn.x = 0.0;
	gradn.y = 0.0;
	gradn.z = 0.0;
	count.x = 0;
	count.y = 0;
	count.z = 0;

	for (p = 1; p < NGRAD; p++) {

	  if (isite[p] == -1) continue;

	  dphi = phi_site[isite[p]] - phi_site[index];
	  gradn.x += bs_cv[p][0]*dphi;
	  gradn.y += bs_cv[p][1]*dphi;
	  gradn.z += bs_cv[p][2]*dphi;

	  gradt[p] = dphi;

	  count.x += bs_cv[p][0]*bs_cv[p][0];
	  count.y += bs_cv[p][1]*bs_cv[p][1];
	  count.z += bs_cv[p][2]*bs_cv[p][2];
	}

	if (count.x) gradn.x /= (Float) count.x;
	if (count.y) gradn.y /= (Float) count.y;
	if (count.z) gradn.z /= (Float) count.z;

	/* Estimate gradient at boundaries */

	for (p = 1; p < NGRAD; p++) {

	  if (isite[p] == -1) {

	    phi_b = phi_site[index] + 0.5*
	    (bs_cv[p][0]*gradn.x + bs_cv[p][1]*gradn.y + bs_cv[p][2]*gradn.z);

	    /* Set gradient of phi at boundary following wetting properties */
	    /* C and H are always zero at the moment */
	    /* rk = 1.0/kappa;
	     * gradt[p] = -(0.0*phi_b - 0.0)*rk; */
	  }
	}

	/* Accumulate the final gradients */

	delsq = 0.0;
	gradn.x = 0.0;
	gradn.y = 0.0;
	gradn.z = 0.0;

	for (p = 1; p < NGRAD; p++) {
	  delsq   += gradt[p];
	  gradn.x += gradt[p]*bs_cv[p][0];
	  gradn.y += gradt[p]*bs_cv[p][1];
	  gradn.z += gradt[p]*bs_cv[p][2];
	}

	delsq   *= r9;
	gradn.x *= r18;
	gradn.y *= r18;
	gradn.z *= r18;

	delsq_phi[index] = delsq;
	grad_phi[index]  = gradn;

      }

  return;
}


/*****************************************************************************
 *
 *  COLL_compute_phi_missing
 *
 *  Extrapolate (actually average) phi values to sites inside
 *  solid particles. This is done by looking at nearby sites
 *  (connected via a basis vector).
 *
 *  This has no physical meaning; it is used to avoid rubbish
 *  values in the phi field for visualisation purposes.
 *
 ****************************************************************************/

void COLL_compute_phi_missing() {

  int     i, j , k, index, indexn, p;
  int     count;
  int     xfac, yfac;
  int     N[3];

  Float   phibar;

  extern double * phi_site;

  get_N_local(N);

  yfac = (N[Z] + 2);
  xfac = (N[Y] + 2)*yfac;


   for (i = 1; i <= N[X]; i++)
    for (j = 1; j <= N[Y]; j++)
      for (k = 1; k <= N[Z]; k++) {

	index = i*xfac + j*yfac + k;

	if (site_map[index] != FLUID) {

	  /* Look at the neigbours and take the average */
	  count = 0;
	  phibar = 0.0;

	  for (p = 1; p < NVEL; p++) {
	    indexn = index + xfac*cv[p][0] + yfac*cv[p][1] + cv[p][2];
	    if (site_map[indexn] == FLUID) {
	      count += 1;
	      phibar += phi_site[indexn];
	    }
	  }

	  if (count > 0)
	    phi_site[index] = phibar / (Float) count;
	}
      }

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

  Float uc;

  uc = u.x*cv[p][0] + u.y*cv[p][1] + u.z*cv[p][2];
  site[inode].f[p] = wv[p]*(1.0 + 3.0*uc);

#ifdef _DEVELOPMET_
  if (site_map[inode] == FLUID) {
    printf("******* Error: trying to set fluid property!\n");
  }
  /* Full expression */
  u2 = u.x*u.x + u.y*u.y + u.z*u.z;
  site[inode].f[p] = wv[p]*(1.0 - 0.0*1.5*u2 + 3.0*uc + 0.0*4.5*uc*uc);
#endif

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
