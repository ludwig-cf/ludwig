/*****************************************************************************
 *
 *  bbl.c
 *
 *  Bounce back on links.
 *
 *  $Id: bbl.c,v 1.11 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing Authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Squimer code from Isaac Llopis and Ricard Matas Navarro (U Barcelona).
 *
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "colloids.h"
#include "colloid_sums.h"
#include "model.h"
#include "util.h"
#include "wall.h"
#include "bbl.h"

static int bbl_pass1(colloids_info_t * cinfo);
static int bbl_pass2(colloids_info_t * cinfo);
static int bbl_mass_conservation_compute_force(colloids_info_t * cinfo);
static int bbl_wall_lubrication_account(colloids_info_t * cinfo);

static int bbl_active_ = 0;  /* Flag for active particles. */
static double deltag_ = 0.0; /* Excess or deficit of phi between steps */
static double stress_[3][3]; /* Surface stress */

/*****************************************************************************
 *
 *  bounce_back_on_links
 *
 *  Driver routine for colloid bounce back on links.
 *
 *  The basic method is:
 *  Nguyen and Ladd [Phys. Rev. E {\bf 66}, 046708 (2002)].
 *
 *  The implicit velocity update requires two sweeps through the
 *  boundary nodes:
 *
 *  (1) Compute the velocity-independent force and torque on each
 *      colloid and the elements of the drag matrix for each colloid.
 *
 *  (2) Update the velocity of each colloid.
 *
 *  (3) Do the actual BBL on distributions with the updated colloid
 *      velocity.
 *
 *****************************************************************************/

int bounce_back_on_links(colloids_info_t * cinfo) {

  int ntotal;

  assert(cinfo);

  colloids_info_ntotal(cinfo, &ntotal);
  if (ntotal == 0) return 0;

  colloid_sums_halo(cinfo, COLLOID_SUM_STRUCTURE);
  bbl_pass0(cinfo);
  bbl_pass1(cinfo);
  colloid_sums_halo(cinfo, COLLOID_SUM_DYNAMICS);

  if (bbl_active_) {
    bbl_mass_conservation_compute_force(cinfo);
    colloid_sums_halo(cinfo, COLLOID_SUM_ACTIVE);
  }

  bbl_update_colloids(cinfo);
  bbl_pass2(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  mass_conservation_compute_force
 *
 *****************************************************************************/

static int bbl_mass_conservation_compute_force(colloids_info_t * cinfo) {

  int ia;
  int ic, jc, kc;
  int ncell[3];
  double dm;
  double c[3];
  double rbxc[3];

  colloid_t * pc;
  colloid_link_t * p_link;

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);

  for (ic = 0; ic <= ncell[X] + 1 ; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1 ; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1 ; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	/* For each colloid in the list */

	for ( ; pc; pc = pc->next) {

	  pc->sump /= pc->sumw;
	  p_link = pc->lnk;

	  for (; p_link; p_link = p_link->next) {

	    if (p_link->status == LINK_UNUSED) {
	      /* ignore */
	    }
	    else {
		
	      if (p_link->status == LINK_FLUID) {

		dm = -wv[p_link->p]*pc->sump;

		for (ia = 0; ia < 3; ia++) {
		  c[ia] = 1.0*cv[p_link->p][ia];
		}

		cross_product(p_link->rb, c, rbxc);

		for (ia = 0; ia < 3; ia++) {
		  pc->fc0[ia] += dm*c[ia];
		  pc->tc0[ia] += dm*rbxc[ia];
		}
	      }
	    }

	    /* Next link */
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
 *  bbl_pass0
 *
 *  Set missing 'internal' distributions
 *
 *****************************************************************************/

int bbl_pass0(colloids_info_t * cinfo) {

  int ic, jc, kc, index;
  int ia, ib, p;
  int nextra = 1;
  int nlocal[3];
  int noffset[3];

  double r[3], r0[3], rb[3], ub[3], wxrb[3];
  double udotc, sdotq;

  colloid_t * pc = NULL;

  assert(cinfo);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    r[X] = 1.0*ic;
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      r[Y] = 1.0*jc;
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {
	r[Z] = 1.0*kc;

	index = coords_index(ic, jc, kc);
	colloids_info_map(cinfo, index, &pc);
	if (pc == NULL) continue;

	r0[X] = pc->s.r[X] - 1.0*noffset[X];
	r0[Y] = pc->s.r[Y] - 1.0*noffset[Y];
	r0[Z] = pc->s.r[Z] - 1.0*noffset[Z];
	coords_minimum_distance(r, r0, rb);
	cross_product(pc->s.w, rb, wxrb);
	ub[X] = pc->s.v[X] + wxrb[X];
	ub[Y] = pc->s.v[Y] + wxrb[Y];
	ub[Z] = pc->s.v[Z] + wxrb[Z];

	for (p = 1; p < NVEL; p++) {
	  udotc = cv[p][X]*ub[X] + cv[p][Y]*ub[Y] + cv[p][Z]*ub[Z];
	  sdotq = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      sdotq += q_[p][ia][ib]*ub[ia]*ub[ib];
	    }
	  }
	  distribution_f_set(index, p, 0,
			     wv[p]*(1.0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq));
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  bbl_pass1
 *
 *  Work out the velocity independent terms before actual BBL takes place.
 *
 *****************************************************************************/

static int bbl_pass1(colloids_info_t * cinfo) {

  int ia;
  int ic, jc, kc;
  int i, j, ij, ji;
  int ncell[3];

  double dm;
  double delta;
  double rsumw;
  double c[3];
  double rbxc[3];
  double rho0;

  colloid_t * p_colloid;
  colloid_link_t * p_link;

  assert(cinfo);

  physics_rho0(&rho0);
  colloids_info_ncell(cinfo, ncell);

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

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
	  for (ia = 0; ia < 3; ia++) {
	    p_colloid->cbar[ia]   *= rsumw;
	    p_colloid->rxcbar[ia] *= rsumw;
	  }
	  p_colloid->deltam   *= rsumw;
	  p_colloid->s.deltaphi *= rsumw;

	  /* Sum over the links */ 

	  while (p_link != NULL) {

	    if (p_link->status == LINK_UNUSED) {
	      /* ignore */
	    }
	    else {
	      i = p_link->i;        /* index site i (outside) */
	      j = p_link->j;        /* index site j (inside) */
	      ij = p_link->p;       /* link velocity index i->j */
	      ji = NVEL - ij;      /* link velocity index j->i */

	      assert(ij > 0 && ij < NVEL);

	      /* For stationary link, the momentum transfer from the
	       * fluid to the colloid is "dm" */

	      if (p_link->status == LINK_FLUID) {
		/* Bounce back of fluid on outside plus correction
		 * arising from changes in shape at previous step. */

		dm =  2.0*distribution_f(i, ij, 0)
		  - wv[ij]*p_colloid->deltam; /* minus */
		delta = 2.0*rcs2*wv[ij]*rho0;

		/* Squirmer section */
		{
		  double mod, rmod, dm_a, cost, plegendre, sint;
		  double tans[3], vector1[3];
		  double fdist;

		  /* We expect s.m to be a unit vector, but for floating
		   * point purposes, we must make sure here. */

		  mod = modulus(p_link->rb)*modulus(p_colloid->s.m);
		  rmod = 0.0;
		  if (mod != 0.0) rmod = 1.0/mod;
		  cost = rmod*dot_product(p_link->rb, p_colloid->s.m);
		  if (cost*cost > 1.0) cost = 1.0;
		  assert(cost*cost <= 1.0);
		  sint = sqrt(1.0 - cost*cost);

		  cross_product(p_link->rb, p_colloid->s.m, vector1);
		  cross_product(vector1, p_link->rb, tans);

		  mod = modulus(tans);
		  rmod = 0.0;
		  if (mod != 0.0) rmod = 1.0/mod;
	          plegendre = -sint*(p_colloid->s.b2*cost + p_colloid->s.b1);

		  dm_a = 0.0;
		  for (ia = 0; ia < 3; ia++) {
		    dm_a += -delta*plegendre*rmod*tans[ia]*cv[ij][ia];
		  }

		  fdist = distribution_f(i, ij, 0);
		  fdist += dm_a;
		  distribution_f_set(i, ij, 0, fdist);

		  dm += dm_a;

		  /* needed for mass conservation   */
                  p_colloid->sump += dm_a;
		}
	      }
	      else {
		/* Virtual momentum transfer for solid->solid links,
		 * but no contribution to drag maxtrix */

		dm = distribution_f(i, ij, 0) + distribution_f(j, ji, 0);
		delta = 0.0;
	      }

	      for (ia = 0; ia < 3; ia++) {
		c[ia] = 1.0*cv[ij][ia];
	      }

	      cross_product(p_link->rb, c, rbxc);

	      /* Now add contribution to the sums required for 
	       * self-consistent evaluation of new velocities. */

	      for (ia = 0; ia < 3; ia++) {
		p_colloid->f0[ia] += dm*c[ia];
		p_colloid->t0[ia] += dm*rbxc[ia];
		/* Corrections when links are missing (close to contact) */
		c[ia] -= p_colloid->cbar[ia];
		rbxc[ia] -= p_colloid->rxcbar[ia];
	      }

	      /* Drag matrix elements */

	      p_colloid->zeta[ 0] += delta*c[X]*c[X];
	      p_colloid->zeta[ 1] += delta*c[X]*c[Y];
	      p_colloid->zeta[ 2] += delta*c[X]*c[Z];
	      p_colloid->zeta[ 3] += delta*c[X]*rbxc[X];
	      p_colloid->zeta[ 4] += delta*c[X]*rbxc[Y];
	      p_colloid->zeta[ 5] += delta*c[X]*rbxc[Z];

	      p_colloid->zeta[ 6] += delta*c[Y]*c[Y];
	      p_colloid->zeta[ 7] += delta*c[Y]*c[Z];
	      p_colloid->zeta[ 8] += delta*c[Y]*rbxc[X];
	      p_colloid->zeta[ 9] += delta*c[Y]*rbxc[Y];
	      p_colloid->zeta[10] += delta*c[Y]*rbxc[Z];

	      p_colloid->zeta[11] += delta*c[Z]*c[Z];
	      p_colloid->zeta[12] += delta*c[Z]*rbxc[X];
	      p_colloid->zeta[13] += delta*c[Z]*rbxc[Y];
	      p_colloid->zeta[14] += delta*c[Z]*rbxc[Z];

	      p_colloid->zeta[15] += delta*rbxc[X]*rbxc[X];
	      p_colloid->zeta[16] += delta*rbxc[X]*rbxc[Y];
	      p_colloid->zeta[17] += delta*rbxc[X]*rbxc[Z];

	      p_colloid->zeta[18] += delta*rbxc[Y]*rbxc[Y];
	      p_colloid->zeta[19] += delta*rbxc[Y]*rbxc[Z];

	      p_colloid->zeta[20] += delta*rbxc[Z]*rbxc[Z];

	    }

	    /* Next link */
	    p_link = p_link->next;
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
 *  bbl_pass2
 *
 *  Implement bounce-back on links having updated the colloid
 *  velocities via the implicit method.
 *
 *  The surface stress is also accumulated here (and it really must
 *  done between the colloid velcoity update and the actual bbl).
 *  There's a separate routine to access it below.
 *
 *****************************************************************************/

static int bbl_pass2(colloids_info_t * cinfo) {

  colloid_t      * p_colloid;
  colloid_link_t * p_link;

  int i, j, ij, ji;
  int ic, jc, kc;
  int ia;
  int ncell[3];

  double dm;
  double vdotc;
  double dms;
  double df, dg;
  double fdist;
  double wxrb[3];

  double dgtm1;
  double rho0;

  assert(cinfo);

  physics_rho0(&rho0);
  colloids_info_ncell(cinfo, ncell);

  /* Account the current phi deficit */
  deltag_ = 0.0;

  /* Zero the surface stress */

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      stress_[i][j] = 0.0;
    }
  }

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	/* Update solid -> fluid links for each colloid in the list */

	while (p_colloid != NULL) {

	  /* Set correction for phi arising from previous step */

	  dgtm1 = p_colloid->s.deltaphi;
	  p_colloid->s.deltaphi = 0.0;

	  /* Correction to the bounce-back for this particle if it is
	   * without full complement of links */

	  dms = 0.0;

	  for (ia = 0; ia < 3; ia++) {
	    dms += p_colloid->s.v[ia]*p_colloid->cbar[ia];
	    dms += p_colloid->s.w[ia]*p_colloid->rxcbar[ia];
	  }

	  dms = 2.0*rcs2*rho0*dms;

	  /* Run through the links */

	  p_link = p_colloid->lnk;

	  while (p_link != NULL) {

	    i = p_link->i;       /* index site i (outside) */
	    j = p_link->j;       /* index site j (inside) */
	    ij = p_link->p;      /* link velocity index i->j */
	    ji = NVEL - ij;      /* link velocity index j->i */

	    if (p_link->status == LINK_FLUID) {

	      dm =  2.0*distribution_f(i, ij, 0)
		- wv[ij]*p_colloid->deltam; /* minus */

	      /* Compute the self-consistent boundary velocity,
	       * and add the correction term for changes in shape. */

	      cross_product(p_colloid->s.w, p_link->rb, wxrb);

	      vdotc = 0.0;
	      for (ia = 0; ia < 3; ia++) {
		vdotc += (p_colloid->s.v[ia] + wxrb[ia])*cv[ij][ia];
	      }
	      vdotc = 2.0*rcs2*wv[ij]*vdotc;
	      df = rho0*vdotc + wv[ij]*p_colloid->deltam;

	      /* Contribution to mass conservation from squirmer */

	      df += wv[ij]*p_colloid->sump; 

	      /* Correction owing to missing links "squeeze term" */

	      df -= wv[ij]*dms;

	      /* The outside site actually undergoes BBL. */

	      fdist = distribution_f(i, ij, 0);
	      fdist = fdist - df;
	      distribution_f_set(j, ji, 0, fdist);

	      /* This is slightly clunky. If the order parameter is
	       * via LB, bounce back with correction. */
	      if (distribution_ndist() > 1) {
		dg = distribution_zeroth_moment(i, 1)*vdotc;
		p_colloid->s.deltaphi += dg;
		dg -= wv[ij]*dgtm1;

		fdist = distribution_f(i, ij, 1);
		fdist = fdist - dg;
		distribution_f_set(j, ji, 1, fdist);
	      }

	      /* The stress is r_b f_b */
	      for (ia = 0; ia < 3; ia++) {
		stress_[ia][X] += p_link->rb[X]*(dm - df)*cv[ij][ia];
		stress_[ia][Y] += p_link->rb[Y]*(dm - df)*cv[ij][ia];
		stress_[ia][Z] += p_link->rb[Z]*(dm - df)*cv[ij][ia];
	      }
	    }
	    else if (p_link->status == LINK_COLLOID) {

	      /* The stress should include the solid->solid term */

	      dm = distribution_f(i, ij, 0) + distribution_f(j, ji, 0);

	      for (ia = 0; ia < 3; ia++) {
		stress_[ia][X] += p_link->rb[X]*dm*cv[ij][ia];
		stress_[ia][Y] += p_link->rb[Y]*dm*cv[ij][ia];
		stress_[ia][Z] += p_link->rb[Z]*dm*cv[ij][ia];
	      }
	    }

	    /* Next link */
	    p_link = p_link->next;
	  }

	  /* Reset factors required for change of shape, etc */

	  p_colloid->deltam = 0.0;
	  p_colloid->sump = 0.0;

	  for (ia = 0; ia < 3; ia++) {
	    p_colloid->f0[ia] = 0.0;
	    p_colloid->t0[ia] = 0.0;
	    p_colloid->fc0[ia] = 0.0;
	    p_colloid->tc0[ia] = 0.0;
	  }

	  deltag_ += p_colloid->s.deltaphi;

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
 *  bbl_update_colloids
 *
 *  Update the velocity and position of each particle.
 *
 *  This is a linear algebra problem, which is always 6x6, and is
 *  solved using a bog-standard Gaussian elimination with partial
 *  pivoting, followed by backsubstitution.
 *
 *****************************************************************************/

int bbl_update_colloids(colloids_info_t * cinfo) {

  colloid_t * pc;

  int ia;
  int ic, jc, kc;
  int ncell[3];

  double xb[6];
  double a[6][6];
  int   ipivot[6];
  int   iprow = 0;                 /* The pivot row */
  int   idash, j, k;

  double mass;
  double moment;
  double tmp;
  double rho0;

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);
  colloids_info_rho0(cinfo, &rho0);

  /* Loop round cells and update each particle velocity */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {

	  /* Set up the matrix problem and solve it here. */

	  /* Mass and moment of inertia are those of a hard sphere
	   * with the input radius */

	  mass = (4.0/3.0)*pi_*rho0*pow(pc->s.a0, 3);
	  moment = (2.0/5.0)*mass*pow(pc->s.a0, 2);

	  /* Add inertial terms to diagonal elements */

	  a[0][0] = mass +   pc->zeta[0];
	  a[0][1] =          pc->zeta[1];
	  a[0][2] =          pc->zeta[2];
	  a[0][3] =          pc->zeta[3];
	  a[0][4] =          pc->zeta[4];
	  a[0][5] =          pc->zeta[5];
	  a[1][1] = mass +   pc->zeta[6];
	  a[1][2] =          pc->zeta[7];
	  a[1][3] =          pc->zeta[8];
	  a[1][4] =          pc->zeta[9];
	  a[1][5] =          pc->zeta[10];
	  a[2][2] = mass +   pc->zeta[11];
	  a[2][3] =          pc->zeta[12];
	  a[2][4] =          pc->zeta[13];
	  a[2][5] =          pc->zeta[14];
	  a[3][3] = moment + pc->zeta[15];
	  a[3][4] =          pc->zeta[16];
	  a[3][5] =          pc->zeta[17];
	  a[4][4] = moment + pc->zeta[18];
	  a[4][5] =          pc->zeta[19];
	  a[5][5] = moment + pc->zeta[20];

	  for (k = 0; k < 3; k++) {
	    a[k][k] -= wall_lubrication(k, pc->s.r, pc->s.ah);
	  }

	  /* Lower triangle */

	  a[1][0] = a[0][1];
	  a[2][0] = a[0][2];
	  a[2][1] = a[1][2];
	  a[3][0] = a[0][3];
	  a[3][1] = a[1][3];
	  a[3][2] = a[2][3];
	  a[4][0] = a[0][4];
	  a[4][1] = a[1][4];
	  a[4][2] = a[2][4];
	  a[4][3] = a[3][4];
	  a[5][0] = a[0][5];
	  a[5][1] = a[1][5];
	  a[5][2] = a[2][5];
	  a[5][3] = a[3][5];
	  a[5][4] = a[4][5];

	  /* Form the right-hand side */

	  for (ia = 0; ia < 3; ia++) {
	    xb[ia] = mass*pc->s.v[ia] + pc->f0[ia] + pc->force[ia];
	    xb[3+ia] = moment*pc->s.w[ia] + pc->t0[ia] + pc->torque[ia];
	  }

	 /* Contribution to mass conservation from squirmer */

	  for (ia = 0; ia < 3; ia++) {
	    xb[ia] += pc->fc0[ia];
	    xb[3+ia] += pc->tc0[ia];
	  }

	  /* Begin the Gaussian elimination */

	  for (k = 0; k < 6; k++) {
	    ipivot[k] = -1;
	  }

	  for (k = 0; k < 6; k++) {

	    /* Find pivot row */
	    tmp = 0.0;
	    for (idash = 0; idash < 6; idash++) {
	      if (ipivot[idash] == -1) {
		if (fabs(a[idash][k]) >= tmp) {
		  tmp = fabs(a[idash][k]);
		  iprow = idash;
		}
	      }
	    }
	    ipivot[k] = iprow;

	    /* divide pivot row by the pivot element a[iprow][k] */

	    if (a[iprow][k] == 0.0) {
	      fatal("Gaussain elimination failed in COLL_update\n");
	    }

	    tmp = 1.0 / a[iprow][k];

	    for (j = k; j < 6; j++) {
	      a[iprow][j] *= tmp;
	    }
	    xb[iprow] *= tmp;

	    /* Subtract the pivot row (scaled) from remaining rows */

	    for (idash = 0; idash < 6; idash++) {
	      if (ipivot[idash] == -1) {
		tmp = a[idash][k];
		for (j = k; j < 6; j++) {
		  a[idash][j] -= tmp*a[iprow][j];
		}
		xb[idash] -= tmp*xb[iprow];
	      }
	    }
	  }

	  /* Now do the back substitution */

	  for (idash = 5; idash > -1; idash--) {
	    iprow = ipivot[idash];
	    tmp = xb[iprow];
	    for (k = idash+1; k < 6; k++) {
	      tmp -= a[iprow][k]*xb[ipivot[k]];
	    }
	    xb[iprow] = tmp;
	  }

	  /* Set the position update, but don't actually move
	   * the particles. This is deferred until the next
	   * call to coll_update() and associated cell list
	   * update.
	   * We use mean of old and new velocity. */

	  for (ia = 0; ia < 3; ia++) {
	    if (pc->s.isfixedr == 0) pc->s.dr[ia] = 0.5*(pc->s.v[ia] + xb[ia]);
	    if (pc->s.isfixedv == 0) pc->s.v[ia] = xb[ia];
	    if (pc->s.isfixedw == 0) pc->s.w[ia] = xb[3+ia];
	  }

	  if (pc->s.isfixeds == 0) {
	    rotate_vector(pc->s.m, xb + 3);
	    rotate_vector(pc->s.s, xb + 3);
	  }

	  /* Record the actual hydrodynamic force on the particle */

	  pc->force[X] = pc->f0[X]
	    -(pc->zeta[0]*pc->s.v[X] +
	      pc->zeta[1]*pc->s.v[Y] +
	      pc->zeta[2]*pc->s.v[Z] +
	      pc->zeta[3]*pc->s.w[X] +
	      pc->zeta[4]*pc->s.w[Y] +
	      pc->zeta[5]*pc->s.w[Z]);
          pc->force[Y] = pc->f0[Y]
	    -(pc->zeta[ 1]*pc->s.v[X] +
	      pc->zeta[ 6]*pc->s.v[Y] +
	      pc->zeta[ 7]*pc->s.v[Z] +
	      pc->zeta[ 8]*pc->s.w[X] +
	      pc->zeta[ 9]*pc->s.w[Y] +
	      pc->zeta[10]*pc->s.w[Z]);
          pc->force[Z] = pc->f0[Z]
	    -(pc->zeta[ 2]*pc->s.v[X] +
	      pc->zeta[ 7]*pc->s.v[Y] +
	      pc->zeta[11]*pc->s.v[Z] +
	      pc->zeta[12]*pc->s.w[X] +
	      pc->zeta[13]*pc->s.w[Y] +
	      pc->zeta[14]*pc->s.w[Z]);

	  pc = pc->next;
	}
      }
    }
  }

  /* As the lubrication force is based on the updated velocity, but
   * the old position, we can account for the total momentum here. */

  bbl_wall_lubrication_account(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  bbl_wall_lubrication_account
 *
 *  This just updates the accounting for the total momentum when a
 *  wall lubrication force is present. There is no change to the
 *  dynamics.
 *
 *  The minus sign in the force is consistent with the sign returned
 *  by wall_lubrication().
 *
 *****************************************************************************/

static int bbl_wall_lubrication_account(colloids_info_t * cinfo) {

  int ic, jc, kc, ia;
  int ncell[3];
  double f[3] = {0.0, 0.0, 0.0};

  colloid_t * pc = NULL;

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {
	  for (ia = 0; ia < 3; ia++) {
	    f[ia] -= pc->s.v[ia]*wall_lubrication(ia, pc->s.r, pc->s.ah);
	  }
	  pc = pc->next;
	}
      }
    }
  }

  wall_accumulate_force(f);

  return 0;
}

/*****************************************************************************
 *
 *  get_order_parameter_deficit
 *
 *  Returns the current order parameter deficit owing to BBL.
 *  This is only relevant for full binary LB.
 *  This is a local value for the local subdomain in parallel.
 *
 *****************************************************************************/

int bbl_order_parameter_deficit(double * delta) {

  assert(delta);

  delta[0] = 0.0;
  if (distribution_ndist() == 2) delta[0] = deltag_;

  return 0;
}

/*****************************************************************************
 *
 *  bbl_surface_stress
 *
 *  Report the current surface stress total.
 *
 *****************************************************************************/

void bbl_surface_stress() {

  double rv;
  double send[9];
  double recv[9];
  int    ia, ib;
  
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      send[ia*3+ib] = stress_[ia][ib];
    }
  }

  MPI_Reduce(send, recv, 9, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      stress_[ia][ib] = recv[ia*3+ib];
    }
  }

  rv = 1.0/(L(X)*L(Y)*L(Z));

  info("stress_s x %12.6g %12.6g %12.6g\n",
       rv*stress_[X][X], rv*stress_[X][Y], rv*stress_[X][Z]);
  info("stress_s y %12.6g %12.6g %12.6g\n",
       rv*stress_[X][Y], rv*stress_[Y][Y], rv*stress_[Y][Z]);
  info("stress_s z %12.6g %12.6g %12.6g\n",
       rv*stress_[X][Z], rv*stress_[Y][Z], rv*stress_[Z][Z]);

  return;
}

/*****************************************************************************
 *
 *  bbl_active_on_set
 *
 *  Switch the active particle flag on.
 *
 *****************************************************************************/

void bbl_active_on_set(void) {

  bbl_active_ = 1;
  return;
}

/*****************************************************************************
 *
 *  bbl_active_on
 *
 *****************************************************************************/

int bbl_active_on(void) {

  return bbl_active_;
}
