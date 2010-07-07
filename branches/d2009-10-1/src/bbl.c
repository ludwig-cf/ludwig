/*****************************************************************************
 *
 *  bbl.c
 *
 *  Bounce back on links.
 *
 *  $Id: bbl.c,v 1.10.2.6 2010-07-07 11:31:15 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "colloids.h"
#include "ccomms.h"
#include "model.h"
#include "timer.h"
#include "util.h"
#include "phi.h"
#include "bbl.h"

static void bounce_back_pass1(void);
static void bounce_back_pass2(void);
static void mass_conservation_compute_force(void);
static void update_colloids(void);

/* move to wall.c ? */
static double wall_lubrication(const int dim, const double r[3],
			       const double ah);

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

void bounce_back_on_links() {

  if (colloid_ntotal() == 0) return;

  TIMER_start(TIMER_BBL);

  CCOM_halo_sum(CHALO_TYPE1);
  bounce_back_pass1();
  CCOM_halo_sum(CHALO_TYPE2);

  /* Might try to miss these two lines if b1, b2 == 0 */
  mass_conservation_compute_force();
  CCOM_halo_sum(CHALO_TYPE8);

  update_colloids();
  bounce_back_pass2();

  TIMER_stop(TIMER_BBL);

  return;
}

/*****************************************************************************
*
*  mass_conservation_compute_force
*
******************************************************************************/

static void mass_conservation_compute_force() {

  int    ia;
  int    ic, jc, kc;
  double dm;
  double c[3];
  double rbxc[3];

  Colloid        * p_colloid;
  colloid_link_t * p_link;

  for (ic = 0; ic <= Ncell(X)+1 ; ic++) {
    for (jc = 0; jc <= Ncell(Y)+1 ; jc++) {
      for (kc = 0; kc <= Ncell(Z)+1 ; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	/* For each colloid in the list */

	while (p_colloid != NULL) {

	  p_colloid->sump /= p_colloid->sumw;
	  p_link = p_colloid->lnk;

	  while (p_link != NULL) {

	    if (p_link->status == LINK_UNUSED) {
	      /* ignore */
	    }
	    else {
		
	      if (p_link->status == LINK_FLUID) {

		dm = -wv[p_link->p]*p_colloid->sump;

		for (ia = 0; ia < 3; ia++) {
		  c[ia] = 1.0*cv[p_link->p][ia];
		}

		cross_product(p_link->rb, c, rbxc);

		for (ia = 0; ia < 3; ia++) {
		  p_colloid->fc0[ia] += dm*c[ia];
		  p_colloid->tc0[ia] += dm*rbxc[ia];
		}
	      }
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

  return;
}

/*****************************************************************************
 *
 *  bounce_back_pass1
 *
 *****************************************************************************/

static void bounce_back_pass1() {

  Colloid   * p_colloid;
  colloid_link_t * p_link;

  double    c[3];
  double    rbxc[3];

  int       ia;
  int       i, j, ij, ji;
  double     dm;
  double     delta;
  double     rsumw;
  double rho0 = colloid_rho0();

  int       ic, jc, kc;

  for (ic = 0; ic <= Ncell(X) + 1; ic++)
    for (jc = 0; jc <= Ncell(Y) + 1; jc++)
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

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
	      i = p_link->i;        /* index site i (inside) */
	      j = p_link->j;        /* index site j (outside) */
	      ij = p_link->p;       /* link velocity index i->j */
	      ji = NVEL - ij;      /* link velocity index j->i */

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
		  double rmod, dm_a, cost, plegendre, sint;
		  double tans[3], vector1[3];
		  double fdist;

		  rmod = 1.0/modulus(p_link->rb);
		  cost = rmod*dot_product(p_link->rb, p_colloid->s.m);
		  sint = sqrt(1.0 - cost*cost);

		  cross_product(p_link->rb, p_colloid->s.m, vector1);
		  cross_product(vector1, p_link->rb, tans);

		  rmod = modulus(tans);
		  if (rmod != 0.0) rmod = 1.0/rmod;
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

  return;
}


/*****************************************************************************
 *
 *  COLL_bounce_back_pass2
 *
 *  Implement bounce-back on links having updated the colloid
 *  velocities via the implicit method.
 *
 *  The surface stress is also accumulated here (and it really must
 *  done between the colloid velcoity update and the actual bbl).
 *  There's a separate routine to access it below.
 *
 *****************************************************************************/

static void bounce_back_pass2() {

  Colloid   * p_colloid;
  colloid_link_t * p_link;

  double wxrb[3];

  int       i, j, ij, ji;
  double     dm;
  double     vdotc;
  double     dms;
  double     df, dg;
  double    fdist;

  double     dgtm1;
  double rho0 = colloid_rho0();

  int       ic, jc, kc;
  int ia;

  /* Account the current phi deficit */
  deltag_ = 0.0;

  /* Zero the surface stress */

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      stress_[i][j] = 0.0;
    }
  }

  for (ic = 0; ic <= Ncell(X) + 1; ic++)
    for (jc = 0; jc <= Ncell(Y) + 1; jc++)
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

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

	      dg = phi_get_phi_site(i)*vdotc;
	      p_colloid->s.deltaphi += dg;
	      dg -= wv[ij]*dgtm1;

	      /* Correction owing to missing links "squeeze term" */

	      df -= wv[ij]*dms;

	      /* The outside site actually undergoes BBL. */

	      fdist = distribution_f(i, ij, 0);
	      fdist = fdist - df;
	      distribution_f_set(j, ji, 0, fdist);

	      /* This is slightly clunky. If the order parameter is
	       * via LB, bounce back with correction. */
	      if (distribution_ndist() > 1) {
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

  return;
}

/*****************************************************************************
 *
 *  update_colloids
 *
 *  Update the velocity and position of each particle.
 *
 *  This is a linear algebra problem, which is always 6x6, and is
 *  solved using a bog-standard Gaussian elimination with partial
 *  pivoting, followed by backsubstitution.
 *
 *****************************************************************************/

static void update_colloids() {

  Colloid   * pc;

  int ia;
  int         ic, jc, kc;

  double xb[6];
  double a[6][6];
  int   ipivot[6];
  int   iprow = 0;                 /* The pivot row */
  int   idash, j, k;

  double mass;
  double moment;
  double tmp;
  double rho0 = colloid_rho0();

  /* Loop round cells and update each particle velocity */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	pc = colloids_cell_list(ic, jc, kc);

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
	    pc->s.dr[ia] = 0.5*(pc->s.v[ia] + xb[ia]);
	    pc->s.v[ia] = xb[ia];
	    pc->s.w[ia] = xb[3+ia];
	  }

	  rotate_vector(pc->s.m, xb + 3);
	  rotate_vector(pc->s.s, xb + 3);

	  /* Record the actual hyrdrodynamic force on the particle */

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

  return;
}

/******************************************************************************
 *
 *  wall_lubrication
 *
 *  For a given colloid, add any appropriate particle-wall lubrication
 *  force. Again, this relies on the fact that the walls have no
 *  component of velocity nornal to their own plane.
 *
 *  The result should be added to the appropriate diagonal element of
 *  the colloid's drag matrix in the implicit update.
 *
 *  Issues
 *    Normal force is added to the diagonal of drag matrix \zeta^FU_zz
 *    Tangential force to \zeta^FU_xx and \zeta^FU_yy
 *
 *****************************************************************************/

double wall_lubrication(const int dim, const double r[3], const double ah) {

  double force;
  double hlub;
  double h;

  force = 0.0;  /* no force by default */
  hlub = 0.5;   /* half a lattice spacing cut off */

  if (dim == Z && is_periodic(Z) == 0) {
    /* Lower, then upper */
    h = r[Z] - Lmin(Z) - ah; 
    if (h < hlub) force = -6.0*pi_*get_eta_shear()*ah*ah*(1.0/h - 1.0/hlub);
    h = Lmin(Z) + L(Z) - r[Z] - ah;
    if (h < hlub) force = -6.0*pi_*get_eta_shear()*ah*ah*(1.0/h - 1.0/hlub);
  }

  return force;
}

/*****************************************************************************
 *
 *  get_order_parameter_deficit
 *
 *  Returns the current order parameter deficit owing to BBL.
 *
 *****************************************************************************/

double bbl_order_parameter_deficit() {
  return deltag_;
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
