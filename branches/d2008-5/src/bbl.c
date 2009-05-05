/*****************************************************************************
 *
 *  bbl.c
 *
 *  Bounce back on links.
 *
 *  $Id: bbl.c,v 1.9 2009-04-09 17:07:12 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
#include "phi.h"
#include "bbl.h"

extern Site * site;

static void bounce_back_pass1(void);
static void bounce_back_pass2(void);
static void update_colloids(void);
static double WALL_lubrication(const int, FVector, double);

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

  if (get_N_colloid() == 0) return;

  TIMER_start(TIMER_BBL);

  CCOM_halo_sum(CHALO_TYPE1);
  bounce_back_pass1();
  CCOM_halo_sum(CHALO_TYPE2);
  update_colloids();
  bounce_back_pass2();

  TIMER_stop(TIMER_BBL);

  return;
}

/*****************************************************************************
 *
 *  bounce_back_pass1
 *
 *****************************************************************************/

static void bounce_back_pass1() {

  Colloid   * p_colloid;
  COLL_Link * p_link;

  FVector   ci;
  FVector   vb;

  int       i, j, ij, ji;
  double     dm;
  double     delta;
  double     rsumw;
  double rho0 = get_colloid_rho0();

  int       ic, jc, kc;

  for (ic = 0; ic <= Ncell(X) + 1; ic++)
    for (jc = 0; jc <= Ncell(Y) + 1; jc++)
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

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

	    if (p_link->status == LINK_UNUSED) {
	      /* ignore */
	    }
	    else {
	      i = p_link->i;        /* index site i (inside) */
	      j = p_link->j;        /* index site j (outside) */
	      ij = p_link->v;       /* link velocity index i->j */
	      ji = NVEL - ij;      /* link velocity index j->i */

	      ci.x = (double) cv[ij][0];
	      ci.y = (double) cv[ij][1];
	      ci.z = (double) cv[ij][2];

	      /* For stationary link, the momentum transfer from the
	       * fluid to the colloid is "dm" */

	      if (p_link->status == LINK_FLUID) {
		/* Bounce back of fluid on outside plus correction
		 * arising from changes in shape at previous step. */
		dm =  2.0*site[i].f[ij]
		  - wv[ij]*p_colloid->deltam; /* minus */
		delta = 2.0*rcs2*wv[ij]*rho0;
#ifdef _ACTIVE2_
		{
		  double rbmod, dm_a, cost, plegendre;
		  double tans[3];
		  FVector va;

		  rbmod = 1.0/UTIL_fvector_mod(p_link->rb);

		  cost = rbmod*UTIL_dot_product(p_link->rb, p_colloid->dir);
		  tans[X] = p_colloid->dir.x - cost*rbmod*p_link->rb.x;
		  tans[Y] = p_colloid->dir.y - cost*rbmod*p_link->rb.y;
		  tans[Z] = p_colloid->dir.z - cost*rbmod*p_link->rb.z;
	
		  rbmod = 1.0/p_colloid->n1_nodes;
		  plegendre = 0.5*(3.0*cost*cost - 1.0);
		  va.x = p_colloid->dp*tans[X]*plegendre*rbmod;
		  va.y = p_colloid->dp*tans[Y]*plegendre*rbmod;
		  va.z = p_colloid->dp*tans[Z]*plegendre*rbmod;
		  
		  dm_a = delta*UTIL_dot_product(va, ci);
		  site[i].f[ij] -= dm_a;
		  dm -= dm_a;
		}
#endif
	      }
	      else {
		/* Virtual momentum transfer for solid->solid links,
		 * but no contribution to drag maxtrix */
		dm = site[i].f[ij] + site[j].f[ji];
		delta = 0.0;
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
  COLL_Link * p_link;

  FVector   vb;
  FVector   ci;

  int       i, j, ij, ji;
  double     dm;
  double     vdotc;
  double     dms;
  double     df, dg;

  double     dgtm1;
  double rho0 = get_colloid_rho0();

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	/* Update solid -> fluid links for each colloid in the list */

	while (p_colloid != NULL) {

	  /* Set correction for phi arising from previous step */

	  dgtm1 = p_colloid->deltaphi;

	  p_colloid->deltaphi = 0.0;

	  /* Correction to the bounce-back for this particle if it is
	   * without full complement of links */

	  dms = 2.0*rcs2*rho0*
	    (UTIL_dot_product(p_colloid->v, p_colloid->cbar)
	     + UTIL_dot_product(p_colloid->omega, p_colloid->rxcbar));

	  /* Run through the links */

	  p_link = p_colloid->lnk;

	  while (p_link != NULL) {

	    i = p_link->i;       /* index site i (outside) */
	    j = p_link->j;       /* index site j (inside) */
	    ij = p_link->v;      /* link velocity index i->j */
	    ji = NVEL - ij;      /* link velocity index j->i */

	    if (p_link->status == LINK_FLUID) {

	      ci.x = (double) cv[ij][0];
	      ci.y = (double) cv[ij][1];
	      ci.z = (double) cv[ij][2];

	      dm =  2.0*site[i].f[ij]
		- wv[ij]*p_colloid->deltam; /* minus */

	      /* Compute the self-consistent boundary velocity,
	       * and add the correction term for changes in shape. */

	      vb = UTIL_cross_product(p_colloid->omega, p_link->rb);
	      vb = UTIL_fvector_add(vb, p_colloid->v);

	      vdotc = 2.0*rcs2*wv[ij]*UTIL_dot_product(vb, ci);

	      df = rho0*vdotc + wv[ij]*p_colloid->deltam;

	      dg = phi_get_phi_site(i)*vdotc;
	      p_colloid->deltaphi += dg;
	      dg -= wv[ij]*dgtm1;

	      /* Correction owing to missing links "squeeze term" */

	      df -= wv[ij]*dms;

	      /* The outside site actually undergoes BBL. */

	      site[j].f[ji] = site[i].f[ij] - df;

#ifndef _SINGLE_FLUID_
	      site[j].g[ji] = site[i].g[ij] - dg;
#endif
	      /* The stress is r_b f_b */
	      for (ia = 0; ia < 3; ia++) {
		stress_[ia][0] += p_link->rb.x*(dm - df)*cv[ij][ia];
		stress_[ia][1] += p_link->rb.y*(dm - df)*cv[ij][ia];
		stress_[ia][2] += p_link->rb.z*(dm - df)*cv[ij][ia];
	      }
	    }
	    else if (p_link->status == LINK_COLLOID) {

	      /* The stress should include the solid->solid term */

	      dm = site[i].f[ij] + site[j].f[ji];

	      for (ia = 0; ia < 3; ia++) {
		stress_[ia][0] += p_link->rb.x*dm*cv[ij][ia];
		stress_[ia][1] += p_link->rb.y*dm*cv[ij][ia];
		stress_[ia][2] += p_link->rb.z*dm*cv[ij][ia];
	      }
	    }

	    /* Next link */
	    p_link = p_link->next;
	  }

	  /* Reset factors required for change of shape */
	  p_colloid->deltam = 0.0;
	  p_colloid->f0 = UTIL_fvector_zero();
	  p_colloid->t0 = UTIL_fvector_zero();
	  deltag_ += p_colloid->deltaphi;

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
 *  pivoting, followed by  backsubstitution.
 *
 *  Issues:
 *    - Could eliminate lower triangle storage (or use library routine)
 *    - Dianostics could be moved elsewhere, although this is the only
 *      place where the true force on the particle is available at the
 *      moment.
 *
 *****************************************************************************/

static void update_colloids() {

  Colloid   * p_colloid;

  int         ic, jc, kc;

  double xb[6];
  double a[6][6];
  int   ipivot[6];
  int   iprow = 0;                 /* The pivot row */
  int   idash, j, k;

  double mass;
  double moment;
  double tmp;
  double rho0 = get_colloid_rho0();

  /* Loop round cells and update each particle velocity */

  for (ic = 0; ic <= Ncell(X) + 1; ic++)
    for (jc = 0; jc <= Ncell(Y) + 1; jc++)
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  /* Set up the matrix problem and solve it here. */

	  /* Mass and moment of inertia are those of a hard sphere
	   * with the input radius */

	  mass = (4.0/3.0)*PI*rho0*pow(p_colloid->a0, 3);
	  moment = (2.0/5.0)*mass*pow(p_colloid->a0, 2);

	  /* Add inertial terms to diagonal elements */

	  a[0][0] = mass +   p_colloid->zeta[0];
	  a[0][1] =          p_colloid->zeta[1];
	  a[0][2] =          p_colloid->zeta[2];
	  a[0][3] =          p_colloid->zeta[3];
	  a[0][4] =          p_colloid->zeta[4];
	  a[0][5] =          p_colloid->zeta[5];
	  a[1][1] = mass +   p_colloid->zeta[6];
	  a[1][2] =          p_colloid->zeta[7];
	  a[1][3] =          p_colloid->zeta[8];
	  a[1][4] =          p_colloid->zeta[9];
	  a[1][5] =          p_colloid->zeta[10];
	  a[2][2] = mass +   p_colloid->zeta[11];
	  a[2][3] =          p_colloid->zeta[12];
	  a[2][4] =          p_colloid->zeta[13];
	  a[2][5] =          p_colloid->zeta[14];
	  a[3][3] = moment + p_colloid->zeta[15];
	  a[3][4] =          p_colloid->zeta[16];
	  a[3][5] =          p_colloid->zeta[17];
	  a[4][4] = moment + p_colloid->zeta[18];
	  a[4][5] =          p_colloid->zeta[19];
	  a[5][5] = moment + p_colloid->zeta[20];

	  for (k = 0; k < 3; k++) {
	    a[k][k] -= WALL_lubrication(k, p_colloid->r, p_colloid->ah);
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

	  xb[0] = mass*p_colloid->v.x  + p_colloid->f0.x + p_colloid->force.x;
	  xb[1] = mass*p_colloid->v.y  + p_colloid->f0.y + p_colloid->force.y;
	  xb[2] = mass*p_colloid->v.z  + p_colloid->f0.z + p_colloid->force.z;

	  xb[3] = moment*p_colloid->omega.x + p_colloid->t0.x
	        + p_colloid->torque.x;
	  xb[4] = moment*p_colloid->omega.y + p_colloid->t0.y
	        + p_colloid->torque.y;
	  xb[5] = moment*p_colloid->omega.z + p_colloid->t0.z
	        + p_colloid->torque.z;

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

	  p_colloid->dr[X] = 0.5*(p_colloid->v.x + xb[0]);
	  p_colloid->dr[Y] = 0.5*(p_colloid->v.y + xb[1]);
	  p_colloid->dr[Z] = 0.5*(p_colloid->v.z + xb[2]);

	  /* Unpack the solution vector. */

	  p_colloid->v.x = xb[0];
	  p_colloid->v.y = xb[1];
	  p_colloid->v.z = xb[2];

	  p_colloid->omega.x = xb[3];
	  p_colloid->omega.y = xb[4];
	  p_colloid->omega.z = xb[5];

	  p_colloid->dir =UTIL_rotate_vector(p_colloid->dir, p_colloid->omega);
	  rotate_vector(p_colloid->s, &xb[3]);

	  /* Record the actual hyrdrodynamic force on the particle */

	  p_colloid->force.x = p_colloid->f0.x
	    -(p_colloid->zeta[0]*p_colloid->v.x +
	      p_colloid->zeta[1]*p_colloid->v.y +
	      p_colloid->zeta[2]*p_colloid->v.z +
	      p_colloid->zeta[3]*p_colloid->omega.x +
	      p_colloid->zeta[4]*p_colloid->omega.y +
	      p_colloid->zeta[5]*p_colloid->omega.z);
          p_colloid->force.y = p_colloid->f0.y
	    -(p_colloid->zeta[ 1]*p_colloid->v.x +
	      p_colloid->zeta[ 6]*p_colloid->v.y +
	      p_colloid->zeta[ 7]*p_colloid->v.z +
	      p_colloid->zeta[ 8]*p_colloid->omega.x +
	      p_colloid->zeta[ 9]*p_colloid->omega.y +
	      p_colloid->zeta[10]*p_colloid->omega.z);
          p_colloid->force.z = p_colloid->f0.z
	    -(p_colloid->zeta[ 2]*p_colloid->v.x +
	      p_colloid->zeta[ 7]*p_colloid->v.y +
	      p_colloid->zeta[11]*p_colloid->v.z +
	      p_colloid->zeta[12]*p_colloid->omega.x +
	      p_colloid->zeta[13]*p_colloid->omega.y +
	      p_colloid->zeta[14]*p_colloid->omega.z);

	  p_colloid = p_colloid->next;
	}
      }

  return;
}

/******************************************************************************
 *
 *  WALL_lubrication
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

double WALL_lubrication(int dim, FVector r, double ah) {

  double force;
  double hlub;
  double h;

  force = 0.0;  /* no force by default */
  hlub = 0.5;   /* half a lattice spacing cut off */

  if (dim == Z && is_periodic(Z) == 0) {
    /* Lower, then upper */
    h = r.z - Lmin(Z) - ah; 
    if (h < hlub) force = -6.0*PI*get_eta_shear()*ah*ah*(1.0/h - 1.0/hlub);
    h = Lmin(Z) + L(Z) - r.z - ah;
    if (h < hlub) force = -6.0*PI*get_eta_shear()*ah*ah*(1.0/h - 1.0/hlub);
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

  MPI_Reduce(send, recv, 9, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

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
