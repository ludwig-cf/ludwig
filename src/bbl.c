/*****************************************************************************
 *
 *  bbl.c
 *
 *  Bounce back on links.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2018 The University of Edinburgh
 *
 *  Contributing Authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Squimer code from Isaac Llopis and Ricard Matas Navarro (U. Barcelona).
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "colloid_sums.h"
#include "model.h"
#include "lb_model_s.h"
#include "util.h"
#include "wall.h"
#include "bbl.h"
#include "colloid.h"
#include "colloids.h"
#include "colloids_s.h"

struct bbl_s {
  pe_t * pe;            /* Parallel environment */
  cs_t * cs;            /* Coordinate system */
  int active;           /* Global flag for active particles. */
  int ndist;            /* Number of LB distributions active */
  double deltag;        /* Excess or deficit of phi between steps */
  double stress[3][3];  /* Surface stress diagnostic */
};

static int bbl_pass1(bbl_t * bbl, lb_t * lb, colloids_info_t * cinfo);
static int bbl_pass2(bbl_t * bbl, lb_t * lb, colloids_info_t * cinfo);
static int bbl_active_conservation(bbl_t * bbl, colloids_info_t * cinfo);
static int bbl_wall_lubrication_account(bbl_t * bbl, wall_t * wall,
					colloids_info_t * cinfo);

__global__ void bbl_pass0_kernel(kernel_ctxt_t * ktxt, cs_t * cs, lb_t * lb,
				 colloids_info_t * cinfo);

static __constant__ lb_collide_param_t lbp;

/*****************************************************************************
 *
 *  bbl_create
 *
 *  The lattice Boltzmann distributions must be available.
 *
 *****************************************************************************/

int bbl_create(pe_t * pe, cs_t * cs, lb_t * lb, bbl_t ** pobj) {

  bbl_t * bbl = NULL;

  assert(pe);
  assert(cs);
  assert(lb);
  assert(pobj);

  bbl = (bbl_t *) calloc(1, sizeof(bbl_t));
  assert(bbl);
  if (bbl == NULL) pe_fatal(pe, "calloc(bbl_t) failed\n");

  bbl->pe = pe;
  bbl->cs = cs;
  lb_ndist(lb, &bbl->ndist);

  *pobj = bbl;

  return 0;
}

/*****************************************************************************
 *
 *  bbl_free
 *
 *****************************************************************************/

int bbl_free(bbl_t * bbl) {

  assert(bbl);

  free(bbl);

  return 0;
}

/*****************************************************************************
 *
 *  bbl_active_set
 *
 *****************************************************************************/

int bbl_active_set(bbl_t * bbl, colloids_info_t * cinfo) {

  int nactive;
  int nactive_local;
  MPI_Comm comm;

  assert(bbl);
  assert(cinfo);

  colloids_info_count_local(cinfo, COLLOID_TYPE_ACTIVE, &nactive_local);

  cs_cart_comm(bbl->cs, &comm);
  MPI_Allreduce(&nactive_local, &nactive, 1, MPI_INT, MPI_SUM, comm);

  bbl->active = nactive;

  return 0;
}

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

__host__
int bounce_back_on_links(bbl_t * bbl, lb_t * lb, wall_t * wall,
			 colloids_info_t * cinfo) {

  int ntotal;
  int nlocal[3];

  assert(bbl);
  assert(lb);
  assert(cinfo);

  cs_nlocal(bbl->cs, nlocal);

  colloids_info_ntotal(cinfo, &ntotal);
  if (ntotal == 0) return 0;

  colloid_sums_halo(cinfo, COLLOID_SUM_STRUCTURE);

  bbl_pass0(bbl, lb, cinfo);

  /* __NVCC__ TODO: remove */
  lb_memcpy(lb, tdpMemcpyDeviceToHost);

  bbl_pass1(bbl, lb, cinfo);

  colloid_sums_halo(cinfo, COLLOID_SUM_DYNAMICS);

  if (bbl->active) {
    bbl_active_conservation(bbl, cinfo);
    colloid_sums_halo(cinfo, COLLOID_SUM_ACTIVE);
  }

  bbl_update_colloids(bbl, wall, cinfo);

  bbl_pass2(bbl, lb, cinfo);

  /* __NVCC__ TODO: remove */
  lb_memcpy(lb, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  bbl_active_conservation
 *
 *****************************************************************************/

static int bbl_active_conservation(bbl_t * bbl, colloids_info_t * cinfo) {

  int ia;
  double dm;
  double c[3];
  double rbxc[3];

  colloid_t * pc;
  colloid_link_t * p_link;

  assert(bbl);
  assert(cinfo);

  colloids_info_all_head(cinfo, &pc);

  /* For each colloid in the list */

  for ( ; pc; pc = pc->nextall) {

    pc->sump /= pc->sumw;
    p_link = pc->lnk;

    for (; p_link; p_link = p_link->next) {

      if (p_link->status != LINK_FLUID) continue;

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

  return 0;
}

/*****************************************************************************
 *
 *  bbl_pass0
 *
 *  Set missing 'internal' distributions. Driver routine.
 *
 *****************************************************************************/

int bbl_pass0(bbl_t * bbl, lb_t * lb, colloids_info_t * cinfo) {

  int nlocal[3];
  int nextra;
  dim3 nblk, ntpb;
  cs_t * cstarget = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(bbl);
  assert(lb);
  assert(cinfo);

  cs_nlocal(bbl->cs, nlocal);
  cs_target(bbl->cs, &cstarget);

  nextra = 1;

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  tdpMemcpyToSymbol(tdpSymbol(lbp), lb->param, sizeof(lb_collide_param_t), 0,
		    tdpMemcpyHostToDevice);

  kernel_ctxt_create(bbl->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(bbl_pass0_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, cstarget, lb->target, cinfo->target);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  bbl_pass0_kernel
 *
 *  Set missing 'internal' distributions.
 *
 *****************************************************************************/

__global__ void bbl_pass0_kernel(kernel_ctxt_t * ktxt, cs_t * cs, lb_t * lb,
				 colloids_info_t * cinfo) {

  int kindex;
  int kiter;
  const double rcs2 = 3.0;

  assert(ktxt);
  assert(cs);
  assert(lb);
  assert(cinfo);

  kiter = kernel_iterations(ktxt);

  for_simt_parallel(kindex, kiter, 1) {

    int ic, jc, kc, index;
    int ia, ib, p;
    int noffset[3];

    double r[3], r0[3], rb[3], ub[3];
    double udotc, sdotq;

    colloid_t * pc = NULL;

    ic = kernel_coords_ic(ktxt, kindex);
    jc = kernel_coords_jc(ktxt, kindex);
    kc = kernel_coords_kc(ktxt, kindex);

    index = kernel_coords_index(ktxt, ic, jc, kc);

    pc = cinfo->map_new[index];
	
    if (pc) { 
      cs_nlocal_offset(cs, noffset);
      r[X] = 1.0*(noffset[X] + ic);
      r[Y] = 1.0*(noffset[Y] + jc);
      r[Z] = 1.0*(noffset[Z] + kc);

      r0[X] = pc->s.r[X];
      r0[Y] = pc->s.r[Y];
      r0[Z] = pc->s.r[Z];

      cs_minimum_distance(cs, r0, r, rb);

      /* u_b = v + omega x r_b */

      ub[X] = pc->s.v[X] + pc->s.w[Y]*rb[Z] - pc->s.w[Z]*rb[Y];
      ub[Y] = pc->s.v[Y] + pc->s.w[Z]*rb[X] - pc->s.w[X]*rb[Z];
      ub[Z] = pc->s.v[Z] + pc->s.w[X]*rb[Y] - pc->s.w[Y]*rb[X];

      for (p = 1; p < NVEL; p++) {
	udotc = lbp.cv[p][X]*ub[X] + lbp.cv[p][Y]*ub[Y] + lbp.cv[p][Z]*ub[Z];
	sdotq = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    sdotq += lbp.q[p][ia][ib]*ub[ia]*ub[ib];
	  }
	}

	lb->f[ LB_ADDR(lb->nsite, lb->ndist, NVEL, index, LB_RHO, p) ]
	  = lbp.wv[p]*(1.0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  bbl_pass1
 *
 *  Work out the velocity independent terms before actual BBL takes place.
 *
 *****************************************************************************/

static int bbl_pass1(bbl_t * bbl, lb_t * lb, colloids_info_t * cinfo) {

  int ia;
  int i, j, ij, ji;

  double dm;
  double delta;
  double rsumw;
  double c[3];
  double rbxc[3];
  double rho0;
  double mod, rmod, dm_a, cost, plegendre, sint;
  double tans[3], vector1[3];
  double fdist;

  physics_t * phys = NULL;
  colloid_t * pc = NULL;
  colloid_link_t * p_link = NULL;

  assert(bbl);
  assert(lb);
  assert(cinfo);

  physics_ref(&phys);
  physics_rho0(phys, &rho0);

  /* All colloids, including halo */

  colloids_info_all_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextall) {

    p_link = pc->lnk;

    for (i = 0; i < 21; i++) {
      pc->zeta[i] = 0.0;
    }

    /* We need to normalise link quantities by the sum of weights
     * over the particle. Note that sumw cannot be zero here during
     * correct operation (implies the particle has no links). */

    rsumw = 1.0 / pc->sumw;
    for (ia = 0; ia < 3; ia++) {
      pc->cbar[ia]   *= rsumw;
      pc->rxcbar[ia] *= rsumw;
    }
    pc->deltam   *= rsumw;
    pc->s.deltaphi *= rsumw;

    /* Sum over the links */ 

    for (; p_link; p_link = p_link->next) {

      if (p_link->status == LINK_UNUSED) continue;

      i = p_link->i;        /* index site i (outside) */
      j = p_link->j;        /* index site j (inside) */
      ij = p_link->p;       /* link velocity index i->j */
      ji = NVEL - ij;       /* link velocity index j->i */

      assert(ij > 0 && ij < NVEL);

      /* For stationary link, the momentum transfer from the
       * fluid to the colloid is "dm" */

      if (p_link->status == LINK_FLUID) {
	/* Bounce back of fluid on outside plus correction
	 * arising from changes in shape at previous step.
	 * Note minus sign. */

	lb_f(lb, i, ij, 0, &fdist);
	dm =  2.0*fdist - wv[ij]*pc->deltam;
	delta = 2.0*rcs2*wv[ij]*rho0;

	/* Squirmer section */
	if (pc->s.type == COLLOID_TYPE_ACTIVE) {

	  /* We expect s.m to be a unit vector, but for floating
	   * point purposes, we must make sure here. */

	  mod = modulus(p_link->rb)*modulus(pc->s.m);
	  rmod = 0.0;
	  if (mod != 0.0) rmod = 1.0/mod;
	  cost = rmod*dot_product(p_link->rb, pc->s.m);
	  if (cost*cost > 1.0) cost = 1.0;
	  assert(cost*cost <= 1.0);
	  sint = sqrt(1.0 - cost*cost);

	  cross_product(p_link->rb, pc->s.m, vector1);
	  cross_product(vector1, p_link->rb, tans);

	  mod = modulus(tans);
	  rmod = 0.0;
	  if (mod != 0.0) rmod = 1.0/mod;
	  plegendre = -sint*(pc->s.b2*cost + pc->s.b1);

	  dm_a = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    dm_a += -delta*plegendre*rmod*tans[ia]*cv[ij][ia];
	  }

	  lb_f(lb, i, ij, 0, &fdist);
	  fdist += dm_a;
	  lb_f_set(lb, i, ij, 0, fdist);

	  dm += dm_a;

	  /* needed for mass conservation   */
	  pc->sump += dm_a;
	}
      }
      else {
	/* Virtual momentum transfer for solid->solid links,
	 * but no contribution to drag maxtrix */

	lb_f(lb, i, ij, 0, &fdist);
	dm = fdist;
	lb_f(lb, j, ji, 0, &fdist);
	dm += fdist;
	delta = 0.0;
      }

      for (ia = 0; ia < 3; ia++) {
	c[ia] = 1.0*cv[ij][ia];
      }

      cross_product(p_link->rb, c, rbxc);

      /* Now add contribution to the sums required for 
       * self-consistent evaluation of new velocities. */

      for (ia = 0; ia < 3; ia++) {
	pc->f0[ia] += dm*c[ia];
	pc->t0[ia] += dm*rbxc[ia];
	/* Corrections when links are missing (close to contact) */
	c[ia] -= pc->cbar[ia];
	rbxc[ia] -= pc->rxcbar[ia];
      }

      /* Drag matrix elements */

      pc->zeta[ 0] += delta*c[X]*c[X];
      pc->zeta[ 1] += delta*c[X]*c[Y];
      pc->zeta[ 2] += delta*c[X]*c[Z];
      pc->zeta[ 3] += delta*c[X]*rbxc[X];
      pc->zeta[ 4] += delta*c[X]*rbxc[Y];
      pc->zeta[ 5] += delta*c[X]*rbxc[Z];

      pc->zeta[ 6] += delta*c[Y]*c[Y];
      pc->zeta[ 7] += delta*c[Y]*c[Z];
      pc->zeta[ 8] += delta*c[Y]*rbxc[X];
      pc->zeta[ 9] += delta*c[Y]*rbxc[Y];
      pc->zeta[10] += delta*c[Y]*rbxc[Z];

      pc->zeta[11] += delta*c[Z]*c[Z];
      pc->zeta[12] += delta*c[Z]*rbxc[X];
      pc->zeta[13] += delta*c[Z]*rbxc[Y];
      pc->zeta[14] += delta*c[Z]*rbxc[Z];

      pc->zeta[15] += delta*rbxc[X]*rbxc[X];
      pc->zeta[16] += delta*rbxc[X]*rbxc[Y];
      pc->zeta[17] += delta*rbxc[X]*rbxc[Z];

      pc->zeta[18] += delta*rbxc[Y]*rbxc[Y];
      pc->zeta[19] += delta*rbxc[Y]*rbxc[Z];

      pc->zeta[20] += delta*rbxc[Z]*rbxc[Z];

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

static int bbl_pass2(bbl_t * bbl, lb_t * lb, colloids_info_t * cinfo) {

  int i, j, ij, ji;
  int ia;
  int ndist;

  double dm;
  double vdotc;
  double dms;
  double df, dg;
  double fdist;
  double wxrb[3];

  double dgtm1;
  double rho0;

  physics_t * phys = NULL;
  colloid_t * pc = NULL;
  colloid_link_t * p_link;


  assert(bbl);
  assert(lb);
  assert(cinfo);

  physics_ref(&phys);
  physics_rho0(phys, &rho0);

  ndist=lb->ndist;

  /* Account the current phi deficit */
  bbl->deltag = 0.0;

  /* Zero the surface stress */

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      bbl->stress[i][j] = 0.0;
    }
  }

  /* All colloids, including halo */

  colloids_info_all_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextall) {

    /* Set correction for phi arising from previous step */

    dgtm1 = pc->s.deltaphi;
    pc->s.deltaphi = 0.0;

    /* Correction to the bounce-back for this particle if it is
     * without full complement of links */

    dms = 0.0;

    for (ia = 0; ia < 3; ia++) {
      dms += pc->s.v[ia]*pc->cbar[ia];
      dms += pc->s.w[ia]*pc->rxcbar[ia];
    }

    dms = 2.0*rcs2*rho0*dms;

    /* Run through the links */

    p_link = pc->lnk;

    for ( ; p_link; p_link = p_link->next) {

      i = p_link->i;       /* index site i (outside) */
      j = p_link->j;       /* index site j (inside) */
      ij = p_link->p;      /* link velocity index i->j */
      ji = NVEL - ij;      /* link velocity index j->i */

      if (p_link->status == LINK_FLUID) {

	lb_f(lb, i, ij, 0, &fdist);
	dm =  2.0*fdist - wv[ij]*pc->deltam;

	/* Compute the self-consistent boundary velocity,
	 * and add the correction term for changes in shape. */

	cross_product(pc->s.w, p_link->rb, wxrb);

	vdotc = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  vdotc += (pc->s.v[ia] + wxrb[ia])*cv[ij][ia];
	}
	vdotc = 2.0*rcs2*wv[ij]*vdotc;
	df = rho0*vdotc + wv[ij]*pc->deltam;

	/* Contribution to mass conservation from squirmer */

	df += wv[ij]*pc->sump; 

	/* Correction owing to missing links "squeeze term" */

	df -= wv[ij]*dms;

	/* The outside site actually undergoes BBL. */

	lb_f(lb, i, ij, LB_RHO, &fdist);
	fdist = fdist - df;
	lb_f_set(lb, j, ji, LB_RHO, fdist);

	/* This is slightly clunky. If the order parameter is
	 * via LB, bounce back with correction. */

	if (ndist > 1) {
	  lb_0th_moment(lb, i, LB_PHI, &dg);
	  dg *= vdotc;
	  pc->s.deltaphi += dg;
	  dg -= wv[ij]*dgtm1;

	  lb_f(lb, i, ij, LB_PHI, &fdist);
	  fdist = fdist - dg;
	  lb_f_set(lb, j, ji, LB_PHI, fdist);
	}

	/* The stress is r_b f_b */
	for (ia = 0; ia < 3; ia++) {
	  bbl->stress[ia][X] += p_link->rb[X]*(dm - df)*cv[ij][ia];
	  bbl->stress[ia][Y] += p_link->rb[Y]*(dm - df)*cv[ij][ia];
	  bbl->stress[ia][Z] += p_link->rb[Z]*(dm - df)*cv[ij][ia];
	}
      }
      else if (p_link->status == LINK_COLLOID) {

	/* The stress should include the solid->solid term */

	lb_f(lb, i, ij, 0, &fdist);
	dm = fdist;
	lb_f(lb, j, ji, 0, &fdist);
	dm += fdist;

	for (ia = 0; ia < 3; ia++) {
	  bbl->stress[ia][X] += p_link->rb[X]*dm*cv[ij][ia];
	  bbl->stress[ia][Y] += p_link->rb[Y]*dm*cv[ij][ia];
	  bbl->stress[ia][Z] += p_link->rb[Z]*dm*cv[ij][ia];
	}
      }
      /* Next link */
    }

    /* Reset factors required for change of shape, etc */

    pc->deltam = 0.0;
    pc->sump = 0.0;

    for (ia = 0; ia < 3; ia++) {
      pc->f0[ia] = 0.0;
      pc->t0[ia] = 0.0;
      pc->fc0[ia] = 0.0;
      pc->tc0[ia] = 0.0;
    }

    bbl->deltag += pc->s.deltaphi;
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

int bbl_update_colloids(bbl_t * bbl, wall_t * wall, colloids_info_t * cinfo) {

  int ia;
  int ipivot[6];
  int iprow = 0;
  int idash, j, k;

  double mass;    /* Assumes (4/3) rho pi r^3 */
  double moment;  /* also assumes (2/5) mass r^2 for sphere */
  double tmp;
  double rho0;
  double dwall[3];
  double xb[6];
  double a[6][6];

  colloid_t * pc;
  PI_DOUBLE(pi);

  assert(bbl);
  assert(cinfo);

  colloids_info_rho0(cinfo, &rho0);

  /* All colloids, including halo */

  colloids_info_all_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextall) {

    /* Set up the matrix problem and solve it here. */

    /* Mass and moment of inertia are those of a hard sphere
     * with the input radius */

    mass = (4.0/3.0)*pi*rho0*pow(pc->s.a0, 3);
    moment = (2.0/5.0)*mass*pow(pc->s.a0, 2);

    /* Wall lubrication correction */
    wall_lubr_sphere(wall, pc->s.ah, pc->s.r, dwall);

    /* Add inertial terms to diagonal elements */

    a[0][0] = mass +   pc->zeta[0] - dwall[X];
    a[0][1] =          pc->zeta[1];
    a[0][2] =          pc->zeta[2];
    a[0][3] =          pc->zeta[3];
    a[0][4] =          pc->zeta[4];
    a[0][5] =          pc->zeta[5];
    a[1][1] = mass +   pc->zeta[6] - dwall[Y];
    a[1][2] =          pc->zeta[7];
    a[1][3] =          pc->zeta[8];
    a[1][4] =          pc->zeta[9];
    a[1][5] =          pc->zeta[10];
    a[2][2] = mass +   pc->zeta[11] - dwall[Z];
    a[2][3] =          pc->zeta[12];
    a[2][4] =          pc->zeta[13];
    a[2][5] =          pc->zeta[14];
    a[3][3] = moment + pc->zeta[15];
    a[3][4] =          pc->zeta[16];
    a[3][5] =          pc->zeta[17];
    a[4][4] = moment + pc->zeta[18];
    a[4][5] =          pc->zeta[19];
    a[5][5] = moment + pc->zeta[20];

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
	pe_fatal(bbl->pe, "Gaussian elimination failed in bbl_update\n");
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
      if (pc->s.isfixedrxyz[ia] == 0) pc->s.dr[ia] = 0.5*(pc->s.v[ia] + xb[ia]);
      if (pc->s.isfixedvxyz[ia] == 0) pc->s.v[ia] = xb[ia];
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
  }

  /* As the lubrication force is based on the updated velocity, but
   * the old position, we can account for the total momentum here. */

  bbl_wall_lubrication_account(bbl, wall, cinfo);

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

static int bbl_wall_lubrication_account(bbl_t * bbl, wall_t * wall,
					colloids_info_t * cinfo) {

  double f[3] = {0.0, 0.0, 0.0};
  double dwall[3];
  colloid_t * pc = NULL;

  assert(cinfo);

  /* Local colloids */

  colloids_info_local_head(cinfo, &pc);

  for (; pc; pc = pc->nextlocal) {
    wall_lubr_sphere(wall, pc->s.ah, pc->s.r, dwall);
    f[X] -= pc->s.v[X]*dwall[X];
    f[Y] -= pc->s.v[Y]*dwall[Y];
    f[Z] -= pc->s.v[Z]*dwall[Z];
  }

  wall_momentum_add(wall, f);

  return 0;
}

/*****************************************************************************
 *
 *  get_order_parameter_deficit
 *
 *  Returns the current order parameter deficit owing to BBL.
 *  This is only relevant for full binary LB (ndist == 2).
 *  This is a local value for the local subdomain in parallel.
 *
 *****************************************************************************/

int bbl_order_parameter_deficit(bbl_t * bbl, double * delta) {

  assert(bbl);
  assert(delta);

  delta[0] = 0.0;
  if (bbl->ndist == 2) delta[0] = bbl->deltag;

  return 0;
}

/*****************************************************************************
 *
 *  bbl_surface_stress
 *
 *  Return the current local surface stress total.
 *  This is normalised by the volume of the system.
 *
 *****************************************************************************/

int bbl_surface_stress(bbl_t * bbl, double slocal[3][3]) {

  int ia, ib;
  double rv;
  double ltot[3];

  assert(bbl);

  cs_ltot(bbl->cs, ltot);

  rv = 1.0/(ltot[X]*ltot[Y]*ltot[Z]);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      slocal[ia][ib] = rv*bbl->stress[ia][ib];
    }
  }

  return 0;
}
