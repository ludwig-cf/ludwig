/*****************************************************************************
 *
 *  bbl.c
 *
 *  Bounce back on links.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *  Contributing Authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Squimer code from Isaac Llopis and Ricard Matas Navarro (U. Barcelona).
 *  Ellipsoids (including active ellipsoids) by Sumesh Thampi.
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
#include "util.h"
#include "util_ellipsoid.h"
#include "util_vector.h"
#include "wall.h"
#include "bbl.h"
#include "colloid.h"
#include "colloids.h"

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
static int bbl_active_conservation(bbl_t * bbl, lb_t * lb,
				   colloids_info_t * cinfo);
static int bbl_wall_lubrication_account(bbl_t * bbl, wall_t * wall,
					colloids_info_t * cinfo);

__global__ void bbl_pass0_kernel(kernel_ctxt_t * ktxt, cs_t * cs, lb_t * lb,
				 colloids_info_t * cinfo);

static __constant__ lb_collide_param_t lbp;

int bbl_update_colloid_default(bbl_t * bbl, wall_t * wall, colloid_t * pc,
			       double rho0, double xb[6]);
int bbl_update_ellipsoid(bbl_t * bbl, wall_t * wall, colloid_t * pc,
			 double rho0, double xb[6]);

void setter_ladd_ellipsoid(colloid_t *pc, wall_t * wall,double rho0,
			   double a[6][6], double xb[6]);
void record_force_torque(colloid_t * pc);

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
 *  Set a single global flag to see if any active particles are present.
 *  If there is none, we can avoid the additional communication steps
 *  associated with active particles.
 *
 *****************************************************************************/

int bbl_active_set(bbl_t * bbl, colloids_info_t * cinfo) {

  int nactive = 0;
  int nactive_local = 0;
  MPI_Comm comm = MPI_COMM_NULL;
  colloid_t * pc = NULL;

  assert(bbl);
  assert(cinfo);

  colloids_info_local_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextlocal) {
    if (pc->s.active) nactive_local += 1;
  }

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
    bbl_active_conservation(bbl, lb, cinfo);
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

static int bbl_active_conservation(bbl_t * bbl, lb_t * lb,
				   colloids_info_t * cinfo) {
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

    if (pc->s.active == 0) continue;

    pc->sump /= pc->sumw;
    p_link = pc->lnk;

    for (; p_link; p_link = p_link->next) {

      if (p_link->status != LINK_FLUID) continue;

      dm = -lb->model.wv[p_link->p]*pc->sump;

      for (ia = 0; ia < 3; ia++) {
	c[ia] = 1.0*lb->model.cv[p_link->p][ia];
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
  LB_CS2_DOUBLE(cs2);
  LB_RCS2_DOUBLE(rcs2);

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

    if (pc && (pc->s.bc == COLLOID_BC_BBL)) {
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

      for (p = 1; p < lbp.nvel; p++) {
	udotc = lbp.cv[p][X]*ub[X] + lbp.cv[p][Y]*ub[Y] + lbp.cv[p][Z]*ub[Z];
	sdotq = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    double dab = (ia == ib);
	    sdotq += (lbp.cv[p][ia]*lbp.cv[p][ib] - cs2*dab)*ub[ia]*ub[ib];
	  }
	}

	lb->f[ LB_ADDR(lb->nsite, lb->ndist, lbp.nvel, index, LB_RHO, p) ]
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
  double mod, rmod, cost, plegendre, sint;
  double tans[3], vector1[3];
  double fdist;
  LB_RCS2_DOUBLE(rcs2);

  double *elabc;
  double elc;
  double ele,ele2;
  double ela,ela2;
  double elz,elz2;

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

    if (pc->s.bc != COLLOID_BC_BBL) continue;


    elabc = pc->s.elabc;
    elc = sqrt(elabc[0]*elabc[0] - elabc[1]*elabc[1]);
    ele = elc/elabc[0];
    ela = colloid_principal_radius(&pc->s);

    /* Diagnostic record of f0 before additions are made. */
    /* Really, f0 should not be used for dual purposes... */

    pc->diagnostic.fbuild[X] = pc->f0[X];
    pc->diagnostic.fbuild[Y] = pc->f0[Y];
    pc->diagnostic.fbuild[Z] = pc->f0[Z];

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

      i = p_link->i;              /* index site i (outside) */
      j = p_link->j;              /* index site j (inside) */
      ij = p_link->p;             /* link velocity index i->j */
      ji = lb->model.nvel - ij;   /* link velocity index j->i */

      assert(ij > 0 && ij < lb->model.nvel);

      /* For stationary link, the momentum transfer from the
       * fluid to the colloid is "dm" */

      if (p_link->status == LINK_FLUID) {
	/* Bounce back of fluid on outside plus correction
	 * arising from changes in shape at previous step.
	 * Note minus sign. */

	double dm_a = 0.0;

	lb_f(lb, i, ij, 0, &fdist);
	dm =  2.0*fdist - lb->model.wv[ij]*pc->deltam;
	delta = 2.0*rcs2*lb->model.wv[ij]*rho0;


	/* Squirmer section */

	if (pc->s.active && pc->s.shape == COLLOID_SHAPE_SPHERE) {

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

	  /* Compute correction to bbl for a sphere: */
	  dm_a = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    dm_a += -delta*plegendre*rmod*tans[ia]*lb->model.cv[ij][ia];
	  }
	}

	/* Ellipsoidal squirmer */

	if (pc->s.active && pc->s.shape == COLLOID_SHAPE_ELLIPSOID) {
	  double elr, sdotez;
	  double *elbz;
	  double denom, term1, term2;
	  double elrho[3], xi1, xi2, xi;
	  double diff1, diff2, gridin[3], elzin;

	  /* This is the tangent calculation, which might be replaced
	   * by the surface_tanget function ... to be confirmed ... */
	  elbz = pc->s.m;
	  elz = dot_product(p_link->rb, elbz);
	  for (ia = 0; ia < 3; ia++) {
	    elrho[ia] = p_link->rb[ia] - elz*elbz[ia];
	  }

	  elr = modulus(elrho);
	  rmod = 0.0;
	  if (elr != 0.0) rmod = 1.0/elr;
	  for (ia = 0; ia < 3; ia++) {
	    elrho[ia] = elrho[ia]*rmod;
	  }
	  ela2 = ela*ela;
	  elz2 = elz*elz;
	  ele2 = ele*ele;
	  diff1 = ela2-elz2;
	  diff2 = ela2-ele2*elz2;

	  /* Taking care of the unusual circumstances in which the grid
	   * point lies outside the particle and elz > ela. Then the
	   * tangent vector is calculated for the neighbouring grid
	   * point inside*/

	  if (diff1 < 0.0) {
	    for (ia = 0; ia < 3; ia++) {
	      gridin[ia] = p_link->rb[ia]+lb->model.cv[ij][ia];
	      elzin = dot_product(gridin, elbz);
	      elz2 = elzin*elzin;
	      diff1 = ela2-elz2;
	    }
	    /* diff1 is a more stringent criterion */
	    if (diff2 < 0.0) diff2 = ela2 - ele2*elz2;
	  }
	  denom = sqrt(diff2);
	  term1 = -sqrt(diff1)/denom;
	  term2 = sqrt(1.0-ele*ele)*elz/denom;
	  for (ia = 0; ia < 3; ia++) {
	    tans[ia] = term1*elbz[ia] + term2*elrho[ia];
	  }
	  sdotez = dot_product(tans, elbz);
	  xi1 = sqrt(elr*elr+(elz+elc)*(elz+elc));
	  xi2 = sqrt(elr*elr+(elz-elc)*(elz-elc));
	  xi = (xi1 - xi2)/(2.0*elc);

	  plegendre = -(pc->s.b1)*sdotez - (pc->s.b2)*xi*sdotez;

	  mod = modulus(tans);
	  rmod = 0.0;
	  if (mod != 0.0) rmod = 1.0/mod;

	  /* Compute contribution to bbl - dm_a - for an ellipsoid */
	  dm_a = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    dm_a += -delta*plegendre*rmod*tans[ia]*lb->model.cv[ij][ia];
	  }
	}

	lb_f(lb, i, ij, 0, &fdist);
	fdist += dm_a;
	lb_f_set(lb, i, ij, 0, fdist);

	dm += dm_a;

	/* needed for mass conservation   */
	pc->sump += dm_a;
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
	c[ia] = 1.0*lb->model.cv[ij][ia];
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
  LB_RCS2_DOUBLE(rcs2);

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

    if (pc->s.bc != COLLOID_BC_BBL) continue;

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

      i = p_link->i;              /* index site i (outside) */
      j = p_link->j;              /* index site j (inside) */
      ij = p_link->p;             /* link velocity index i->j */
      ji = lb->model.nvel - ij;   /* link velocity index j->i */

      if (p_link->status == LINK_FLUID) {

	lb_f(lb, i, ij, 0, &fdist);
	dm =  2.0*fdist - lb->model.wv[ij]*pc->deltam;

	/* Compute the self-consistent boundary velocity,
	 * and add the correction term for changes in shape. */

	cross_product(pc->s.w, p_link->rb, wxrb);

	vdotc = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  vdotc += (pc->s.v[ia] + wxrb[ia])*lb->model.cv[ij][ia];
	}
	vdotc = 2.0*rcs2*lb->model.wv[ij]*vdotc;
	df = rho0*vdotc + lb->model.wv[ij]*pc->deltam;

	/* Contribution to mass conservation from squirmer */

	df += lb->model.wv[ij]*pc->sump;

	/* Correction owing to missing links "squeeze term" */

	df -= lb->model.wv[ij]*dms;

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
	  dg -= lb->model.wv[ij]*dgtm1;

	  lb_f(lb, i, ij, LB_PHI, &fdist);
	  fdist = fdist - dg;
	  lb_f_set(lb, j, ji, LB_PHI, fdist);
	}

	/* The stress is r_b f_b */
	for (ia = 0; ia < 3; ia++) {
	  bbl->stress[ia][X] += p_link->rb[X]*(dm - df)*lb->model.cv[ij][ia];
	  bbl->stress[ia][Y] += p_link->rb[Y]*(dm - df)*lb->model.cv[ij][ia];
	  bbl->stress[ia][Z] += p_link->rb[Z]*(dm - df)*lb->model.cv[ij][ia];
	}
      }
      else if (p_link->status == LINK_COLLOID) {

	/* The stress should include the solid->solid term */

	lb_f(lb, i, ij, 0, &fdist);
	dm = fdist;
	lb_f(lb, j, ji, 0, &fdist);
	dm += fdist;

	for (ia = 0; ia < 3; ia++) {
	  bbl->stress[ia][X] += p_link->rb[X]*dm*lb->model.cv[ij][ia];
	  bbl->stress[ia][Y] += p_link->rb[Y]*dm*lb->model.cv[ij][ia];
	  bbl->stress[ia][Z] += p_link->rb[Z]*dm*lb->model.cv[ij][ia];
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

  int iret = 0;
  double rho0;
  double xb[6] = {0};
  colloid_t * pc = NULL;

  assert(bbl);
  assert(cinfo);

  colloids_info_rho0(cinfo, &rho0);

  /* All colloids, including halo */

  colloids_info_all_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextall) {

    if (pc->s.bc != COLLOID_BC_BBL) continue;

    if (pc->s.shape == COLLOID_SHAPE_SPHERE) {
      iret = bbl_update_colloid_default(bbl, wall, pc, rho0, xb);
    }

    if (pc->s.shape == COLLOID_SHAPE_ELLIPSOID) {
      iret = bbl_update_ellipsoid(bbl, wall, pc, rho0, xb);
    }

    if (iret != 0) {
      pe_fatal(bbl->pe, "Gaussian elimination failed in bbl_update\n");
    }

    /* Set the position update, but don't actually move
     * the particles. This is deferred until the next
     * call to coll_update() and associated cell list
     * update.
     * We use mean of old and new velocity. */

    for (int ia = 0; ia < 3; ia++) {
      if (pc->s.isfixedrxyz[ia] == 0) pc->s.dr[ia] = 0.5*(pc->s.v[ia] + xb[ia]);
      if (pc->s.isfixedvxyz[ia] == 0) pc->s.v[ia] = xb[ia];
      if (pc->s.isfixedw == 0) pc->s.w[ia] = xb[3+ia];
    }

    if (pc->s.isfixeds == 0) {
      rotate_vector(pc->s.m, xb + 3);
      rotate_vector(pc->s.s, xb + 3);
    }

    /* Record the actual hydrodynamic force on the particle */
    record_force_torque(pc);

    /* Next colloid */
  }

  /* As the lubrication force is based on the updated velocity, but
   * the old position, we can account for the total momentum here. */

  bbl_wall_lubrication_account(bbl, wall, cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  bbl_update_colloids_default
 *
 *  Calculate the velocity of each particle for the default case, spheres.
 *
 *****************************************************************************/

int bbl_update_colloid_default(bbl_t * bbl, wall_t * wall, colloid_t * pc,
			       double rho0, double xb[6]) {

  int ia;
  int iret = 0;

  double mass;    /* Assumes (4/3) rho pi r^3 */
  double moment;  /* also assumes (2/5) mass r^2 for sphere */
  double dwall[3];
  double a[6][6];

  PI_DOUBLE(pi);

  assert(bbl);
  assert(wall);
  assert(pc);

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

  iret = bbl_6x6_gaussian_elimination(a, xb);

  return iret;
}

/*****************************************************************************
 *
 *  bbl_update_ellipsoid
 *
 *****************************************************************************/

int bbl_update_ellipsoid(bbl_t * bbl, wall_t * wall, colloid_t * pc,
			 double rho0, double xb[6]) {

  int iret = 0;

  double a[6][6];
  double quaternext[4];
  double owathalf[3];
  double qbar[4];
  double v1[3]={1.0,0.0,0.0};

  assert(bbl);
  assert(wall);
  assert(pc);

  /* Set up the matrix problem and solve it here. */

  setter_ladd_ellipsoid(pc, wall, rho0, a, xb);
  iret = bbl_6x6_gaussian_elimination(a, xb);

  /* And then finding the new quaternions */

  for (int i = 0; i < 3; i++) owathalf[i] = 0.5*(pc->s.w[i]+xb[3+i]);

  if (pc->s.isfixeds == 0) {
    util_q4_from_omega(owathalf, 0.5, qbar);
    util_q4_product(qbar, pc->s.quater, quaternext);
    util_vector_copy(4, pc->s.quater, pc->s.quaterold);
    util_vector_copy(4, quaternext, pc->s.quater);
  }

  /* Re-orient swimming direction */

  util_q4_rotate_vector(pc->s.quater, v1, pc->s.m);

  return iret;
}

/*****************************************************************************
 *
 *  Record force torque
 *
*****************************************************************************/
__host__ void record_force_torque(colloid_t *pc){

  assert(pc);

    pc->diagnostic.fhydro[X] = pc->f0[X]
      -(pc->zeta[0]*pc->s.v[X] +
	pc->zeta[1]*pc->s.v[Y] +
	pc->zeta[2]*pc->s.v[Z] +
	pc->zeta[3]*pc->s.w[X] +
	pc->zeta[4]*pc->s.w[Y] +
	pc->zeta[5]*pc->s.w[Z]);
    pc->diagnostic.fhydro[Y] = pc->f0[Y]
      -(pc->zeta[ 1]*pc->s.v[X] +
	pc->zeta[ 6]*pc->s.v[Y] +
	pc->zeta[ 7]*pc->s.v[Z] +
	pc->zeta[ 8]*pc->s.w[X] +
	pc->zeta[ 9]*pc->s.w[Y] +
	pc->zeta[10]*pc->s.w[Z]);
    pc->diagnostic.fhydro[Z] = pc->f0[Z]
      -(pc->zeta[ 2]*pc->s.v[X] +
	pc->zeta[ 7]*pc->s.v[Y] +
	pc->zeta[11]*pc->s.v[Z] +
	pc->zeta[12]*pc->s.w[X] +
	pc->zeta[13]*pc->s.w[Y] +
	pc->zeta[14]*pc->s.w[Z]);
    pc->diagnostic.Thydro[X] = pc->t0[X]
      -(pc->zeta[3]*pc->s.v[X] +
	pc->zeta[8]*pc->s.v[Y] +
	pc->zeta[12]*pc->s.v[Z] +
	pc->zeta[15]*pc->s.w[X] +
	pc->zeta[16]*pc->s.w[Y] +
	pc->zeta[17]*pc->s.w[Z]);
    pc->diagnostic.Thydro[Y] = pc->t0[Y]
      -(pc->zeta[ 4]*pc->s.v[X] +
	pc->zeta[ 9]*pc->s.v[Y] +
	pc->zeta[13]*pc->s.v[Z] +
	pc->zeta[16]*pc->s.w[X] +
	pc->zeta[18]*pc->s.w[Y] +
	pc->zeta[19]*pc->s.w[Z]);
    pc->diagnostic.Thydro[Z] = pc->t0[Z]
      -(pc->zeta[ 5]*pc->s.v[X] +
	pc->zeta[10]*pc->s.v[Y] +
	pc->zeta[14]*pc->s.v[Z] +
	pc->zeta[17]*pc->s.w[X] +
	pc->zeta[19]*pc->s.w[Y] +
	pc->zeta[20]*pc->s.w[Z]);

    /* Copy non-hydrodynamic contribution for the diagnostic record. */

    pc->diagnostic.fnonhy[X] = pc->force[X];
    pc->diagnostic.fnonhy[Y] = pc->force[Y];
    pc->diagnostic.fnonhy[Z] = pc->force[Z];

return;
}

/*****************************************************************************
 *
 *  Setting up  6 x 6 equations of Ladd for an ellipsoid
 *
*****************************************************************************/
void setter_ladd_ellipsoid(colloid_t *pc, wall_t * wall, double rho0, double a[6][6], double xb[6]) {

  double mass;    /* Assumes (4/3) rho pi abc */
  double mI_P[3];  /* also assumes that for an ellipsoid */
  double mI[3][3];  /* also assumes that for an ellipsoid */
  double mIold[3][3];
  double dIijdt[3][3];
  double dwall[3]={0.0,0.0,0.0};
  /*Flag = 0 dI/dt using quaternions d/dt(qqIqq)) */
  /*Flag = 1 dI/dt from previous time step, I(t)-I(t-\Delta t)*/
  int ddtmI_fd_flag = 0;

  double * elabc;
  double * zeta;
  double frn = 1.0;

  double IijOj;

  assert(pc);
  PI_DOUBLE(pi);

  zeta=pc->zeta;
   /* Mass and moment of inertia are those of a hard ellipsoid*/

  elabc=pc->s.elabc;
  mass = (4.0/3.0)*pi*rho0*elabc[0]*elabc[1]*elabc[2];
  mI_P[0] = (1.0/5.0)*mass*(pow(elabc[1],2)+pow(elabc[2],2));
  mI_P[1] = (1.0/5.0)*mass*(pow(elabc[0],2)+pow(elabc[2],2));
  mI_P[2] = (1.0/5.0)*mass*(pow(elabc[0],2)+pow(elabc[1],2));
  inertia_tensor_quaternion(pc->s.quater, mI_P, mI);

  wall_lubr_sphere(wall, pc->s.ah, pc->s.r, dwall);
  /* Add inertial terms to diagonal elements */

  a[0][0] = (mass/frn) +   zeta[0] - dwall[X];
  a[0][1] =          zeta[1];
  a[0][2] =          zeta[2];
  a[0][3] =          zeta[3];
  a[0][4] =          zeta[4];
  a[0][5] =          zeta[5];
  a[1][1] = (mass/frn) +   zeta[6] - dwall[Y];
  a[1][2] =          zeta[7];
  a[1][3] =          zeta[8];
  a[1][4] =          zeta[9];
  a[1][5] =          zeta[10];
  a[2][2] = (mass/frn) +   zeta[11] - dwall[Z];
  a[2][3] =          zeta[12];
  a[2][4] =          zeta[13];
  a[2][5] =          zeta[14];
  a[3][3] = (mI[0][0]/frn) + zeta[15];
  a[3][4] = (mI[0][1]/frn) + zeta[16];
  a[3][5] = (mI[0][2]/frn) + zeta[17];
  a[4][4] = (mI[1][1]/frn) + zeta[18];
  a[4][5] = (mI[1][2]/frn) + zeta[19];
  a[5][5] = (mI[2][2]/frn) + zeta[20];

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

  /*Add unsteady moment of inertia terms - pick one of the method*/
  if (ddtmI_fd_flag == 1) {
    inertia_tensor_quaternion(pc->s.quaterold, &mI_P[0], mIold);
    for (int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
        dIijdt[i][j] = (mI[i][j] - mIold[i][j]);
      }
    }
  }
  else {
    unsteady_mI(pc->s.quater, mI_P, pc->s.w, dIijdt);
  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      a[3+i][3+j] += dIijdt[i][j];
    }
  }

  /* Form the right-hand side */

  for (int ia = 0; ia < 3; ia++) {
    xb[ia] = (mass/frn)*pc->s.v[ia] + pc->f0[ia] + pc->force[ia];
    IijOj  = 0.0;
    for (int j = 0; j < 3; j++) {
      IijOj += mI[ia][j]*pc->s.w[j];
    }
    xb[3+ia] = IijOj/frn + pc->t0[ia] + pc->torque[ia];
  }

  /* Contribution to mass conservation from squirmer */

  for (int ia = 0; ia < 3; ia++) {
    xb[ia] += pc->fc0[ia];
    xb[3+ia] += pc->tc0[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  bbl_6x6_gaussian_elimination
 *
 *  Gaussian elimination for 6 x 6 system of equations
 *
 *  a[6][6] is destroyed on exit
 *  xb[6] is the rhs on entry and the solution on successful exit.
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

int bbl_6x6_gaussian_elimination(double a[6][6], double xb[6]) {

  int ipivot[6];
  int iprow = 0;
  int idash,j,k;
  double tmp;

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
        return -1;
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
    if (pc->s.bc != COLLOID_BC_BBL) continue;
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
