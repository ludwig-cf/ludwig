/*****************************************************************************
 *
 *  colloids_Q_tensor.c
 *
 *  Routines to set the Q tensor inside a colloid to correspond
 *  to homeotropic (normal) or planar anchoring at the surface.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Juho Lintuvuori (jlintuvu@ph.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *  
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "pe.h"
#include "build.h"
#include "coords.h"
#include "colloids.h"
#include "phi.h"
#include "io_harness.h"
#include "util.h"
#include "model.h"
#include "site_map.h"
#include "blue_phase.h"
#include "colloids_Q_tensor.h"

static int anchoring_coll_ = ANCHORING_NORMAL;
static int anchoring_wall_ = ANCHORING_NORMAL;
static double w_surface_ = 0.0; /* Anchoring strength in free energy */
static double w_surface_wall_ = 0.0; /* Anchoring strength in free energy */

/*****************************************************************************
 *
 *  colloids_q_boundary_normal
 *
 *  Find the 'true' outward unit normal at fluid site index. Note that
 *  the half-way point is not used to provide a simple unique value in
 *  the gradient calculation.
 *
 *  The unit lattice vector which is the discrete outward normal is di[3].
 *  The result is returned in unit vector dn.
 *
 *****************************************************************************/

void colloids_q_boundary_normal(const int index, const int di[3],
				double dn[3]) {
  int ia, index1;
  int noffset[3];
  int isite[3];

  double rd;
  colloid_t * pc;
  colloid_t * colloid_at_site_index(int);

  coords_index_to_ijk(index, isite);

  index1 = coords_index(isite[X] - di[X], isite[Y] - di[Y], isite[Z] - di[Z]);
  pc = colloid_at_site_index(index1);

  if (pc) {
    coords_nlocal_offset(noffset);
    for (ia = 0; ia < 3; ia++) {
      dn[ia] = 1.0*(noffset[ia] + isite[ia]);
      dn[ia] -= pc->s.r[ia];
    }

    rd = modulus(dn);
    assert(rd > 0.0);
    rd = 1.0/rd;

    for (ia = 0; ia < 3; ia++) {
      dn[ia] *= rd;
    }
  }
  else {
    /* Assume di is the true outward normal (e.g., flat wall) */
    for (ia = 0; ia < 3; ia++) {
      dn[ia] = 1.0*di[ia];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_q_boundary
 *
 *  Produce an estimate of the surface order parameter Q^0_ab for
 *  normal or planar anchoring.
 *
 *  This will depend on the outward surface normal nhat, and in the
 *  case of planar anchoring may depend on the estimate of the
 *  existing order parameter at the surface Qs_ab.
 *
 *  This planar anchoring idea follows e.g., Fournier and Galatola
 *  Europhys. Lett. 72, 403 (2005).
 *
 *****************************************************************************/

void colloids_q_boundary(const double nhat[3], double qs[3][3],
			 double q0[3][3], char site_map_status) {
  int ia, ib, ic, id;
  int anchoring;

  double qtilde[3][3];
  double amplitude;
  double  nfix[3] = {0.0, 1.0, 0.0};

  assert(site_map_status == COLLOID || site_map_status == BOUNDARY);

  anchoring = anchoring_coll_;
  if (site_map_status == BOUNDARY) anchoring = anchoring_wall_;

  amplitude = blue_phase_amplitude_compute();

  if (anchoring == ANCHORING_FIXED) blue_phase_q_uniaxial(amplitude, nfix, q0);
  if (anchoring == ANCHORING_NORMAL) blue_phase_q_uniaxial(amplitude, nhat, q0);

  if (anchoring == ANCHORING_PLANAR) {

    /* Planar: use the fluid Q_ab to find ~Q_ab */

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	qtilde[ia][ib] = qs[ia][ib] + 0.5*amplitude*d_[ia][ib];
      }
    }

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	q0[ia][ib] = 0.0;
	for (ic = 0; ic < 3; ic++) {
	  for (id = 0; id < 3; id++) {
	    q0[ia][ib] += (d_[ia][ic] - nhat[ia]*nhat[ic])*qtilde[ic][id]
	      *(d_[id][ib] - nhat[id]*nhat[ib]);
	  }
	}
	/* Return Q^0_ab = ~Q_ab - (1/2) A d_ab */
	q0[ia][ib] -= 0.5*amplitude*d_[ia][ib];
      }
    }

  }

  return;
}

/*****************************************************************************
 *
 *  colloids_fix_swd
 *
 *  The velocity gradient tensor used in the Beris-Edwards equations
 *  requires some approximation to the velocity at lattice sites
 *  inside particles. Here we set the lattice velocity based on
 *  the solid body rotation u = v + Omega x rb
 *
 *****************************************************************************/

int colloids_fix_swd(hydro_t * hydro) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];
  const int nextra = 1;

  double u[3];
  double rb[3];
  double x, y, z;

  colloid_t * p_c;
  colloid_t * colloid_at_site_index(int);

  assert(hydro);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    x = noffset[X] + ic;
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	if (site_map_get_status_index(index) != FLUID) {
	  u[X] = 0.0;
	  u[Y] = 0.0;
	  u[Z] = 0.0;
	  hydro_u_set(hydro, index, u);
	}

	p_c = colloid_at_site_index(index);

	if (p_c) {
	  /* Set the lattice velocity here to the solid body
	   * rotational velocity */

	  rb[X] = x - p_c->s.r[X];
	  rb[Y] = y - p_c->s.r[Y];
	  rb[Z] = z - p_c->s.r[Z];

	  cross_product(p_c->s.w, rb, u);

	  u[X] += p_c->s.v[X];
	  u[Y] += p_c->s.v[Y];
	  u[Z] += p_c->s.v[Z];

	  hydro_u_set(hydro, index, u);

	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_q_tensor_anchoring_set
 *
 *****************************************************************************/

void colloids_q_tensor_anchoring_set(const int type) {

  assert(type == ANCHORING_PLANAR || type == ANCHORING_NORMAL);

  anchoring_coll_ = type;

  return;
}

/*****************************************************************************
 *
 *  wall_anchoring_set
 *
 *****************************************************************************/

void wall_anchoring_set(const int type) {

  assert(type == ANCHORING_PLANAR || type == ANCHORING_NORMAL ||
	 type == ANCHORING_FIXED);

  anchoring_wall_ = type;

  return;
}

/*****************************************************************************
 *
 *  colloids_q_tensor_w
 *
 *****************************************************************************/

double colloids_q_tensor_w(void) {

  return w_surface_;
}

/*****************************************************************************
 *
 *  wall_w_get
 *
 *****************************************************************************/

double wall_w_get(void) {

  return w_surface_wall_;
}

/*****************************************************************************
 *
 *  colloids_q_tensor_w_set
 *
 *****************************************************************************/

void colloids_q_tensor_w_set(double w) {

  w_surface_ = w;
  return;
}

/*****************************************************************************
 *
 *  wall_w_set
 *
 *****************************************************************************/

void wall_w_set(double w) {

  w_surface_wall_ = w;
  return;
}
