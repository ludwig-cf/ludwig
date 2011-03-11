/*****************************************************************************
 *
 *  gradient_3d_7pt_solid.c
 *
 *  Gradient operations for 3D seven point stencil.
 *
 *                        (ic, jc+1, kc)
 *         (ic-1, jc, kc) (ic, jc  , kc) (ic+1, jc, kc)
 *                        (ic, jc-1, kc)
 *
 *  ...and so in z-direction
 *
 *  d_x Q = [Q(ic+1,jc,kc) - Q(ic-1,jc,kc)] / 2
 *  d_y Q = [Q(ic,jc+1,kc) - Q(ic,jc-1,kc)] / 2
 *  d_z Q = [Q(ic,jc,kc+1) - Q(ic,jc,kc-1)] / 2
 *
 *  nabla^2 Q = Q(ic+1,jc,kc) + Q(ic-1,jc,kc)
 *            + Q(ic,jc+1,kc) + Q(ic,jc-1,kc)
 *            + Q(ic,jc,kc+1) + Q(ic,jc,kc-1) - 6 Q(ic,jc,kc)
 *
 *  The cholesteric anchoring boundary condition specifies the surface
 *  free energy
 *
 *  f_s = w (Q_ab - Q^s_ab)^2
 *
 *  Taking the functional derivative, and equating to kappa | grad Q_ab .n|
 *  we get
 *
 *    grad Q_ab ~ (wL/kappa)*(Q_ab - Q^s_ab) at solid fluid surface
 *
 *  This is a test routine for tensor order parameter with anchoring
 *  Q^s specified in colloids_Q_tensor.c at the moment. We take the
 *  length scale L = 1, the grid scale.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "gradient.h"
#include "site_map.h"
#include "gradient_3d_7pt_solid.h"


static void gradient_3d_7pt_fluid_operator(const int nop, const double * field,
					   double * grad, double * delsq,
					   const int nextra);

static const int cv_[6][3] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0},
			      {0,0,1}, {0,0,-1}};
static const double wlrk_ = 1.0;

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_init
 *
 *****************************************************************************/

void gradient_3d_7pt_solid_init(void) {

  gradient_d2_set(gradient_3d_7pt_solid_d2);

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_d2
 *
 *****************************************************************************/

void gradient_3d_7pt_solid_d2(const int nop, const double * field,
			      double * grad, double * delsq) {
  int nextra;

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);
  assert(nop == 5); /* Cholesterics only for now */

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_3d_7pt_fluid_operator(nop, field, grad, delsq, nextra);

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_operator
 *
 *****************************************************************************/

static void gradient_3d_7pt_fluid_operator(const int nop,
					   const double * field,
					   double * grad,
					   double * del2,
					   const int nextra) {
  int nlocal[3];
  int nhalo;
  int n;
  int ic, jc, kc;
  int index, index1;

  double dx1[5], dx2[5], dy1[5], dy2[5], dz1[5], dz2[5], qs[5];
  extern void colloids_q_boundary(int, const int di[3], double qs[5]);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);

	if (site_map_get_status_index(index) != FLUID) continue;

	index1 = coords_index(ic+1, jc, kc);
	if (site_map_get_status_index(index1) == COLLOID) {
	    colloids_q_boundary(index, cv_[0], qs);
	    for (n = 0; n < nop; n++) {
		dx1[n] = -wlrk_*(field[nop*index + n] - qs[n]);
	    }
	}
	else {
	    for (n = 0; n < nop; n++) {
		dx1[n] = field[nop*index1 + n] - field[nop*index + n];
	    }
	}

	index1 = coords_index(ic-1, jc, kc);
	if (site_map_get_status_index(index1) == COLLOID) {
	    colloids_q_boundary(index, cv_[1], qs);
	    for (n = 0; n < nop; n++) {
		dx2[n] = -wlrk_*(field[nop*index + n] - qs[n]);
	    }
	}
	else {
	    for (n = 0; n < nop; n++) {
		dx2[n] = -(field[nop*index + n] - field[nop*index1 + n]);
	    }
	}

	index1 = coords_index(ic, jc+1, kc);
	if (site_map_get_status_index(index1) == COLLOID) {
	    colloids_q_boundary(index, cv_[2], qs);

	    for (n = 0; n < nop; n++) {
		dy1[n] = -wlrk_*(field[nop*index + n] - qs[n]);
	    }
	}
	else {
	    for (n = 0; n < nop; n++) {
		dy1[n] = field[nop*index1 + n] - field[nop*index + n];
	    }
	}

	index1 = coords_index(ic, jc-1, kc);
	if (site_map_get_status_index(index1) == COLLOID) {
	    colloids_q_boundary(index, cv_[3], qs);
	    for (n = 0; n < nop; n++) {
		dy2[n] = -wlrk_*(field[nop*index + n] - qs[n]);
	    }
	}
	else {
	    for (n = 0; n < nop; n++) {
		dy2[n] = -(field[nop*index + n] - field[nop*index1 + n]);
	    }
	}

	index1 = coords_index(ic, jc, kc+1);
	if (site_map_get_status_index(index1) == COLLOID) {
	    colloids_q_boundary(index, cv_[4], qs);
	    for (n = 0; n < nop; n++) {
		dz1[n] = -wlrk_*(field[nop*index + n] - qs[n]);
	    }
	}
	else {
	    for (n = 0; n < nop; n++) {
		dz1[n] = field[nop*index1 + n] - field[nop*index + n];
	    }
	}

	index1 = coords_index(ic, jc, kc-1);
	if (site_map_get_status_index(index1) == COLLOID) {
	    colloids_q_boundary(index, cv_[5], qs);
	    for (n = 0; n < nop; n++) {
		dz2[n] = -wlrk_*(field[nop*index + n] - qs[n]);
	    }
	}
	else {
	    for (n = 0; n < nop; n++) {
		dz2[n] = -(field[nop*index + n] - field[nop*index1 + n]);
	    }
	}

	/* Use these values in the stencil */

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + X] = 0.5*(dx1[n] - dx2[n]);
	  grad[3*(nop*index + n) + Y] = 0.5*(dy1[n] - dy2[n]);
	  grad[3*(nop*index + n) + Z] = 0.5*(dz1[n] - dz2[n]);
	  del2[nop*index + n]
	    = dx1[n] + dx2[n] + dy1[n] + dy2[n] + dz1[n] + dz2[n];
	}
      }
    }
  }

  return;
}
