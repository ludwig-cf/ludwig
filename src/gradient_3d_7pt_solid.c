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
 *  There is a correction related to the surface component of the
 *  molecular field
 *
 *        w (Q_ab - Q^s_ab)
 *
 *  and for cholesterics, one related to the pitch wavenumber q0.
 *  Anchoring is specified in colloids_Q_tensor.c at the moment.
 *
 *  A special treatment of edges and corners is required for colloids,
 *  where an iterative approach is used to get two or three orthogonal
 *  gradients at fluid points.
 *
 *  This will also cope with parallel boundaries separated by one fluid
 *  points, whatever the solid involved.
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
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "gradient.h"
#include "site_map.h"
#include "free_energy.h"
#include "colloids.h"
#include "colloids_Q_tensor.h"
#include "gradient_3d_7pt_solid.h"

/* Only tensor order parameter relevant */

#define NOP 5
#define NITERATION 40

static void gradient_bcs(double kappa0, double kappa1, const int dn[3],
			 double dq[NOP][3], double bc[NOP][NOP][3]);

static void gradient_general(const double * field, double * grad,
			     double * del2, const int nextra);
static int util_gaussian(double a[NOP][NOP], double xb[NOP]);
static void util_q5_to_qab(double q[3][3], const double * phi);

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
 *  gradient_3d_7pt_solid_operator
 *
 *  Compute the gradients in the fluid, and then correct for the presence
 *  of various solids.
 *
 *****************************************************************************/

void gradient_3d_7pt_solid_d2(const int nop, const double * field,
			      double * grad, double * delsq) {
  int nextra;

  assert(nop == NOP);
  assert(field);
  assert(grad);
  assert(delsq);

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);
  gradient_general(field, grad, delsq, nextra);

  return;
}

/*****************************************************************************
 *
 *  gradient_general
 *
 *  General routine to deal with solid in all configurations.
 *
 *****************************************************************************/

static void gradient_general(const double * field, double * grad,
			     double * del2, const int nextra) {
  int nlocal[3];
  int nhalo;
  int ic, jc, kc;
  int ia, ib, ig, ih;
  int index, n, ns, n1, n2;
  int niterate;

  int str[3];
  int mask[6];
  char status[6];

  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  const int normal[6] = {X, X, Y, Y, Z, Z};
  const int nsolid[6] = {0, 1, 0, 1, 0, 1};

  double gradn[NOP][3][2];                  /* Partial gradients */
  double q0[3][3];                          /* Prefered surface Q_ab */
  double qs[3][3];                          /* 'Surface' Q_ab */
  double a[NOP][NOP];                       /* Matrix for linear system */
  double b[NOP];                            /* RHS / unknown */
  double dq[NOP][3];                        /* normal/tangential gradients */
  double bc[NOP][NOP][3];                   /* Terms in boundary condition */
  double c[6][3][3];                        /* Constant terms in BC. */
  double dn[3];                             /* Unit normal. */

  double w;                                 /* Anchoring strength parameter */
  double q_0;                               /* Cholesteric pitch wavevector */
  double kappa0;                            /* Elastic constants */
  double kappa1;

  double blue_phase_q0(void);

  assert(NOP == 5);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  str[Z] = 1;
  str[Y] = str[Z]*(nlocal[Z] + 2*nhalo);
  str[X] = str[Y]*(nlocal[Y] + 2*nhalo);

  kappa0 = fe_kappa();
  kappa1 = fe_kappa(); /* One elastic constant */ 

  q_0 = blue_phase_q0();
  w = colloids_q_tensor_w();

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);
	if (site_map_get_status_index(index) != FLUID) continue;

	status[0] = site_map_get_status(ic+1, jc, kc);
	status[1] = site_map_get_status(ic-1, jc, kc);
	status[2] = site_map_get_status(ic, jc+1, kc);
	status[3] = site_map_get_status(ic, jc-1, kc);
	status[4] = site_map_get_status(ic, jc, kc+1);
	status[5] = site_map_get_status(ic, jc, kc-1);

	/* Set up partial gradients, and gradients */

	for (n1 = 0; n1 < NOP; n1++) {
	  for (ia = 0; ia < 3; ia++) {
	    gradn[n1][ia][0] =
	      field[NOP*(index + str[ia]) + n1] - field[NOP*index + n1];
	    gradn[n1][ia][1] =
	      field[NOP*index + n1] - field[NOP*(index - str[ia]) + n1];
	  }
	}

	for (n1 = 0; n1 < NOP; n1++) {
	  del2[NOP*index + n1] = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    grad[3*(NOP*index + n1) + ia] = 
	      0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	    del2[NOP*index + n1] += gradn[n1][ia][0] - gradn[n1][ia][1];
	  }
	}

	ns = 0;
	for (n = 0; n < 6; n++) {
	  mask[n] = (status[n] != FLUID);
	  ns += mask[n];
	}

	if (ns == 0) continue;

	/* Solid boundary condition corrections are required. */

	util_q5_to_qab(qs, field + NOP*index);

	for (n = 0; n < 6; n++) {
	  if (status[n] == FLUID) continue;

	  colloids_q_boundary_normal(index, bcs[n], dn);
	  colloids_q_boundary(dn, qs, q0, status[n]);

	  /* Check for wall/colloid */
	  if (status[n] == COLLOID) w = colloids_q_tensor_w();
	  if (status[n] == BOUNDARY) w = wall_w_get();
	  assert(status[n] == COLLOID || status[n] == BOUNDARY);

	  /* Compute c[n][a][b] */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      c[n][ia][ib] = 0.0;
	      for (ig = 0; ig < 3; ig++) {
		for (ih = 0; ih < 3; ih++) {
		  c[n][ia][ib] -= kappa1*q_0*bcs[n][ig]*
		    (e_[ia][ig][ih]*qs[ih][ib] + e_[ib][ig][ih]*qs[ih][ia]);
		}
	      }
	      c[n][ia][ib] -= w*(qs[ia][ib] - q0[ia][ib]);
	    }
	  }
	}

	/* Set up initial approximation to grad using partial gradients
	 * where solid sites are involved (or zero where none available) */

	for (n1 = 0; n1 < NOP; n1++) {
	  for (ia = 0; ia < 3; ia++) {
	    gradn[n1][ia][0] *= (1 - mask[2*ia]);
	    gradn[n1][ia][1] *= (1 - mask[2*ia + 1]);
	    grad[3*(NOP*index + n1) + ia] =
	      0.5*(1.0 + ((mask[2*ia] + mask[2*ia+1]) % 2))*
	      (gradn[n1][ia][0] + gradn[n1][ia][1]);
	  }
	}

	/* Iterate to a solution. */

	for (niterate = 0; niterate < NITERATION; niterate++) {

	  for (n = 0; n < 6; n++) {

	    if (status[n] == FLUID) continue;

	    for (n1 = 0; n1 < NOP; n1++) {
	      for (ia = 0; ia < 3; ia++) {
		dq[n1][ia] = grad[3*(NOP*index + n1) + ia];

	      }
	      dq[n1][normal[n]] = 1.0;
	    }

	    /* Construct boundary condition terms. */

	    gradient_bcs(kappa0, kappa1, bcs[n], dq, bc);

	    for (n1 = 0; n1 < NOP; n1++) {
	      b[n1] = 0.0;
	      for (n2 = 0; n2 < NOP; n2++) {
		a[n1][n2] = bc[n1][n2][normal[n]];
		b[n1] -= bc[n1][n2][normal[n]];
		for (ia = 0; ia < 3; ia++) {
		  b[n1] += bc[n1][n2][ia];
		}
	      }
	    }

	    b[XX] = -(b[XX] +     c[n][X][X]);
	    b[XY] = -(b[XY] + 2.0*c[n][X][Y]);
	    b[XZ] = -(b[XZ] + 2.0*c[n][X][Z]);
	    b[YY] = -(b[YY] +     c[n][Y][Y]);
	    b[YZ] = -(b[YZ] + 2.0*c[n][Y][Z]);

	    util_gaussian(a, b);

	    for (n1 = 0; n1 < NOP; n1++) {
	      gradn[n1][normal[n]][nsolid[n]] = b[n1];
	    }
	  }

	  /* Now recompute gradients */

	  for (n1 = 0; n1 < NOP; n1++) {
	    del2[NOP*index + n1] = 0.0;
	    for (ia = 0; ia < 3; ia++) {
	      grad[3*(NOP*index + n1) + ia] =
		0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	      del2[NOP*index + n1] += gradn[n1][ia][0] - gradn[n1][ia][1];
	    }
	  }

	  /* No iteration required if only one boundary. */
	  if (ns < 2) break;
	}

	/* Next site. */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_bcs
 *
 *****************************************************************************/

static void gradient_bcs(double kappa0, double kappa1, const int dn[3],
			 double dq[NOP][3], double bc[NOP][NOP][3]) {

  double kappa2;

  kappa2 = kappa0 + kappa1;

  /* XX equation */

  bc[XX][XX][X] =  kappa0*dn[X]*dq[XX][X];
  bc[XX][XY][X] = -kappa1*dn[Y]*dq[XY][X];
  bc[XX][XZ][X] = -kappa1*dn[Z]*dq[XZ][X];
  bc[XX][YY][X] =  0.0;
  bc[XX][YZ][X] =  0.0;

  bc[XX][XX][Y] = kappa1*dn[Y]*dq[XX][Y];
  bc[XX][XY][Y] = kappa0*dn[X]*dq[XY][Y];
  bc[XX][XZ][Y] = 0.0;
  bc[XX][YY][Y] = 0.0;
  bc[XX][YZ][Y] = 0.0;

  bc[XX][XX][Z] = kappa1*dn[Z]*dq[XX][Z];
  bc[XX][XY][Z] = 0.0;
  bc[XX][XZ][Z] = kappa0*dn[X]*dq[XZ][Z];
  bc[XX][YY][Z] = 0.0;
  bc[XX][YZ][Z] = 0.0;

  /* XY equation */

  bc[XY][XX][X] =  kappa0*dn[Y]*dq[XX][X];
  bc[XY][XY][X] =  kappa2*dn[X]*dq[XY][X];
  bc[XY][XZ][X] =  0.0;
  bc[XY][YY][X] = -kappa1*dn[Y]*dq[YY][X];
  bc[XY][YZ][X] = -kappa1*dn[Z]*dq[YZ][X];

  bc[XY][XX][Y] = -kappa1*dn[X]*dq[XX][Y];
  bc[XY][XY][Y] =  kappa2*dn[Y]*dq[XY][Y];
  bc[XY][XZ][Y] = -kappa1*dn[Z]*dq[XZ][Y];
  bc[XY][YY][Y] =  kappa0*dn[X]*dq[YY][Y];
  bc[XY][YZ][Y] =  0.0;

  bc[XY][XX][Z] = 0.0;
  bc[XY][XY][Z] = 2.0*kappa1*dn[Z]*dq[XY][Z];
  bc[XY][XZ][Z] = kappa0*dn[Y]*dq[XZ][Z];
  bc[XY][YY][Z] = 0.0;
  bc[XY][YZ][Z] = kappa0*dn[X]*dq[YZ][Z];

  /* XZ equation */

  bc[XZ][XX][X] =  kappa2*dn[Z]*dq[XX][X];
  bc[XZ][XY][X] =  0.0;
  bc[XZ][XZ][X] =  kappa2*dn[X]*dq[XZ][X];
  bc[XZ][YY][X] =  kappa1*dn[Z]*dq[YY][X];
  bc[XZ][YZ][X] = -kappa1*dn[Y]*dq[YZ][X];

  bc[XZ][XX][Y] = 0.0;
  bc[XZ][XY][Y] = kappa0*dn[Z]*dq[XY][Y];
  bc[XZ][XZ][Y] = 2.0*kappa1*dn[Y]*dq[XZ][Y];
  bc[XZ][YY][Y] = 0.0;
  bc[XZ][YZ][Y] = kappa0*dn[X]*dq[YZ][Y];

  bc[XZ][XX][Z] = -kappa2*dn[X]*dq[XX][Z];
  bc[XZ][XY][Z] = -kappa1*dn[Y]*dq[XY][Z];
  bc[XZ][XZ][Z] =  kappa2*dn[Z]*dq[XZ][Z];
  bc[XZ][YY][Z] = -kappa0*dn[X]*dq[YY][Z];
  bc[XZ][YZ][Z] =  0.0;

  /* YY equation */

  bc[YY][XX][X] = 0.0;
  bc[YY][XY][X] = kappa0*dn[Y]*dq[XY][X];
  bc[YY][XZ][X] = 0.0;
  bc[YY][YY][X] = kappa1*dn[X]*dq[YY][X];
  bc[YY][YZ][X] = 0.0;

  bc[YY][XX][Y] =  0.0;
  bc[YY][XY][Y] = -kappa1*dn[X]*dq[XY][Y];
  bc[YY][XZ][Y] =  0.0;
  bc[YY][YY][Y] =  kappa0*dn[Y]*dq[YY][Y];
  bc[YY][YZ][Y] = -kappa1*dn[Z]*dq[YZ][Y];

  bc[YY][XX][Z] = 0.0;
  bc[YY][XY][Z] = 0.0;
  bc[YY][XZ][Z] = 0.0;
  bc[YY][YY][Z] = kappa1*dn[Z]*dq[YY][Z];
  bc[YY][YZ][Z] = kappa0*dn[Y]*dq[YZ][Z];

  /* YZ equation */

  bc[YZ][XX][X] = 0.0;
  bc[YZ][XY][X] = kappa0*dn[Z]*dq[XY][X];
  bc[YZ][XZ][X] = kappa0*dn[Y]*dq[XZ][X];
  bc[YZ][YY][X] = 0.0;
  bc[YZ][YZ][X] = 2.0*kappa1*dn[X]*dq[YZ][X];

  bc[YZ][XX][Y] =  kappa1*dn[Z]*dq[XX][Y];
  bc[YZ][XY][Y] =  0.0;
  bc[YZ][XZ][Y] = -kappa1*dn[X]*dq[XZ][Y];
  bc[YZ][YY][Y] =  kappa2*dn[Z]*dq[YY][Y];
  bc[YZ][YZ][Y] =  kappa2*dn[Y]*dq[YZ][Y];

  bc[YZ][XX][Z] = -kappa0*dn[Y]*dq[XX][Z];
  bc[YZ][XY][Z] = -kappa1*dn[X]*dq[XY][Z];
  bc[YZ][XZ][Z] =  0.0;
  bc[YZ][YY][Z] = -kappa2*dn[Y]*dq[YY][Z];
  bc[YZ][YZ][Z] =  kappa2*dn[Z]*dq[YZ][Z];

  return;
}

/*****************************************************************************
 *
 *  util_gaussian
 *
 *  Solve linear system via Gaussian elimination. For the problems in this
 *  file, we only need to exchange rows, ie., have a partial pivot.
 *
 *  We solve Ax = b for A[NOP][NOP].
 *  xb is RHS on entry, and solution on exit.
 *  A is destroyed.
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

static int util_gaussian(double a[NOP][NOP], double xb[NOP]) {

  int i, j, k;
  int ifail = 0;
  int iprow;
  int ipivot[NOP];

  double tmp;

  iprow = -1;
  for (k = 0; k < NOP; k++) {
    ipivot[k] = -1;
  }

  for (k = 0; k < NOP; k++) {

    /* Find pivot row */
    tmp = 0.0;
    for (i = 0; i < NOP; i++) {
      if (ipivot[i] == -1) {
	if (fabs(a[i][k]) >= tmp) {
	  tmp = fabs(a[i][k]);
	  iprow = i;
	}
      }
    }
    ipivot[k] = iprow;

    /* divide pivot row by the pivot element a[iprow][k] */

    if (a[iprow][k] == 0.0) {
      fatal("Gaussian elimination failed in gradient calculation\n");
    }

    tmp = 1.0 / a[iprow][k];
    for (j = k; j < NOP; j++) {
      a[iprow][j] *= tmp;
    }
    xb[iprow] *= tmp;

    /* Subtract the pivot row (scaled) from remaining rows */

    for (i = 0; i < NOP; i++) {
      if (ipivot[i] == -1) {
	tmp = a[i][k];
	for (j = k; j < NOP; j++) {
	  a[i][j] -= tmp*a[iprow][j];
	}
	xb[i] -= tmp*xb[iprow];
      }
    }
  }

  /* Now do the back substitution */

  for (i = NOP - 1; i > -1; i--) {
    iprow = ipivot[i];
    tmp = xb[iprow];
    for (k = i + 1; k < NOP; k++) {
      tmp -= a[iprow][k]*xb[ipivot[k]];
    }
    xb[iprow] = tmp;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_q5_to_qab
 *
 *  A utility to expand condensed tensor.
 *
 *****************************************************************************/

static void util_q5_to_qab(double q[3][3], const double * phi) {

  q[X][X] = phi[0];
  q[X][Y] = phi[1];
  q[X][Z] = phi[2];
  q[Y][X] = phi[1];
  q[Y][Y] = phi[3];
  q[Y][Z] = phi[4];
  q[Z][X] = phi[2];
  q[Z][Y] = phi[4];
  q[Z][Z] = -phi[0] - phi[3];

  return;
}
