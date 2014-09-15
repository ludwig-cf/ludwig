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
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "gradient.h"
#include "site_map.h"
#include "free_energy.h"
#include "blue_phase.h"
#include "colloids.h"
#include "colloids_Q_tensor.h"
#include "gradient_3d_7pt_solid.h"

/* Only tensor order parameter relevant */

#define NOP 5
#define NMAX_ITERATION 400

static void gradient_bcs6(double kappa0, double kappa1, const int dn[3],
			  double dq[6][3], double bc[6][6][3]);
static void gradient_general6(const double * field, double * grad,
			     double * del2, const int nextra);

static void gradient_general6x5(const double * field, double * grad,
				double * del2, const int nextra);
static void gradient_bcs6x5(double kappa0, double kappa1, const int dn[3],
			    double dq[NOP][3], double bc[6][NOP][3]);

static int gradient_no_iteration(const double * field, double * grad,
				 double * del2, const int nextra);

static void gradient_bcs6x5_block(double kappa0, double kappa1,
				  const int dn[3],
				  double dq[NOP], double bc[6][NOP],
				  int id);

static int util_gaussian6(double a[6][6], double xb[6]);
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
  int method = 1;

  assert(nop == NOP);
  assert(field);
  assert(grad);
  assert(delsq);

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);
  if (method == 1) gradient_general6(field, grad, delsq, nextra);
  if (method == 2) gradient_general6x5(field, grad, delsq, nextra);
  if (method == 3) gradient_no_iteration(field, grad, delsq, nextra);

  return;
}

/*****************************************************************************
 *
 *  gradient_general6
 *
 *  Solve 6x6 system of equations for the boundary condition
 *  and post-hoc set the trace of the gradient Q to be zero.
 *
 *  This works for up to 4 solid neighbours.
 *
 *****************************************************************************/

static void gradient_general6(const double * field, double * grad,
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

  double gradn[6][3][2];                    /* Partial gradients */
  double q0[3][3];                          /* Prefered surface Q_ab */
  double qs[3][3];                          /* 'Surface' Q_ab */
  double a[6][6];                           /* Matrix for linear system */
  double b[6];                            /* RHS / unknown */
  double dq[6][3];                        /* normal/tangential gradients */
  double bc[6][6][3];                   /* Terms in boundary condition */
  double c[6][3][3];                        /* Constant terms in BC. */
  double dn[3];                             /* Unit normal. */
  double tmp;

  double w1_coll;                           /* Anchoring strength parameter */
  double w2_coll;                           /* Second anchoring parameter */
  double w1_wall;
  double w2_wall;
  double w1=0.0;
  double w2=0.0;
  double q_0;                               /* Cholesteric pitch wavevector */
  double kappa0;                            /* Elastic constants */
  double kappa1;

  double amplitude;                         /* Scalar order parameter */
  double qtilde[3][3];                      /* For planar anchoring */
  double q2;                                /* Contraction Q_ab Q_ab */
  double trace;

  double del2m1[NOP];
  double del2diff;

  assert(NOP == 5);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  str[Z] = 1;
  str[Y] = str[Z]*(nlocal[Z] + 2*nhalo);
  str[X] = str[Y]*(nlocal[Y] + 2*nhalo);

  kappa0 = fe_kappa();
  kappa1 = fe_kappa(); /* One elastic constant */ 

  q_0 = blue_phase_q0();
  blue_phase_coll_w12(&w1_coll, &w2_coll);
  blue_phase_wall_w12(&w1_wall, &w2_wall);
  amplitude = blue_phase_amplitude_compute(); 

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

	for (ia = 0; ia < 3; ia++) {
	  gradn[ZZ][ia][0] = -gradn[XX][ia][0] - gradn[YY][ia][0];
	  gradn[ZZ][ia][1] = -gradn[XX][ia][1] - gradn[YY][ia][1];
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

	/* For planar anchoring we require qtilde_ab of Fournier and
	 * Galatola, and its square */

	q2 = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    qtilde[ia][ib] = qs[ia][ib] + 0.5*amplitude*d_[ia][ib];
	    q2 += qtilde[ia][ib]*qtilde[ia][ib]; 
	  }
	}

        /* Look at the neighbours */

	for (n = 0; n < 6; n++) {
	  if (status[n] == FLUID) continue;

	  colloids_q_boundary_normal(index, bcs[n], dn);
	  colloids_q_boundary(dn, qs, q0, status[n]);

	  /* Check for wall/colloid */
	  if (status[n] == COLLOID) {
	    w1 = w1_coll;
	    w2 = w2_coll;
	  }

	  if (status[n] == BOUNDARY) {
	    w1 = w1_wall;
	    w2 = w2_wall;
	  }
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
	      /* Normal anchoring: w2 must be zero and q0 is preferred Q
	       * Planar anchoring: in w1 term q0 is effectively
	       *                   (Qtilde^perp - 0.5S_0) while in w2 we
	       *                   have Qtilde appearing explicitly.
               *                   See colloids_q_boundary() etc */
	      c[n][ia][ib] +=
		-w1*(qs[ia][ib] - q0[ia][ib])
		-w2*(2.0*q2 - 4.5*amplitude*amplitude)*qtilde[ia][ib];
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

	for (ia = 0; ia < 3; ia++) {
	  gradn[ZZ][ia][0] = - gradn[XX][ia][0] - gradn[YY][ia][0];
	  gradn[ZZ][ia][1] = - gradn[XX][ia][1] - gradn[YY][ia][1];
	}

	/* Iterate to a solution. */

	for (niterate = 0; niterate < NMAX_ITERATION; niterate++) {

	  for (n1 = 0; n1 < NOP; n1++) {
	    del2m1[n1] = del2[NOP*index + n1];
	  }

	  for (n = 0; n < 6; n++) {

	    if (status[n] == FLUID) continue;

	    for (n1 = 0; n1 < NOP; n1++) {
	      for (ia = 0; ia < 3; ia++) {
		dq[n1][ia] = grad[3*(NOP*index + n1) + ia];

	      }
	      dq[n1][normal[n]] = 1.0;
	    }

	    for (ia = 0; ia < 3; ia++) {
	      dq[ZZ][ia] = - dq[XX][ia] - dq[YY][ia];
	    }
	    dq[ZZ][normal[n]] = 1.0;

	    /* Construct boundary condition linear algebra problem.
	     * Estimated terms are moved to the right-hand side b[],
	     * while the coefficients of the unkonwn terms are set
	     * up in matrix a[][]. Finally, the constant terms are
	     * also added to the right-hand side (with a factor of
	     * two in the off-diagaonal parts). */

	    gradient_bcs6(kappa0, kappa1, bcs[n], dq, bc);

	    for (n1 = 0; n1 < 6; n1++) {
	      b[n1] = 0.0;
	      for (n2 = 0; n2 < 6; n2++) {
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
	    b[ZZ] = -(b[ZZ] +     c[n][Z][Z]);

	    util_gaussian6(a, b);

	    /* Fix the resulting trace */

	    trace = r3_*(b[XX] + b[YY] + b[ZZ]);
	    b[XX] -= trace;
	    b[YY] -= trace;
	    b[ZZ] -= trace;

	    for (n1 = 0; n1 < 6; n1++) {
	      gradn[n1][normal[n]][nsolid[n]] = b[n1];
	    }
	  }

	  /* Do not update gradients if solid neighbours in both directions */

	  for (ia = 0; ia < 3; ia++) {
	    tmp = 1.0*(1 - (mask[2*ia] && mask[2*ia+1]));
	    for (n1 = 0; n1 < 6; n1++) {
	      gradn[n1][ia][0] *= tmp;
	      gradn[n1][ia][1] *= tmp;
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

	  del2diff = 0.0;
	  for (n1 = 0; n1 < NOP; n1++) {
	    del2diff += fabs(del2[NOP*index + n1] - del2m1[n1]);
	  }

	  if (del2diff < FLT_EPSILON) break;
	}

	/* Next site. */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_general6x5
 *
 *  Here's the 6x5 version solved via SVD. It gives the same answer
 *  as the above when the boundary normals are along the coordinate
 *  direction. It may be more robust if ever move away from this
 *  situation, so retained here.
 *
 *****************************************************************************/

static void gradient_general6x5(const double * field, double * grad,
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
  double ** a;                         /* Matrix for linear system */
  double b[6];                              /* RHS */
  double x[NOP];                            /* Solution */
  double dq[NOP][3];                        /* normal/tangential gradients */
  double bc[6][NOP][3];                   /* Terms in boundary condition */
  double c[6][3][3];                        /* Constant terms in BC. */
  double dn[3];                             /* Unit normal. */
  double tmp;

  double w1_coll;                           /* Anchoring strength parameter */
  double w2_coll;                           /* Second anchoring parameter */
  double w1_wall;
  double w2_wall;
  double w1;
  double w2;
  double q_0;                               /* Cholesteric pitch wavevector */
  double kappa0;                            /* Elastic constants */
  double kappa1;

  double amplitude;                         /* Scalar order parameter */
  double qtilde[3][3];                      /* For planar anchoring */
  double q2;                                /* Contraction Q_ab Q_ab */

  assert(NOP == 5);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  util_matrix_create(6, NOP, &a);

  str[Z] = 1;
  str[Y] = str[Z]*(nlocal[Z] + 2*nhalo);
  str[X] = str[Y]*(nlocal[Y] + 2*nhalo);

  kappa0 = fe_kappa();
  kappa1 = fe_kappa(); /* One elastic constant */ 

  q_0 = blue_phase_q0();
  blue_phase_coll_w12(&w1_coll, &w2_coll);
  blue_phase_wall_w12(&w1_wall, &w2_wall);
  amplitude = blue_phase_amplitude_compute(); 

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

	/* For planar anchoring we require qtilde_ab of Fournier and
	 * Galatola, and its square */

	q2 = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    qtilde[ia][ib] = qs[ia][ib] + 0.5*amplitude*d_[ia][ib];
	    q2 += qtilde[ia][ib]*qtilde[ia][ib]; 
	  }
	}

        /* Look at the neighbours */

	for (n = 0; n < 6; n++) {
	  if (status[n] == FLUID) continue;

	  colloids_q_boundary_normal(index, bcs[n], dn);
	  colloids_q_boundary(dn, qs, q0, status[n]);

	  /* Check for wall/colloid */
	  if (status[n] == COLLOID) {
	    w1 = w1_coll;
	    w2 = w2_coll;
	  }

	  if (status[n] == BOUNDARY) {
	    w1 = w1_wall;
	    w2 = w2_wall;
	  }
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
	      /* Normal anchoring: w2 must be zero and q0 is preferred Q
	       * Planar anchoring: in w1 term q0 is effectively
	       *                   (Qtilde^perp - 0.5S_0) while in w2 we
	       *                   have Qtilde appearing explicitly.
               *                   See colloids_q_boundary() etc */
	      c[n][ia][ib] +=
		-w1*(qs[ia][ib] - q0[ia][ib])
		-w2*(2.0*q2 - 4.5*amplitude*amplitude)*qtilde[ia][ib];
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

	for (niterate = 0; niterate < NMAX_ITERATION; niterate++) {

	  for (n = 0; n < 6; n++) {

	    if (status[n] == FLUID) continue;

	    for (n1 = 0; n1 < NOP; n1++) {
	      for (ia = 0; ia < 3; ia++) {
		dq[n1][ia] = grad[3*(NOP*index + n1) + ia];

	      }
	      dq[n1][normal[n]] = 1.0;
	    }

	    /* Construct boundary condition linear algebra problem.
	     * Estimated terms are moved to the right-hand side b[],
	     * while the coefficients of the unkonwn terms are set
	     * up in matrix a[][]. Finally, the constant terms are
	     * also added to the right-hand side (with a factor of
	     * two in the off-diagaonal parts). */

	    gradient_bcs6x5(kappa0, kappa1, bcs[n], dq, bc);

	    for (n1 = 0; n1 < 6; n1++) {
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
	    b[ZZ] = -(b[ZZ] +     c[n][Z][Z]);

	    {
	      int ifail;
	      ifail = util_svd_solve(6, NOP, a, b, x);
	      if (ifail != 0) fatal("SVD failed\n");
	    }

	    for (n1 = 0; n1 < NOP; n1++) {
	      gradn[n1][normal[n]][nsolid[n]] = x[n1];
	    }
	  }

	  /* Do not update gradients if solid neighbours in both directions */

	  for (ia = 0; ia < 3; ia++) {
	    tmp = 1.0*(1 - (mask[2*ia] && mask[2*ia+1]));
	    for (n1 = 0; n1 < NOP; n1++) {
	      gradn[n1][ia][0] *= tmp;
	      gradn[n1][ia][1] *= tmp;
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

  util_matrix_free(6, &a);

  return;
}

/*****************************************************************************
 *
 *  Non-iterative version using SVD
 *
 *****************************************************************************/

static int gradient_no_iteration(const double * field, double * grad,
				 double * del2, const int nextra) {
  int nlocal[3];
  int nhalo;
  int ic, jc, kc;
  int ia, ib, ig, ih;
  int index, n, n1, n2;
  int ifail;

  int str[3];
  char status[6];

  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  int normal[6];
  int nunknown;

  double gradn[NOP][3][2];                  /* Partial gradients */
  double q0[3][3];                          /* Prefered surface Q_ab */
  double qs[3][3];                          /* 'Surface' Q_ab */
  double ** a;                              /* Matrix for linear system */
  double ** a2;
  double ** a3;
  double b[6];                              /* RHS */
  double x[NOP];                            /* Solution */
  double dq[NOP][3];                        /* normal/tangential gradients */
  double bc[6][NOP][3];                   /* Terms in boundary condition */
  double c[6][3][3];                        /* Constant terms in BC. */
  double dn[3];                             /* Unit normal. */

  double w1_coll;                           /* Anchoring strength parameter */
  double w2_coll;                           /* Second anchoring parameter */
  double w1_wall;
  double w2_wall;
  double w1;
  double w2;
  double q_0;                               /* Cholesteric pitch wavevector */
  double kappa0;                            /* Elastic constants */
  double kappa1;

  double amplitude;                         /* Scalar order parameter */
  double qtilde[3][3];                      /* For planar anchoring */
  double q2;                                /* Contraction Q_ab Q_ab */

  assert(NOP == 5);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  util_matrix_create(6, NOP, &a);
  util_matrix_create(2*6, 2*NOP, &a2);
  util_matrix_create(3*6, 3*NOP, &a3);

  str[Z] = 1;
  str[Y] = str[Z]*(nlocal[Z] + 2*nhalo);
  str[X] = str[Y]*(nlocal[Y] + 2*nhalo);

  kappa0 = fe_kappa();
  kappa1 = fe_kappa(); /* One elastic constant */ 

  q_0 = blue_phase_q0();
  blue_phase_coll_w12(&w1_coll, &w2_coll);
  blue_phase_wall_w12(&w1_wall, &w2_wall);
  amplitude = blue_phase_amplitude_compute(); 

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);
	if (site_map_get_status_index(index) != FLUID) continue;

	/* Set up partial gradients */

	for (n1 = 0; n1 < NOP; n1++) {
	  for (ia = 0; ia < 3; ia++) {
	    gradn[n1][ia][0] =
	      field[NOP*(index + str[ia]) + n1] - field[NOP*index + n1];
	    gradn[n1][ia][1] =
	      field[NOP*index + n1] - field[NOP*(index - str[ia]) + n1];
	    dq[n1][ia] = 0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	  }
	}

	/* Identify solid neighbours. If neighbours in both +/- given
	 * coordinate direction are solid, set gradient to zero and
	 * treat as known. */

	n = 0;
	status[n] = site_map_get_status(ic+1, jc, kc);
	if (status[n] != FLUID) {
	  normal[n] = 0; /* +X */
	  n += 1;
	};
	status[n] = site_map_get_status(ic-1, jc, kc);
	if (status[n] != FLUID) {
	  normal[n] = 1; /* -X */
	  n += 1;
	}

	if (n == 2) {
	  for (n1 = 0; n1 < NOP; n1++) {
	    gradn[n1][X][0] = 0.0;
	    gradn[n1][X][1] = 0.0;
	  }
	  n -= 2;
	}
	nunknown = n;

	status[n] = site_map_get_status(ic, jc+1, kc);
	if (status[n] != FLUID) {
	  normal[n] = 2; /* +Y */
	  n += 1;
	}
	status[n] = site_map_get_status(ic, jc-1, kc);
	if (status[n] != FLUID) {
	  normal[n] = 3; /* -Y */
	  n += 1;
	}

	if (n - nunknown == 2) {
	  for (n1 = 0; n1 < NOP; n1++) {
	    gradn[n1][Y][0] = 0.0;
	    gradn[n1][Y][1] = 0.0;
	  }
	  n -= 2;
	}
	nunknown = n;

	status[n] = site_map_get_status(ic, jc, kc+1);
	if (status[n] != FLUID) {
	  normal[n] = 4; /* +Z */
	  n += 1;
	}
	status[n] = site_map_get_status(ic, jc, kc-1);
	if (status[n] != FLUID) {
	  normal[n] = 5; /* -Z */
	  n += 1;
	}

	if (n - nunknown == 2) {
	  for (n1 = 0; n1 < NOP; n1++) {
	    gradn[n1][Z][0] = 0.0;
	    gradn[n1][Z][1] = 0.0;
	  }
	  n -= 2;
	}
	nunknown = n;

	if (nunknown == 0) {
	  /* No boundaries, so fall through to compute fainal answer */
	}
	else {

	  /* For planar anchoring we require qtilde_ab of Fournier and
	   * Galatola, and its square */

	  util_q5_to_qab(qs, field + NOP*index);

	  q2 = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      qtilde[ia][ib] = qs[ia][ib] + 0.5*amplitude*d_[ia][ib];
	      q2 += qtilde[ia][ib]*qtilde[ia][ib]; 
	    }
	  }

	  /* For each solid boundary, set up the boundary condition terms */

	  for (n = 0; n < nunknown; n++) {

	    colloids_q_boundary_normal(index, bcs[normal[n]], dn);
	    colloids_q_boundary(dn, qs, q0, status[n]);

	    /* Check for wall/colloid */
	    if (status[n] == COLLOID) {
	      w1 = w1_coll;
	      w2 = w2_coll;
	    }

	    if (status[n] == BOUNDARY) {
	      w1 = w1_wall;
	      w2 = w2_wall;
	    }
	    assert(status[n] == COLLOID || status[n] == BOUNDARY);

	    /* Compute c[n][a][b] */

	    for (ia = 0; ia < 3; ia++) {
	      for (ib = 0; ib < 3; ib++) {
		c[n][ia][ib] = 0.0;
		for (ig = 0; ig < 3; ig++) {
		  for (ih = 0; ih < 3; ih++) {
		    c[n][ia][ib] -= kappa1*q_0*bcs[normal[n]][ig]*
		      (e_[ia][ig][ih]*qs[ih][ib] + e_[ib][ig][ih]*qs[ih][ia]);
		  }
		}
		/* Normal anchoring: w2 must be zero and q0 is preferred Q
		 * Planar anchoring: in w1 term q0 is effectively
		 *                   (Qtilde^perp - 0.5S_0) while in w2 we
		 *                   have Qtilde appearing explicitly.
		 *                   See colloids_q_boundary() etc */
		c[n][ia][ib] +=
		  -w1*(qs[ia][ib] - q0[ia][ib])
		  -w2*(2.0*q2 - 4.5*amplitude*amplitude)*qtilde[ia][ib];
	      }
	    }

	    /* Set unknown gradients to unity */

	    for (n1 = 0; n1 < NOP; n1++) {
	      gradn[n1][normal[n]/2][normal[n]%2] = 0.0;
	      dq[n1][normal[n]/2] = 1.0;
	    }

	  }

	  /* Now set up the system */

	  if (nunknown == 1) {

	    n = 0;

	    /* Solve the system and assign unknown partial gradients */

	    gradient_bcs6x5(kappa0, kappa1, bcs[normal[n]], dq, bc);

            for (n1 = 0; n1 < 6; n1++) {
              b[n1] = 0.0;
              for (n2 = 0; n2 < NOP; n2++) {
                a[n1][n2] = bc[n1][n2][normal[n]/2];
                b[n1] -= bc[n1][n2][normal[n]/2];
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
            b[ZZ] = -(b[ZZ] +     c[n][Z][Z]);

	    ifail = util_svd_solve(6, NOP, a, b, x);

	    for (n1 = 0; n1 < NOP; n1++) {
	      gradn[n1][normal[n]/2][normal[n] % 2] = x[n1];
	    }

	  }
	  else if (nunknown == 2) {

	    int idb, na, nb;
	    double dq1[NOP];
	    double bc1[6][NOP];
	    double b2[2*6];
	    double x2[2*NOP];
	    double f;

	    if (normal[0]/2 == X && normal[1]/2 == Y) normal[2] = Z;
	    if (normal[0]/2 == X && normal[1]/2 == Z) normal[2] = Y;
	    if (normal[0]/2 == Y && normal[1]/2 == Z) normal[2] = X;

	    for (n1 = 0; n1 < 2*6; n1++) {
	      b2[n1] = 0.0;
	    }

	    for (ia = 0; ia < nunknown; ia++) {

	      na = normal[ia];    /* normal for unknown ia */

	      for (ib = 0; ib < nunknown; ib++) {

		nb = normal[ib];
		idb = normal[ib]/2; /* coordinate direction of normal */

		/* Compute 6x5 block (ia,ib) in system matrix */

		for (n2 = 0; n2 < NOP; n2++) {
		  dq1[n2] = 1.0;
		}
		gradient_bcs6x5_block(kappa0, kappa1, bcs[na], dq1, bc1, idb);

		f = 0.5*(1.0 + d_[ia][ib]);
		for (n1 = 0; n1 < 6; n1++) {
		  for (n2 = 0; n2 < NOP; n2++) {
		    a2[6*ia + n1][NOP*ib + n2] = f*bc1[n1][n2];
		  }
		}

		/* Add known contributions to right hand side */

		for (n2 = 0; n2 < NOP; n2++) {
		  dq1[n2] = gradn[n2][idb][1 - (nb % 2)];
		}

		gradient_bcs6x5_block(kappa0, kappa1, bcs[na], dq1, bc1, idb);

		f = 0.5*(1.0 - d_[ia][ib]);
		for (n1 = 0; n1 < 6; n1++) {
		  for (n2 = 0; n2 < NOP; n2++) {
		    b2[6*ia + n1] -= f*bc1[n1][n2];
		  }
		}

		/* Next block */
	      }

	      /* Known block ia and constants go to right-hand side */
	      assert(nunknown == 2); /* Known coordinate is normal[2] */

	      idb = normal[2];

	      for (n2 = 0; n2 < NOP; n2++) {
		dq1[n2] = dq[n2][idb];
	      }

	      gradient_bcs6x5_block(kappa0, kappa1, bcs[na], dq1, bc1, idb);

	      for (n1 = 0; n1 < 6; n1++) {
		for (n2 = 0; n2 < NOP; n2++) {
		  b2[6*ia + n1] -= bc1[n1][n2];
		}
	      }
	      b2[6*ia + XX] -= c[ia][X][X];
	      b2[6*ia + XY] -= 2.0*c[ia][X][Y];
	      b2[6*ia + XZ] -= 2.0*c[ia][X][Z];
	      b2[6*ia + YY] -= c[ia][Y][Y];
	      b2[6*ia + YZ] -= 2.0*c[ia][Y][Z];
	      b2[6*ia + ZZ] -= c[ia][Z][Z];
	    }

	    /* Solve */

	    ifail = util_svd_solve(2*6, 2*NOP, a2, b2, x2);
	    if (ifail != 0) fatal("SVD failed n = 2\n");

	    n = normal[0];
	    for (n1 = 0; n1 < NOP; n1++) {
	      gradn[n1][n/2][n % 2] = x2[n1];
	    }
	    n = normal[1];
	    for (n1 = 0; n1 < NOP; n1++) {
	      gradn[n1][n/2][n % 2] = x2[NOP + n1];
	    }
	  }
	  else if (nunknown == 3) {

	    int idb, na, nb;
	    double dq1[NOP];
	    double bc1[6][NOP];
	    double b3[3*6];
	    double x3[3*NOP];
	    double f;

	    for (n1 = 0; n1 < 3*6; n1++) {
	      b3[n1] = 0.0;
	    }

	    for (ia = 0; ia < nunknown; ia++) {

	      na = normal[ia];    /* normal for unknown ia */

	      for (ib = 0; ib < nunknown; ib++) {

		nb = normal[ib];
		idb = normal[ib]/2; /* coordinate direction of normal */

		/* Compute 6x5 block (ia,ib) in system matrix */

		for (n2 = 0; n2 < NOP; n2++) {
		  dq1[n2] = 1.0;
		}
		gradient_bcs6x5_block(kappa0, kappa1, bcs[na], dq1, bc1, idb);

		f = 0.5*(1.0 + d_[ia][ib]);
		for (n1 = 0; n1 < 6; n1++) {
		  for (n2 = 0; n2 < NOP; n2++) {
		    a3[6*ia + n1][NOP*ib + n2] = f*bc1[n1][n2];
		  }
		}

		/* Add known contributions to right hand side */

		for (n2 = 0; n2 < NOP; n2++) {
		  dq1[n2] = gradn[n2][idb][1 - (nb % 2)];
		}

		gradient_bcs6x5_block(kappa0, kappa1, bcs[na], dq1, bc1, idb);

		f = 0.5*(1.0 - d_[ia][ib]);
		for (n1 = 0; n1 < 6; n1++) {
		  for (n2 = 0; n2 < NOP; n2++) {
		    b3[6*ia + n1] -= f*bc1[n1][n2];
		  }
		}

		/* Next block */
	      }

	      /* Constants go to right-hand side */

	      b3[6*ia + XX] -= c[ia][X][X];
	      b3[6*ia + XY] -= 2.0*c[ia][X][Y];
	      b3[6*ia + XZ] -= 2.0*c[ia][X][Z];
	      b3[6*ia + YY] -= c[ia][Y][Y];
	      b3[6*ia + YZ] -= 2.0*c[ia][Y][Z];
	      b3[6*ia + ZZ] -= c[ia][Z][Z];
	    }

	    /* Solve */

	    ifail = util_svd_solve(3*6, 3*NOP, a3, b3, x3);
	    if (ifail != 0) fatal("SVD failed n = 3\n");

	    for (n = 0; n < nunknown; n++) {
	      for (n1 = 0; n1 < NOP; n1++) {
		gradn[n1][normal[n]/2][normal[n] % 2] = x3[NOP*n + n1];
	      }
	    }

	  }
	}

	/* The final answer is the sum of the partial gradients */

	for (n1 = 0; n1 < NOP; n1++) {
	  del2[NOP*index + n1] = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    grad[3*(NOP*index + n1) + ia] =
	      0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	    del2[NOP*index + n1] += gradn[n1][ia][0] - gradn[n1][ia][1];
	  }
	}

	/* Next site */
      }
    }
  }

  util_matrix_free(3*6, &a3);
  util_matrix_free(2*6, &a2);
  util_matrix_free(6, &a);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_bcs6
 *
 *  These are the coefficients of the d_c Q_ab terms in the
 *  boundary condition equation.
 *
 *  For the XX equation the coefficent of d_c Q_ab is bc[XX][ab][c]
 *  and so on. For known terms dq is unity, otherwise it is an
 *  estimate of the gradient d_c Q_ab which will end up on the
 *  right-hand side.
 *
 *  The XY, XZ, and YZ equations are multiplied a factor of two.
 *  The ZZ equation is included.
 *
 *****************************************************************************/

static void gradient_bcs6(double kappa0, double kappa1, const int dn[3],
			 double dq[6][3], double bc[6][6][3]) {

  double kappa2;

  kappa2 = kappa0 + kappa1;

  /* XX equation */

  bc[XX][XX][X] =  kappa0*dn[X]*dq[XX][X];
  bc[XX][XY][X] = -kappa1*dn[Y]*dq[XY][X];
  bc[XX][XZ][X] = -kappa1*dn[Z]*dq[XZ][X];
  bc[XX][YY][X] =  0.0;
  bc[XX][YZ][X] =  0.0;
  bc[XX][ZZ][X] =  0.0;

  bc[XX][XX][Y] = kappa1*dn[Y]*dq[XX][Y];
  bc[XX][XY][Y] = kappa0*dn[X]*dq[XY][Y];
  bc[XX][XZ][Y] = 0.0;
  bc[XX][YY][Y] = 0.0;
  bc[XX][YZ][Y] = 0.0;
  bc[XX][ZZ][Y] = 0.0;

  bc[XX][XX][Z] = kappa1*dn[Z]*dq[XX][Z];
  bc[XX][XY][Z] = 0.0;
  bc[XX][XZ][Z] = kappa0*dn[X]*dq[XZ][Z];
  bc[XX][YY][Z] = 0.0;
  bc[XX][YZ][Z] = 0.0;
  bc[XX][ZZ][Z] = 0.0;

  /* XY equation */

  bc[XY][XX][X] =  kappa0*dn[Y]*dq[XX][X];
  bc[XY][XY][X] =  kappa2*dn[X]*dq[XY][X];
  bc[XY][XZ][X] =  0.0;
  bc[XY][YY][X] = -kappa1*dn[Y]*dq[YY][X];
  bc[XY][YZ][X] = -kappa1*dn[Z]*dq[YZ][X];
  bc[XY][ZZ][X] = 0.0;

  bc[XY][XX][Y] = -kappa1*dn[X]*dq[XX][Y];
  bc[XY][XY][Y] =  kappa2*dn[Y]*dq[XY][Y];
  bc[XY][XZ][Y] = -kappa1*dn[Z]*dq[XZ][Y];
  bc[XY][YY][Y] =  kappa0*dn[X]*dq[YY][Y];
  bc[XY][YZ][Y] =  0.0;
  bc[XY][ZZ][Y] =  0.0;

  bc[XY][XX][Z] = 0.0;
  bc[XY][XY][Z] = 2.0*kappa1*dn[Z]*dq[XY][Z];
  bc[XY][XZ][Z] = kappa0*dn[Y]*dq[XZ][Z];
  bc[XY][YY][Z] = 0.0;
  bc[XY][YZ][Z] = kappa0*dn[X]*dq[YZ][Z];
  bc[XY][ZZ][Z] = 0.0;

  /* XZ equation */

  bc[XZ][XX][X] =  kappa0*dn[Z]*dq[XX][X];
  bc[XZ][XY][X] =  0.0;
  bc[XZ][XZ][X] =  kappa2*dn[X]*dq[XZ][X];
  bc[XZ][YY][X] =  0.0;
  bc[XZ][YZ][X] = -kappa1*dn[Y]*dq[YZ][X];
  bc[XZ][ZZ][X] = -kappa1*dn[Z]*dq[ZZ][X];

  bc[XZ][XX][Y] = 0.0;
  bc[XZ][XY][Y] = kappa0*dn[Z]*dq[XY][Y];
  bc[XZ][XZ][Y] = 2.0*kappa1*dn[Y]*dq[XZ][Y];
  bc[XZ][YY][Y] = 0.0;
  bc[XZ][YZ][Y] = kappa0*dn[X]*dq[YZ][Y];
  bc[XZ][ZZ][Y] = 0.0;

  bc[XZ][XX][Z] = -kappa1*dn[X]*dq[XX][Z];
  bc[XZ][XY][Z] = -kappa1*dn[Y]*dq[XY][Z];
  bc[XZ][XZ][Z] =  kappa2*dn[Z]*dq[XZ][Z];
  bc[XZ][YY][Z] =  0.0;
  bc[XZ][YZ][Z] =  0.0;
  bc[XZ][ZZ][Z] =  kappa0*dn[X]*dq[ZZ][Z];

  /* YY equation */

  bc[YY][XX][X] = 0.0;
  bc[YY][XY][X] = kappa0*dn[Y]*dq[XY][X];
  bc[YY][XZ][X] = 0.0;
  bc[YY][YY][X] = kappa1*dn[X]*dq[YY][X];
  bc[YY][YZ][X] = 0.0;
  bc[YY][ZZ][X] = 0.0;

  bc[YY][XX][Y] =  0.0;
  bc[YY][XY][Y] = -kappa1*dn[X]*dq[XY][Y];
  bc[YY][XZ][Y] =  0.0;
  bc[YY][YY][Y] =  kappa0*dn[Y]*dq[YY][Y];
  bc[YY][YZ][Y] = -kappa1*dn[Z]*dq[YZ][Y];
  bc[YY][ZZ][Y] =  0.0;

  bc[YY][XX][Z] = 0.0;
  bc[YY][XY][Z] = 0.0;
  bc[YY][XZ][Z] = 0.0;
  bc[YY][YY][Z] = kappa1*dn[Z]*dq[YY][Z];
  bc[YY][YZ][Z] = kappa0*dn[Y]*dq[YZ][Z];
  bc[YY][ZZ][Z] = 0.0;

  /* YZ equation */

  bc[YZ][XX][X] = 0.0;
  bc[YZ][XY][X] = kappa0*dn[Z]*dq[XY][X];
  bc[YZ][XZ][X] = kappa0*dn[Y]*dq[XZ][X];
  bc[YZ][YY][X] = 0.0;
  bc[YZ][YZ][X] = 2.0*kappa1*dn[X]*dq[YZ][X];
  bc[YZ][ZZ][X] = 0.0;

  bc[YZ][XX][Y] =  0.0;
  bc[YZ][XY][Y] =  0.0;
  bc[YZ][XZ][Y] = -kappa1*dn[X]*dq[XZ][Y];
  bc[YZ][YY][Y] =  kappa0*dn[Z]*dq[YY][Y];
  bc[YZ][YZ][Y] =  kappa2*dn[Y]*dq[YZ][Y];
  bc[YZ][ZZ][Y] = -kappa1*dn[Z]*dq[ZZ][Y];

  bc[YZ][XX][Z] =  0.0;
  bc[YZ][XY][Z] = -kappa1*dn[X]*dq[XY][Z];
  bc[YZ][XZ][Z] =  0.0;
  bc[YZ][YY][Z] = -kappa1*dn[Y]*dq[YY][Z];
  bc[YZ][YZ][Z] =  kappa2*dn[Z]*dq[YZ][Z];
  bc[YZ][ZZ][Z] =  kappa0*dn[Y]*dq[ZZ][Z];

  /* ZZ equation */

  bc[ZZ][XX][X] = 0.0;
  bc[ZZ][XY][X] = 0.0;
  bc[ZZ][XZ][X] = kappa0*dn[Z]*dq[XZ][X];
  bc[ZZ][YY][X] = 0.0;
  bc[ZZ][YZ][X] = 0.0;
  bc[ZZ][ZZ][X] = kappa1*dn[X]*dq[ZZ][X];
  
  bc[ZZ][XX][Y] = 0.0;
  bc[ZZ][XY][Y] = 0.0;
  bc[ZZ][XZ][Y] = 0.0;
  bc[ZZ][YY][Y] = 0.0;
  bc[ZZ][YZ][Y] = kappa0*dn[Z]*dq[YZ][Y];
  bc[ZZ][ZZ][Y] = kappa1*dn[Y]*dq[ZZ][Y];
  
  bc[ZZ][XX][Z] =  0.0;
  bc[ZZ][XY][Z] =  0.0;
  bc[ZZ][XZ][Z] = -kappa1*dn[X]*dq[XZ][Z];
  bc[ZZ][YY][Z] =  0.0;
  bc[ZZ][YZ][Z] = -kappa1*dn[Y]*dq[YZ][Z];
  bc[ZZ][ZZ][Z] =  kappa0*dn[Z]*dq[ZZ][Z];

  return;
}

/*****************************************************************************
 *
 *  gradient_bcs6x5
 *
 *  Here is a version which treats the 6x5 problem. The ZZ equation
 *  is interpreted as a constraint on the trace. dq[ZZ] does not
 *  appear, as in the above.
 *
 *****************************************************************************/

static void gradient_bcs6x5(double kappa0, double kappa1, const int dn[3],
			 double dq[NOP][3], double bc[6][NOP][3]) {

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

  /* ZZ equation */

  bc[ZZ][XX][X] = -kappa1*dn[X]*dq[XX][X];
  bc[ZZ][XY][X] =  0.0;
  bc[ZZ][XZ][X] =  kappa0*dn[Z]*dq[XZ][X];
  bc[ZZ][YY][X] = -kappa1*dn[X]*dq[YY][X];
  bc[ZZ][YZ][X] =  0.0;
  
  bc[ZZ][XX][Y] = -kappa1*dn[Y]*dq[XX][Y];
  bc[ZZ][XY][Y] =  0.0;
  bc[ZZ][XZ][Y] =  0.0;
  bc[ZZ][YY][Y] = -kappa1*dn[Y]*dq[YY][Y];
  bc[ZZ][YZ][Y] =  kappa0*dn[Z]*dq[YZ][Y];
  
  bc[ZZ][XX][Z] = -kappa0*dn[Z]*dq[XX][Z];
  bc[ZZ][XY][Z] =  0.0;
  bc[ZZ][XZ][Z] = -kappa1*dn[X]*dq[XZ][Z];
  bc[ZZ][YY][Z] = -kappa0*dn[Z]*dq[YY][Z];
  bc[ZZ][YZ][Z] = -kappa1*dn[Y]*dq[YZ][Z];

  return;
}

static void gradient_bcs6x5_block(double kappa0, double kappa1,
				  const int dn[3],
				  double dq[NOP], double bc[6][NOP],
				  int id) {

  double kappa2;

  kappa2 = kappa0 + kappa1;

  if (id == X) {

    /* d_x Q_ab terms, in which case dq_ab = d_x q_ab */

    bc[XX][XX] =  kappa0*dn[X]*dq[XX];
    bc[XX][XY] = -kappa1*dn[Y]*dq[XY];
    bc[XX][XZ] = -kappa1*dn[Z]*dq[XZ];
    bc[XX][YY] =  0.0;
    bc[XX][YZ] =  0.0;

    bc[XY][XX] =  kappa0*dn[Y]*dq[XX];
    bc[XY][XY] =  kappa2*dn[X]*dq[XY];
    bc[XY][XZ] =  0.0;
    bc[XY][YY] = -kappa1*dn[Y]*dq[YY];
    bc[XY][YZ] = -kappa1*dn[Z]*dq[YZ];

    bc[XZ][XX] =  kappa2*dn[Z]*dq[XX];
    bc[XZ][XY] =  0.0;
    bc[XZ][XZ] =  kappa2*dn[X]*dq[XZ];
    bc[XZ][YY] =  kappa1*dn[Z]*dq[YY];
    bc[XZ][YZ] = -kappa1*dn[Y]*dq[YZ];

    bc[YY][XX] = 0.0;
    bc[YY][XY] = kappa0*dn[Y]*dq[XY];
    bc[YY][XZ] = 0.0;
    bc[YY][YY] = kappa1*dn[X]*dq[YY];
    bc[YY][YZ] = 0.0;

    bc[YZ][XX] = 0.0;
    bc[YZ][XY] = kappa0*dn[Z]*dq[XY];
    bc[YZ][XZ] = kappa0*dn[Y]*dq[XZ];
    bc[YZ][YY] = 0.0;
    bc[YZ][YZ] = 2.0*kappa1*dn[X]*dq[YZ];

    bc[ZZ][XX] = -kappa1*dn[X]*dq[XX];
    bc[ZZ][XY] =  0.0;
    bc[ZZ][XZ] =  kappa0*dn[Z]*dq[XZ];
    bc[ZZ][YY] = -kappa1*dn[X]*dq[YY];
    bc[ZZ][YZ] =  0.0;
  }

  if (id == Y) {

    /* d_y Q_ab terms */

    bc[XX][XX] = kappa1*dn[Y]*dq[XX];
    bc[XX][XY] = kappa0*dn[X]*dq[XY];
    bc[XX][XZ] = 0.0;
    bc[XX][YY] = 0.0;
    bc[XX][YZ] = 0.0;

    bc[XY][XX] = -kappa1*dn[X]*dq[XX];
    bc[XY][XY] =  kappa2*dn[Y]*dq[XY];
    bc[XY][XZ] = -kappa1*dn[Z]*dq[XZ];
    bc[XY][YY] =  kappa0*dn[X]*dq[YY];
    bc[XY][YZ] =  0.0;

    bc[XZ][XX] = 0.0;
    bc[XZ][XY] = kappa0*dn[Z]*dq[XY];
    bc[XZ][XZ] = 2.0*kappa1*dn[Y]*dq[XZ];
    bc[XZ][YY] = 0.0;
    bc[XZ][YZ] = kappa0*dn[X]*dq[YZ];

    bc[YY][XX] =  0.0;
    bc[YY][XY] = -kappa1*dn[X]*dq[XY];
    bc[YY][XZ] =  0.0;
    bc[YY][YY] =  kappa0*dn[Y]*dq[YY];
    bc[YY][YZ] = -kappa1*dn[Z]*dq[YZ];

    bc[YZ][XX] =  kappa1*dn[Z]*dq[XX];
    bc[YZ][XY] =  0.0;
    bc[YZ][XZ] = -kappa1*dn[X]*dq[XZ];
    bc[YZ][YY] =  kappa2*dn[Z]*dq[YY];
    bc[YZ][YZ] =  kappa2*dn[Y]*dq[YZ];
  
    bc[ZZ][XX] = -kappa1*dn[Y]*dq[XX];
    bc[ZZ][XY] =  0.0;
    bc[ZZ][XZ] =  0.0;
    bc[ZZ][YY] = -kappa1*dn[Y]*dq[YY];
    bc[ZZ][YZ] =  kappa0*dn[Z]*dq[YZ];
  }

  if (id == Z) {

    /* d_z Q_ab terms */

    bc[XX][XX] = kappa1*dn[Z]*dq[XX];
    bc[XX][XY] = 0.0;
    bc[XX][XZ] = kappa0*dn[X]*dq[XZ];
    bc[XX][YY] = 0.0;
    bc[XX][YZ] = 0.0;

    bc[XY][XX] = 0.0;
    bc[XY][XY] = 2.0*kappa1*dn[Z]*dq[XY];
    bc[XY][XZ] = kappa0*dn[Y]*dq[XZ];
    bc[XY][YY] = 0.0;
    bc[XY][YZ] = kappa0*dn[X]*dq[YZ];

    bc[XZ][XX] = -kappa2*dn[X]*dq[XX];
    bc[XZ][XY] = -kappa1*dn[Y]*dq[XY];
    bc[XZ][XZ] =  kappa2*dn[Z]*dq[XZ];
    bc[XZ][YY] = -kappa0*dn[X]*dq[YY];
    bc[XZ][YZ] =  0.0;

    bc[YY][XX] = 0.0;
    bc[YY][XY] = 0.0;
    bc[YY][XZ] = 0.0;
    bc[YY][YY] = kappa1*dn[Z]*dq[YY];
    bc[YY][YZ] = kappa0*dn[Y]*dq[YZ];

    bc[YZ][XX] = -kappa0*dn[Y]*dq[XX];
    bc[YZ][XY] = -kappa1*dn[X]*dq[XY];
    bc[YZ][XZ] =  0.0;
    bc[YZ][YY] = -kappa2*dn[Y]*dq[YY];
    bc[YZ][YZ] =  kappa2*dn[Z]*dq[YZ];
  
    bc[ZZ][XX] = -kappa0*dn[Z]*dq[XX];
    bc[ZZ][XY] =  0.0;
    bc[ZZ][XZ] = -kappa1*dn[X]*dq[XZ];
    bc[ZZ][YY] = -kappa0*dn[Z]*dq[YY];
    bc[ZZ][YZ] = -kappa1*dn[Y]*dq[YZ];
  }

  return;
}

/*****************************************************************************
 *
 *  util_gaussian6
 *
 *  Solve linear system via Gaussian elimination. For the problems in this
 *  file, we only need to exchange rows, ie., have a partial pivot.
 *
 *  We solve Ax = b for A[6][6].
 *  xb is RHS on entry, and solution on exit.
 *  A is destroyed.
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

static int util_gaussian6(double a[6][6], double xb[6]) {

  int i, j, k;
  int ifail = 0;
  int iprow;
  int ipivot[6];

  double tmp;

  iprow = -1;
  for (k = 0; k < 6; k++) {
    ipivot[k] = -1;
  }

  for (k = 0; k < 6; k++) {

    /* Find pivot row */
    tmp = 0.0;
    for (i = 0; i < 6; i++) {
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
    for (j = k; j < 6; j++) {
      a[iprow][j] *= tmp;
    }
    xb[iprow] *= tmp;

    /* Subtract the pivot row (scaled) from remaining rows */

    for (i = 0; i < 6; i++) {
      if (ipivot[i] == -1) {
	tmp = a[i][k];
	for (j = k; j < 6; j++) {
	  a[i][j] -= tmp*a[iprow][j];
	}
	xb[i] -= tmp*xb[iprow];
      }
    }
  }

  /* Now do the back substitution */

  for (i = 6 - 1; i > -1; i--) {
    iprow = ipivot[i];
    tmp = xb[iprow];
    for (k = i + 1; k < 6; k++) {
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
