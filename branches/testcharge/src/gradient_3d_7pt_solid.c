/*****************************************************************************
 *
 *  gradient_3d_7pt_solid.c
 *
 *  Liquid Crystal tensor order parameter Q_ab.
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
#include "free_energy.h"
#include "blue_phase.h"
#include "colloids.h"
#include "colloids_Q_tensor.h"
#include "gradient_3d_7pt_solid.h"

static map_t * map_ = NULL;

/* Only tensor order parameter relevant */

#define NQAB 5

int util_gauss_solve(int mrow, double ** a, double * x, int * pivot);
int gradient_bcs6x5_coeff(double kappa0, double kappa1, const int dn[3],
			  double bc[6][NQAB][3]);
int gradient_bcs6x6_coeff(double kappa0, double kappa1, const int dn[3],
			  double bc[NSYMM][NSYMM][3]);

static int gradient_6x5_svd(const double * field, double * grad,
			    double * del2, const int nextra);
static int gradient_6x6_gauss_elim(const double * field, double * grad,
				   double * del2, const int nextra);

static void util_q5_to_qab(double q[3][3], const double * phi);

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_map_set
 *
 *****************************************************************************/

int gradient_3d_7pt_solid_map_set(map_t * map_in) {

  assert(map_in);

  map_ = map_in;

  return 0;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_operator
 *
 *  Compute the gradients in the fluid, and then correct for the presence
 *  of various solids.
 *
 *****************************************************************************/

int gradient_3d_7pt_solid_d2(const int nop, const double * field,
			     double * grad, double * delsq) {
  int nextra;
  int method = 1;

  assert(nop == NQAB);
  assert(map_);
  assert(field);
  assert(grad);
  assert(delsq);

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  if (method == 1) gradient_6x5_svd(field, grad, delsq, nextra);
  if (method == 2) gradient_6x6_gauss_elim(field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_6x5_svd
 *
 *  Six equations with five unknowns (not including Qzz).
 *  The Qzz equation does the job of a contraint on the trace.
 *
 *  Depending on how many boundary points are nearby, one gets
 *  a system of 6, 12, or 18 equations.
 *
 *****************************************************************************/

static int gradient_6x5_svd(const double * field, double * grad,
			    double * del2, const int nextra) {
  int nlocal[3];
  int nhalo;
  int ic, jc, kc;
  int ia, ib, ig, ih;
  int index, n, n1, n2;
  int ifail;
  int known[3];

  int str[3];
  int status0, status[6];

  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  int normal[6];
  int nunknown;
  int jcol;
  int idb;

  double gradn[NQAB][3][2];               /* Partial gradients */
  double q0[3][3];                        /* Prefered surface Q_ab */
  double qs[3][3];                        /* 'Surface' Q_ab */
  double ** a18;
  double b18[18];
  double x15[15];
  double bc[6][NQAB][3];                  /* Terms in boundary condition */
  double c[3][3];                         /* Constant terms in BC. */
  double dn[3];                           /* Unit normal. */
  double dq;
  double diag;
  double unkn;

  double w1_coll;                         /* Anchoring strength parameter */
  double w2_coll;                         /* Second anchoring parameter */
  double w1_wall;
  double w2_wall;
  double w1 = 0.0;
  double w2 = 0.0;
  double q_0;                             /* Cholesteric pitch wavevector */
  double kappa0;                          /* Elastic constants */
  double kappa1;

  double amplitude;                         /* Scalar order parameter */
  double qtilde[3][3];                      /* For planar anchoring */
  double q2;                                /* Contraction Q_ab Q_ab */

  assert(NQAB == 5);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  util_matrix_create(3*6, 3*NQAB, &a18);

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
	map_status(map_, index, &status0);

	if (status0 != MAP_FLUID) continue;

	/* Set up partial gradients and identify solid neighbours
	 * (unknowns) in various directions. If both neighbours
	 * in one coordinate direction are solid, treat as known. */

	nunknown = 0;

	for (ia = 0; ia < 3; ia++) {

	  known[ia] = 1;
	  normal[ia] = ia;

	  /* Look for outward normals is bcs[] with ib 2*ia + 1
	   * status at 2*ia, which is what we want */

	  ib = 2*ia + 1;
	  ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
	  map_status(map_, index + ib, status + 2*ia);

	  ib = 2*ia;
	  ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
	  map_status(map_, index + ib, status + 2*ia + 1);

	  ig = (status[2*ia    ] != MAP_FLUID);
	  ih = (status[2*ia + 1] != MAP_FLUID);

	  /* Calculate half-gradients assuming they are all knowns */

	  for (n1 = 0; n1 < NQAB; n1++) {
	    gradn[n1][ia][0] =
	      field[NQAB*(index + str[ia]) + n1] - field[NQAB*index + n1];
	    gradn[n1][ia][1] =
	      field[NQAB*index + n1] - field[NQAB*(index - str[ia]) + n1];
	  }

	  /* Set unknown, with direction, or treat as known (zero grad) */

	  if (ig + ih == 1) {
	    known[ia] = 0;
	    normal[nunknown] = 2*ia + ih;
	    nunknown += 1;
	  }
	  else  if (ig && ih) {
	    for (n1 = 0; n1 < NQAB; n1++) {
	      gradn[n1][ia][0] = 0.0;
	      gradn[n1][ia][1] = 0.0;
	    }
	  }

	}

	/* For planar anchoring we require qtilde_ab of Fournier and
	 * Galatola, and its square (reduendent for fluid sites) */

	util_q5_to_qab(qs, field + NQAB*index);

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
	  colloids_q_boundary(dn, qs, q0, status[normal[n]]);

	  /* Check for wall/colloid */
	  if (status[normal[n]] == MAP_COLLOID) {
	    w1 = w1_coll;
	    w2 = w2_coll;
	  }

	  if (status[normal[n]] == MAP_BOUNDARY) {
	    w1 = w1_wall;
	    w2 = w2_wall;
	  }

	  /* Compute c[a][b] */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      c[ia][ib] = 0.0;
	      for (ig = 0; ig < 3; ig++) {
		for (ih = 0; ih < 3; ih++) {
		  c[ia][ib] -= kappa1*q_0*bcs[normal[n]][ig]*
		    (e_[ia][ig][ih]*qs[ih][ib] + e_[ib][ig][ih]*qs[ih][ia]);
		}
	      }
	      /* Normal anchoring: w2 must be zero and q0 is preferred Q
	       * Planar anchoring: in w1 term q0 is effectively
	       *                   (Qtilde^perp - 0.5S_0) while in w2 we
	       *                   have Qtilde appearing explicitly.
	       *                   See colloids_q_boundary() etc */
	      c[ia][ib] +=
		-w1*(qs[ia][ib] - q0[ia][ib])
		-w2*(2.0*q2 - 4.5*amplitude*amplitude)*qtilde[ia][ib];
	    }
	  }


	  /* Now set up the system */
	  /* Initialise whole rows of A and b */

	  for (n1 = 0; n1 < 6; n1++) {
	    for (n2 = 0; n2 < 3*NQAB; n2++) {
	      a18[6*n + n1][n2] = 0.0;
	    }
	    b18[6*n + n1] = 0.0;
	  }

	  gradient_bcs6x5_coeff(kappa0, kappa1, bcs[normal[n]], bc);

	  /* Three blocks of columns for each row; note that the index
	   * ib is used to compute known terms, while idb is the
	   * appropriate normal direction for unknowns (counted by jcol) */

	  jcol = 0;

	  for (ib = 0; ib < 3; ib++) {

	    diag = d_[normal[n]/2][ib]; /* block diagonal? */
	    unkn = 1.0 - known[ib];     /* is ib unknown direction? */
	    idb = normal[jcol]/2;       /* normal direction for this unknown */

	    for (n1 = 0; n1 < 6; n1++) {
	      for (n2 = 0; n2 < NQAB; n2++) {

		/* Unknown diagonal blocks contribute full bc to A */
		/* Unknown off-diagonal blocks contribute (1/2) bc
		 * to A and (1/2) known dq to RHS.
		 * Known off-diagonals to RHS */

		a18[6*n + n1][NQAB*jcol + n2] += unkn*diag*bc[n1][n2][idb];
		a18[6*n + n1][NQAB*jcol + n2] += 0.5*unkn*(1.0 - diag)*bc[n1][n2][idb];

		dq = gradn[n2][idb][1 - (normal[jcol] % 2)];
		b18[6*n + n1] += 0.5*unkn*(1.0 - diag)*bc[n1][n2][idb]*dq;

		dq = 0.5*known[ib]*(gradn[n2][ib][0] + gradn[n2][ib][1]);
		b18[6*n + n1] += (1.0 - diag)*bc[n1][n2][ib]*dq;
	      }
	    }

	    jcol += (1 - known[ib]);
	  }

	  /* Constant terms all move to RHS (hence -ve sign) */

	  b18[6*n + XX] = -(b18[6*n + XX] +     c[X][X]);
	  b18[6*n + XY] = -(b18[6*n + XY] + 2.0*c[X][Y]);
	  b18[6*n + XZ] = -(b18[6*n + XZ] + 2.0*c[X][Z]);
	  b18[6*n + YY] = -(b18[6*n + YY] +     c[Y][Y]);
	  b18[6*n + YZ] = -(b18[6*n + YZ] + 2.0*c[Y][Z]);
	  b18[6*n + ZZ] = -(b18[6*n + ZZ] +     c[Z][Z]);
	}

	if (nunknown > 0) {
	  ifail = util_svd_solve(6*nunknown, NQAB*nunknown, a18, b18, x15);
	}

	for (n = 0; n < nunknown; n++) {
	  for (n1 = 0; n1 < NQAB; n1++) {
	    gradn[n1][normal[n]/2][normal[n] % 2] = x15[NQAB*n + n1];
	  }
	}

	/* The final answer is the sum of the partial gradients */

	for (n1 = 0; n1 < NQAB; n1++) {
	  del2[NQAB*index + n1] = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    grad[3*(NQAB*index + n1) + ia] =
	      0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	    del2[NQAB*index + n1] += gradn[n1][ia][0] - gradn[n1][ia][1];
	  }
	}

	/* Next site */
      }
    }
  }

  util_matrix_free(3*6, &a18);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_6x6_gauss_elim
 *
 *  This version treats Qzz as an unknown giving rise to
 *  same number of equations as unknowns, so we can use
 *  Gaussian elimination.
 *
 *  The contraint of tracelessness is imposed post-hoc.
 *
 *  The system is either 6x6, 12x12 or 18x18, depending on
 *  the geometry.
 *
 *****************************************************************************/

static int gradient_6x6_gauss_elim(const double * field, double * grad,
				   double * del2, const int nextra) {
  int nlocal[3];
  int nhalo;
  int ic, jc, kc;
  int ia, ib, ig, ih;
  int index, n, n1, n2;
  int ifail;
  int known[3];

  int str[3];
  int status0, status[6];
  int pivot18[18];

  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  int normal[6];
  int nunknown;
  int jcol;
  int idb;

  double gradn[NSYMM][3][2];              /* Partial gradients */
  double q0[3][3];                        /* Prefered surface Q_ab */
  double qs[3][3];                        /* 'Surface' Q_ab */
  double ** a18;
  double xb18[18];
  double bc[NSYMM][NSYMM][3];             /* Terms in boundary condition */
  double c[3][3];                         /* Constant terms in BC. */
  double dn[3];                           /* Unit normal. */
  double dq;
  double diag;
  double unkn;
  double tr;

  double w1_coll;                         /* Anchoring strength parameter */
  double w2_coll;                         /* Second anchoring parameter */
  double w1_wall;
  double w2_wall;
  double w1 = 0.0;
  double w2 = 0.0;
  double q_0;                             /* Cholesteric pitch wavevector */
  double kappa0;                          /* Elastic constants */
  double kappa1;

  double amplitude;                         /* Scalar order parameter */
  double qtilde[3][3];                      /* For planar anchoring */
  double q2;                                /* Contraction Q_ab Q_ab */

  assert(NQAB == 5);
  assert(NSYMM == 6);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  util_matrix_create(3*NSYMM, 3*NSYMM, &a18);

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
	map_status(map_, index, &status0);

	if (status0 != MAP_FLUID) continue;

	/* Set up partial gradients and identify solid neighbours
	 * (unknowns) in various directions. If both neighbours
	 * in one coordinate direction are solid, treat as known. */

	nunknown = 0;

	for (ia = 0; ia < 3; ia++) {

	  known[ia] = 1;
	  normal[ia] = ia;

	  /* Look for ouward normals is bcs[] */

	  ib = 2*ia + 1;
	  ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
	  map_status(map_, index + ib, status + 2*ia);

	  ib = 2*ia;
	  ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
	  map_status(map_, index + ib, status + 2*ia + 1);

	  ig = (status[2*ia    ] != MAP_FLUID);
	  ih = (status[2*ia + 1] != MAP_FLUID);

	  /* Calculate half-gradients assuming they are all knowns */

	  for (n1 = 0; n1 < NQAB; n1++) {
	    gradn[n1][ia][0] =
	      field[NQAB*(index + str[ia]) + n1] - field[NQAB*index + n1];
	    gradn[n1][ia][1] =
	      field[NQAB*index + n1] - field[NQAB*(index - str[ia]) + n1];
	  }

	  gradn[ZZ][ia][0] = -gradn[XX][ia][0] - gradn[YY][ia][0];
	  gradn[ZZ][ia][1] = -gradn[XX][ia][1] - gradn[YY][ia][1];

	  /* Set unknown, with direction, or treat as known (zero grad) */

	  if (ig + ih == 1) {
	    known[ia] = 0;
	    normal[nunknown] = 2*ia + ih;
	    nunknown += 1;
	  }
	  else  if (ig && ih) {
	    for (n1 = 0; n1 < NSYMM; n1++) {
	      gradn[n1][ia][0] = 0.0;
	      gradn[n1][ia][1] = 0.0;
	    }
	  }

	}

	/* For planar anchoring we require qtilde_ab of Fournier and
	 * Galatola, and its square (reduendent for fluid sites) */

	util_q5_to_qab(qs, field + NQAB*index);

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
	  colloids_q_boundary(dn, qs, q0, status[normal[n]]);

	  /* Check for wall/colloid */
	  if (status[normal[n]] == MAP_COLLOID) {
	    w1 = w1_coll;
	    w2 = w2_coll;
	  }

	  if (status[normal[n]] == MAP_BOUNDARY) {
	    w1 = w1_wall;
	    w2 = w2_wall;
	  }

	  /* Compute c[a][b] */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      c[ia][ib] = 0.0;
	      for (ig = 0; ig < 3; ig++) {
		for (ih = 0; ih < 3; ih++) {
		  c[ia][ib] -= kappa1*q_0*bcs[normal[n]][ig]*
		    (e_[ia][ig][ih]*qs[ih][ib] + e_[ib][ig][ih]*qs[ih][ia]);
		}
	      }
	      /* Normal anchoring: w2 must be zero and q0 is preferred Q
	       * Planar anchoring: in w1 term q0 is effectively
	       *                   (Qtilde^perp - 0.5S_0) while in w2 we
	       *                   have Qtilde appearing explicitly.
	       *                   See colloids_q_boundary() etc */
	      c[ia][ib] +=
		-w1*(qs[ia][ib] - q0[ia][ib])
		-w2*(2.0*q2 - 4.5*amplitude*amplitude)*qtilde[ia][ib];
	    }
	  }


	  /* Now set up the system */
	  /* Initialise whole rows of A and b */

	  for (n1 = 0; n1 < NSYMM; n1++) {
	    for (n2 = 0; n2 < 3*NSYMM; n2++) {
	      a18[NSYMM*n + n1][n2] = 0.0;
	    }
	    xb18[NSYMM*n + n1] = 0.0;
	  }

	  gradient_bcs6x6_coeff(kappa0, kappa1, bcs[normal[n]], bc);

	  /* Three blocks of columns for each row; note that the index
	   * ib is used to compute known terms, while idb is the
	   * appropriate normal direction for unknowns (counted by jcol) */

	  jcol = 0;

	  for (ib = 0; ib < 3; ib++) {

	    diag = d_[normal[n]/2][ib]; /* block diagonal? */
	    unkn = 1.0 - known[ib];     /* is ib unknown direction? */
	    idb = normal[jcol]/2;       /* normal direction for this unknown */

	    for (n1 = 0; n1 < NSYMM; n1++) {
	      for (n2 = 0; n2 < NSYMM; n2++) {

		/* Unknown diagonal blocks contribute full bc to A */
		/* Unknown off-diagonal blocks contribute (1/2) bc
		 * to A and (1/2) known dq to RHS.
		 * Known off-diagonals to RHS */

		a18[NSYMM*n + n1][NSYMM*jcol + n2] += unkn*diag*bc[n1][n2][idb];
		a18[NSYMM*n + n1][NSYMM*jcol + n2] += 0.5*unkn*(1.0 - diag)*bc[n1][n2][idb];

		dq = gradn[n2][idb][1 - (normal[jcol] % 2)];
		xb18[NSYMM*n + n1] += 0.5*unkn*(1.0 - diag)*bc[n1][n2][idb]*dq;

		dq = 0.5*known[ib]*(gradn[n2][ib][0] + gradn[n2][ib][1]);
		xb18[NSYMM*n + n1] += (1.0 - diag)*bc[n1][n2][ib]*dq;
	      }
	    }

	    jcol += (1 - known[ib]);
	  }

	  /* Constant terms all move to RHS (hence -ve sign). Factors
	   * of two in off-diagonals agree with coefficients. */

	  xb18[NSYMM*n + XX] = -(xb18[NSYMM*n + XX] +     c[X][X]);
	  xb18[NSYMM*n + XY] = -(xb18[NSYMM*n + XY] + 2.0*c[X][Y]);
	  xb18[NSYMM*n + XZ] = -(xb18[NSYMM*n + XZ] + 2.0*c[X][Z]);
	  xb18[NSYMM*n + YY] = -(xb18[NSYMM*n + YY] +     c[Y][Y]);
	  xb18[NSYMM*n + YZ] = -(xb18[NSYMM*n + YZ] + 2.0*c[Y][Z]);
	  xb18[NSYMM*n + ZZ] = -(xb18[NSYMM*n + ZZ] +     c[Z][Z]);
	}

	if (nunknown > 0) ifail =
	    util_gauss_solve(NSYMM*nunknown, a18, xb18, pivot18);

	for (n = 0; n < nunknown; n++) {

	  /* Fix trace (don't care about Qzz in the end) */

	  tr = r3_*(xb18[NSYMM*n + XX] + xb18[NSYMM*n + YY] + xb18[NSYMM*n + ZZ]);
	  xb18[NSYMM*n + XX] -= tr;
	  xb18[NSYMM*n + YY] -= tr;

	  /* Store missing half gradients */

	  for (n1 = 0; n1 < NQAB; n1++) {
	    gradn[n1][normal[n]/2][normal[n] % 2] = xb18[NSYMM*n + n1];
	  }
	}

	/* The final answer is the sum of the partial gradients */

	for (n1 = 0; n1 < NQAB; n1++) {
	  del2[NQAB*index + n1] = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    grad[3*(NQAB*index + n1) + ia] =
	      0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	    del2[NQAB*index + n1] += gradn[n1][ia][0] - gradn[n1][ia][1];
	  }
	}

	/* Next site */
      }
    }
  }

  util_matrix_free(3*6, &a18);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_bcs6x5_coeff
 *
 *  Coefficients in the boundary condition equation for outward surface
 *  normal dn[3].
 *
 *  This is just a look-up table.
 *  Terms in the off-diagonal equations are multiplied by two for
 *  convenience.
 *
 *****************************************************************************/

int gradient_bcs6x5_coeff(double kappa0, double kappa1, const int dn[3],
			  double bc[6][NQAB][3]) {

  double kappa2;

  kappa2 = kappa0 + kappa1;

  /* XX equation */

  bc[XX][XX][X] =  kappa0*dn[X];
  bc[XX][XY][X] = -kappa1*dn[Y];
  bc[XX][XZ][X] = -kappa1*dn[Z];
  bc[XX][YY][X] =  0.0;
  bc[XX][YZ][X] =  0.0;

  bc[XX][XX][Y] = kappa1*dn[Y];
  bc[XX][XY][Y] = kappa0*dn[X];
  bc[XX][XZ][Y] = 0.0;
  bc[XX][YY][Y] = 0.0;
  bc[XX][YZ][Y] = 0.0;

  bc[XX][XX][Z] = kappa1*dn[Z];
  bc[XX][XY][Z] = 0.0;
  bc[XX][XZ][Z] = kappa0*dn[X];
  bc[XX][YY][Z] = 0.0;
  bc[XX][YZ][Z] = 0.0;

  /* XY equation */

  bc[XY][XX][X] =  kappa0*dn[Y];
  bc[XY][XY][X] =  kappa2*dn[X];
  bc[XY][XZ][X] =  0.0;
  bc[XY][YY][X] = -kappa1*dn[Y];
  bc[XY][YZ][X] = -kappa1*dn[Z];

  bc[XY][XX][Y] = -kappa1*dn[X];
  bc[XY][XY][Y] =  kappa2*dn[Y];
  bc[XY][XZ][Y] = -kappa1*dn[Z];
  bc[XY][YY][Y] =  kappa0*dn[X];
  bc[XY][YZ][Y] =  0.0;

  bc[XY][XX][Z] = 0.0;
  bc[XY][XY][Z] = 2.0*kappa1*dn[Z];
  bc[XY][XZ][Z] = kappa0*dn[Y];
  bc[XY][YY][Z] = 0.0;
  bc[XY][YZ][Z] = kappa0*dn[X];

  /* XZ equation */

  bc[XZ][XX][X] =  kappa2*dn[Z];
  bc[XZ][XY][X] =  0.0;
  bc[XZ][XZ][X] =  kappa2*dn[X];
  bc[XZ][YY][X] =  kappa1*dn[Z];
  bc[XZ][YZ][X] = -kappa1*dn[Y];

  bc[XZ][XX][Y] = 0.0;
  bc[XZ][XY][Y] = kappa0*dn[Z];
  bc[XZ][XZ][Y] = 2.0*kappa1*dn[Y];
  bc[XZ][YY][Y] = 0.0;
  bc[XZ][YZ][Y] = kappa0*dn[X];

  bc[XZ][XX][Z] = -kappa2*dn[X];
  bc[XZ][XY][Z] = -kappa1*dn[Y];
  bc[XZ][XZ][Z] =  kappa2*dn[Z];
  bc[XZ][YY][Z] = -kappa0*dn[X];
  bc[XZ][YZ][Z] =  0.0;

  /* YY equation */

  bc[YY][XX][X] = 0.0;
  bc[YY][XY][X] = kappa0*dn[Y];
  bc[YY][XZ][X] = 0.0;
  bc[YY][YY][X] = kappa1*dn[X];
  bc[YY][YZ][X] = 0.0;

  bc[YY][XX][Y] =  0.0;
  bc[YY][XY][Y] = -kappa1*dn[X];
  bc[YY][XZ][Y] =  0.0;
  bc[YY][YY][Y] =  kappa0*dn[Y];
  bc[YY][YZ][Y] = -kappa1*dn[Z];

  bc[YY][XX][Z] = 0.0;
  bc[YY][XY][Z] = 0.0;
  bc[YY][XZ][Z] = 0.0;
  bc[YY][YY][Z] = kappa1*dn[Z];
  bc[YY][YZ][Z] = kappa0*dn[Y];

  /* YZ equation */

  bc[YZ][XX][X] = 0.0;
  bc[YZ][XY][X] = kappa0*dn[Z];
  bc[YZ][XZ][X] = kappa0*dn[Y];
  bc[YZ][YY][X] = 0.0;
  bc[YZ][YZ][X] = 2.0*kappa1*dn[X];

  bc[YZ][XX][Y] =  kappa1*dn[Z];
  bc[YZ][XY][Y] =  0.0;
  bc[YZ][XZ][Y] = -kappa1*dn[X];
  bc[YZ][YY][Y] =  kappa2*dn[Z];
  bc[YZ][YZ][Y] =  kappa2*dn[Y];

  bc[YZ][XX][Z] = -kappa0*dn[Y];
  bc[YZ][XY][Z] = -kappa1*dn[X];
  bc[YZ][XZ][Z] =  0.0;
  bc[YZ][YY][Z] = -kappa2*dn[Y];
  bc[YZ][YZ][Z] =  kappa2*dn[Z];

  /* ZZ equation */

  bc[ZZ][XX][X] = -kappa1*dn[X];
  bc[ZZ][XY][X] =  0.0;
  bc[ZZ][XZ][X] =  kappa0*dn[Z];
  bc[ZZ][YY][X] = -kappa1*dn[X];
  bc[ZZ][YZ][X] =  0.0;
  
  bc[ZZ][XX][Y] = -kappa1*dn[Y];
  bc[ZZ][XY][Y] =  0.0;
  bc[ZZ][XZ][Y] =  0.0;
  bc[ZZ][YY][Y] = -kappa1*dn[Y];
  bc[ZZ][YZ][Y] =  kappa0*dn[Z];
  
  bc[ZZ][XX][Z] = -kappa0*dn[Z];
  bc[ZZ][XY][Z] =  0.0;
  bc[ZZ][XZ][Z] = -kappa1*dn[X];
  bc[ZZ][YY][Z] = -kappa0*dn[Z];
  bc[ZZ][YZ][Z] = -kappa1*dn[Y];

  return 0;
}

/*****************************************************************************
 *
 *  gradient_bcs6x6_coeff
 *
 *  Full set of coefficients in boundary condition equation for given
 *  surface normal dn.
 *
 *****************************************************************************/

int gradient_bcs6x6_coeff(double kappa0, double kappa1, const int dn[3],
			  double bc[NSYMM][NSYMM][3]) {
  double kappa2;

  kappa2 = kappa0 + kappa1;

  /* XX equation */

  bc[XX][XX][X] =  kappa0*dn[X];
  bc[XX][XY][X] = -kappa1*dn[Y];
  bc[XX][XZ][X] = -kappa1*dn[Z];
  bc[XX][YY][X] =  0.0;
  bc[XX][YZ][X] =  0.0;
  bc[XX][ZZ][X] =  0.0;

  bc[XX][XX][Y] =  kappa1*dn[Y];
  bc[XX][XY][Y] =  kappa0*dn[X];
  bc[XX][XZ][Y] =  0.0;
  bc[XX][YY][Y] =  0.0;
  bc[XX][YZ][Y] =  0.0;
  bc[XX][ZZ][Y] =  0.0;

  bc[XX][XX][Z] =  kappa1*dn[Z];
  bc[XX][XY][Z] =  0.0;
  bc[XX][XZ][Z] =  kappa0*dn[X];
  bc[XX][YY][Z] =  0.0;
  bc[XX][YZ][Z] =  0.0;
  bc[XX][ZZ][Z] =  0.0;

  /* XY equation */

  bc[XY][XX][X] =  kappa0*dn[Y];
  bc[XY][XY][X] =  kappa2*dn[X];
  bc[XY][XZ][X] =  0.0;
  bc[XY][YY][X] = -kappa1*dn[Y];
  bc[XY][YZ][X] = -kappa1*dn[Z];
  bc[XY][ZZ][X] =  0.0;

  bc[XY][XX][Y] = -kappa1*dn[X];
  bc[XY][XY][Y] =  kappa2*dn[Y];
  bc[XY][XZ][Y] = -kappa1*dn[Z];
  bc[XY][YY][Y] =  kappa0*dn[X];
  bc[XY][YZ][Y] =  0.0;
  bc[XY][ZZ][Y] =  0.0;

  bc[XY][XX][Z] =  0.0;
  bc[XY][XY][Z] =  2.0*kappa1*dn[Z];
  bc[XY][XZ][Z] =  kappa0*dn[Y];
  bc[XY][YY][Z] =  0.0;
  bc[XY][YZ][Z] =  kappa0*dn[X];
  bc[XY][ZZ][Z] =  0.0;

  /* XZ equation */

  bc[XZ][XX][X] =  kappa0*dn[Z];
  bc[XZ][XY][X] =  0.0;
  bc[XZ][XZ][X] =  kappa2*dn[X];
  bc[XZ][YY][X] =  0.0;
  bc[XZ][YZ][X] = -kappa1*dn[Y];
  bc[XZ][ZZ][X] = -kappa1*dn[Z];

  bc[XZ][XX][Y] =  0.0;
  bc[XZ][XY][Y] =  kappa0*dn[Z];
  bc[XZ][XZ][Y] =  2.0*kappa1*dn[Y];
  bc[XZ][YY][Y] =  0.0;
  bc[XZ][YZ][Y] =  kappa0*dn[X];
  bc[XZ][ZZ][Y] =  0.0;

  bc[XZ][XX][Z] = -kappa1*dn[X];
  bc[XZ][XY][Z] = -kappa1*dn[Y];
  bc[XZ][XZ][Z] =  kappa2*dn[Z];
  bc[XZ][YY][Z] =  0.0;
  bc[XZ][YZ][Z] =  0.0;
  bc[XZ][ZZ][Z] =  kappa0*dn[X];

  /* YY equation */

  bc[YY][XX][X] =  0.0;
  bc[YY][XY][X] =  kappa0*dn[Y];
  bc[YY][XZ][X] =  0.0;
  bc[YY][YY][X] =  kappa1*dn[X];
  bc[YY][YZ][X] =  0.0;
  bc[YY][ZZ][X] =  0.0;

  bc[YY][XX][Y] =  0.0;
  bc[YY][XY][Y] = -kappa1*dn[X];
  bc[YY][XZ][Y] =  0.0;
  bc[YY][YY][Y] =  kappa0*dn[Y];
  bc[YY][YZ][Y] = -kappa1*dn[Z];
  bc[YY][ZZ][Y] =  0.0;

  bc[YY][XX][Z] =  0.0;
  bc[YY][XY][Z] =  0.0;
  bc[YY][XZ][Z] =  0.0;
  bc[YY][YY][Z] =  kappa1*dn[Z];
  bc[YY][YZ][Z] =  kappa0*dn[Y];
  bc[YY][ZZ][Z] =  0.0;

  /* YZ equation */

  bc[YZ][XX][X] =  0.0;
  bc[YZ][XY][X] =  kappa0*dn[Z];
  bc[YZ][XZ][X] =  kappa0*dn[Y];
  bc[YZ][YY][X] =  0.0;
  bc[YZ][YZ][X] =  2.0*kappa1*dn[X];
  bc[YZ][ZZ][X] =  0.0;

  bc[YZ][XX][Y] =  0.0;
  bc[YZ][XY][Y] =  0.0;
  bc[YZ][XZ][Y] = -kappa1*dn[X];
  bc[YZ][YY][Y] =  kappa0*dn[Z];
  bc[YZ][YZ][Y] =  kappa2*dn[Y];
  bc[YZ][ZZ][Y] = -kappa1*dn[Z];

  bc[YZ][XX][Z] =  0.0;
  bc[YZ][XY][Z] = -kappa1*dn[X];
  bc[YZ][XZ][Z] =  0.0;
  bc[YZ][YY][Z] = -kappa1*dn[Y];
  bc[YZ][YZ][Z] =  kappa2*dn[Z];
  bc[YZ][ZZ][Z] =  kappa0*dn[Y];

  /* ZZ equation */

  bc[ZZ][XX][X] =  0.0;
  bc[ZZ][XY][X] =  0.0;
  bc[ZZ][XZ][X] =  kappa0*dn[Z];
  bc[ZZ][YY][X] =  0.0;
  bc[ZZ][YZ][X] =  0.0;
  bc[ZZ][ZZ][X] =  kappa1*dn[X];
  
  bc[ZZ][XX][Y] =  0.0;
  bc[ZZ][XY][Y] =  0.0;
  bc[ZZ][XZ][Y] =  0.0;
  bc[ZZ][YY][Y] =  0.0;
  bc[ZZ][YZ][Y] =  kappa0*dn[Z];
  bc[ZZ][ZZ][Y] =  kappa1*dn[Y];
  
  bc[ZZ][XX][Z] =  0.0;
  bc[ZZ][XY][Z] =  0.0;
  bc[ZZ][XZ][Z] = -kappa1*dn[X];
  bc[ZZ][YY][Z] =  0.0;
  bc[ZZ][YZ][Z] = -kappa1*dn[Y];
  bc[ZZ][ZZ][Z] =  kappa0*dn[Z];

  return 0;
}

/*****************************************************************************
 *
 *  util_gauss_solve
 *
 *  Solve linear system via Gaussian elimination. For the problems in this
 *  file, we only need to exchange rows, ie., have a partial pivot.
 *
 *  We solve Ax = b for A[MROW][MROW].
 *  x[MROW] is RHS on entry, and solution on exit.
 *  A is destroyed.
 *  Workspace for the pivot rows must be supplied: pivot[MROW].
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int util_gauss_solve(int mrow, double ** a, double * x, int * pivot) {

  int i, j, k;
  int iprow;
  double tmp;

  assert(a);
  assert(x);
  assert(pivot);

  iprow = -1;
  for (k = 0; k < mrow; k++) {
    pivot[k] = -1;
  }

  for (k = 0; k < mrow; k++) {

    /* Find pivot row */
    tmp = 0.0;
    for (i = 0; i < mrow; i++) {
      if (pivot[i] == -1) {
	if (fabs(a[i][k]) >= tmp) {
	  tmp = fabs(a[i][k]);
	  iprow = i;
	}
      }
    }
    pivot[k] = iprow;

    /* divide pivot row by the pivot element a[iprow][k] */

    if (a[iprow][k] == 0.0) return -1;

    tmp = 1.0 / a[iprow][k];
    for (j = k; j < mrow; j++) {
      a[iprow][j] *= tmp;
    }
    x[iprow] *= tmp;

    /* Subtract the pivot row (scaled) from remaining rows */

    for (i = 0; i < mrow; i++) {
      if (pivot[i] == -1) {
	tmp = a[i][k];
	for (j = k; j < mrow; j++) {
	  a[i][j] -= tmp*a[iprow][j];
	}
	x[i] -= tmp*x[iprow];
      }
    }
  }

  /* Now do the back substitution */

  for (i = mrow - 1; i > -1; i--) {
    iprow = pivot[i];
    tmp = x[iprow];
    for (k = i + 1; k < mrow; k++) {
      tmp -= a[iprow][k]*x[pivot[k]];
    }
    x[iprow] = tmp;
  }

  return 0;
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
