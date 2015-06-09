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
#include "colloids_s.h"
#include "colloids_Q_tensor.h"
#include "gradient_3d_7pt_solid.h"
#include "targetDP.h"
#include "map_s.h"

static map_t * map_ = NULL;

/* Only tensor order parameter relevant */

#define NQAB 5

int util_gauss_solve(int mrow, double ** a, double * x, int * pivot);
__targetHost__ __target__ int gradient_bcs6x5_coeff(double kappa0, double kappa1, const int dn[3],
			  double bc[6][NQAB][3]);
__targetHost__ __target__ int gradient_bcs6x6_coeff(double kappa0, double kappa1, const int dn[3],
			  double bc[NSYMM][NSYMM][3]);

static int gradient_6x5_svd(const double * field, double * grad,
			    double * del2, const int nextra);
static int gradient_6x6_gauss_elim(const double * field, double * grad,
				   double * del2, const int nextra);
static int gradient_6x6_gpu(const double * field, double * grad,
				   double * del2, const int nextra);

__targetHost__ __target__ static void util_q5_to_qab(double q[3][3], const double * phi);

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_map_set
 *
 *****************************************************************************/

__targetHost__ int gradient_3d_7pt_solid_map_set(map_t * map_in) {

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

__targetHost__ int gradient_3d_7pt_solid_d2(const int nop, const double * field,double * t_field,
				double * grad,double * t_grad, double * delsq, double * t_delsq) {
  int nextra;
  //  int method = 1;

    int method = 3;

  assert(nop == NQAB);
  assert(map_);
  assert(field);
  assert(grad);
  assert(delsq);

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  int Nall[3];
  int nlocal[3];
  coords_nlocal(nlocal);
  int nhalo=coords_nhalo();
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];

  if (method == 1) gradient_6x5_svd(field, grad, delsq, nextra);
  if (method == 2) gradient_6x6_gauss_elim(field, grad, delsq, nextra);
  if (method == 3){

    #ifndef KEEPFIELDONTARGET
    copyToTarget(t_field,field,nop*nSites*sizeof(double)); 
    #endif

    gradient_6x6_gpu(t_field, t_grad, t_delsq, nextra);
    
    #ifndef KEEPFIELDONTARGET
    copyFromTarget(grad,t_grad,nop*3*nSites*sizeof(double)); 
    copyFromTarget(delsq,t_delsq,nop*nSites*sizeof(double)); 
    #endif

  }
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
  ifail = 0;

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
	  ifail += util_svd_solve(6*nunknown, NQAB*nunknown, a18, b18, x15);
	  assert(ifail == 0);
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
  if (ifail > 0) fatal("Failure in gradient SVD\n");

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
  ifail = 0;

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

	if (nunknown > 0) {
	  ifail += util_gauss_solve(NSYMM*nunknown, a18, xb18, pivot18);
	  assert(ifail == 0);
	}

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
  if (ifail > 0) fatal("Failure in gradient Gaussian elimination\n");

  return 0;
}

/*****************************************************************************
 *
 *  gpu version
 *
 *  This solves the boundary condition equation by pre-computing
 *  the inverse of the system matrix for a number of cases.
 *
 *****************************************************************************/


__targetConst__ double tc_a6inv[3][6]; 
__targetConst__ double tc_a18inv[18][18]; 
__targetConst__ double tc_a12inv[3][12][12]; 


__targetEntry__ void gradient_6x6_gpu_lattice(const double * field, double * grad,
					      double * del2,  map_t * map,bluePhaseKernelConstants_t* pbpc, colloids_info_t* cinfo) {

  int ic, jc, kc, index;
  int str[3];
  int ia, ib, n1, n2;
  int ih, ig;
  int n, nunknown;
  int status[6];
  int normal[3];
  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  const double bcsign[6] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};

  double gradn[6][3][2];          /* one-sided partial gradients */
  double dq;
  double qs[3][3];
  double c[3][3];

  double bc[6][6][3];
  double b18[18];
  double x18[18];
  double tr;
  const double r3 = (1.0/3.0);

  str[Z]=1;
  str[Y]=tc_Nall[Z];
  str[X]=tc_Nall[Z]*tc_Nall[Y];

__targetTLP__(index,tc_nSites){
    
    int coords[3];
    targetCoords3D(coords,tc_Nall,index);

    // if not a halo site:
    if (coords[0] >= (tc_nhalo-tc_nextra) &&
	coords[1] >= (tc_nhalo-tc_nextra) &&
	coords[2] >= (tc_nhalo-tc_nextra) &&
	coords[0] < tc_Nall[X]-(tc_nhalo-tc_nextra) &&
	coords[1] < tc_Nall[Y]-(tc_nhalo-tc_nextra)  &&
	coords[2] < tc_Nall[Z]-(tc_nhalo-tc_nextra) ){

      
      ic=coords[0]+1-tc_nhalo;
      jc=coords[1]+1-tc_nhalo;
      kc=coords[2]+1-tc_nhalo;
      
      //map_status(map_, index, &status0);
      
      //if (status0 != MAP_FLUID) continue;
      
      if ((map->status[index])==MAP_FLUID){

      /* Set up partial gradients and identify solid neighbours
       * (unknowns) in various directions. If both neighbours
       * in one coordinate direction are solid, treat as known. */
      
      nunknown = 0;
      
      for (ia = 0; ia < 3; ia++) {
	
	normal[ia] = ia;
	
	/* Look for ouward normals is bcs[] */
	
	ib = 2*ia + 1;
	ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
	//map_status(map_, index + ib, status + 2*ia);
	status[2*ia]=map->status[index+ib];	

	ib = 2*ia;
	ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
	//map_status(map_, index + ib, status + 2*ia + 1);
	status[2*ia+1]=map->status[index+ib];	


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
	  normal[nunknown] = 2*ia + ih;
	  nunknown += 1;
	}
	else if (ig && ih) {
	  for (n1 = 0; n1 < NSYMM; n1++) {
	    gradn[n1][ia][0] = 0.0;
	    gradn[n1][ia][1] = 0.0;
	  }
	}
	
      }
      

      /* Boundary condition constant terms */
      
      if (nunknown > 0) {
	
	/* Fluid Qab at surface */
	
	util_q5_to_qab(qs, field + NQAB*index);


	
	//TODO NEED TO PORT
	q_boundary_constants(ic, jc, kc, qs, bcs[normal[0]],
			     status[normal[0]], c, pbpc, cinfo);
	
	/* Constant terms all move to RHS (hence -ve sign). Factors
	 * of two in off-diagonals agree with matrix coefficients. */
	
	b18[XX] = -1.0*c[X][X];
	b18[XY] = -2.0*c[X][Y];
	b18[XZ] = -2.0*c[X][Z];
	b18[YY] = -1.0*c[Y][Y];
	b18[YZ] = -2.0*c[Y][Z];
	b18[ZZ] = -1.0*c[Z][Z];
	
	/* Fill a a known value in unknown position so we
	 * and compute a gradient as 0.5*(grad[][][0] + gradn[][][1]) */
	ig = normal[0]/2;
	ih = normal[0]%2;
	for (n1 = 0; n1 < NSYMM; n1++) {
	  gradn[n1][ig][ih] = gradn[n1][ig][1 - ih];
	}
      }
      
      if (nunknown > 1) {
	
	//TODO NEED TO PORT
	q_boundary_constants(ic, jc, kc, qs, bcs[normal[1]],
			     status[normal[1]], c, pbpc, cinfo);
	
	b18[1*NSYMM + XX] = -1.0*c[X][X];
	b18[1*NSYMM + XY] = -2.0*c[X][Y];
	b18[1*NSYMM + XZ] = -2.0*c[X][Z];
	b18[1*NSYMM + YY] = -1.0*c[Y][Y];
	b18[1*NSYMM + YZ] = -2.0*c[Y][Z];
	b18[1*NSYMM + ZZ] = -1.0*c[Z][Z];
	
	ig = normal[1]/2;
	ih = normal[1]%2;
	for (n1 = 0; n1 < NSYMM; n1++) {
	  gradn[n1][ig][ih] = gradn[n1][ig][1 - ih];
	}
	
      }
      
      if (nunknown > 2) {
	
	//TODO NEED TO PORT
	q_boundary_constants(ic, jc, kc, qs, bcs[normal[2]],
			     status[normal[2]], c, pbpc, cinfo);
	
	b18[2*NSYMM + XX] = -1.0*c[X][X];
	b18[2*NSYMM + XY] = -2.0*c[X][Y];
	b18[2*NSYMM + XZ] = -2.0*c[X][Z];
	b18[2*NSYMM + YY] = -1.0*c[Y][Y];
	b18[2*NSYMM + YZ] = -2.0*c[Y][Z];
	b18[2*NSYMM + ZZ] = -1.0*c[Z][Z];
	
	ig = normal[2]/2;
	ih = normal[2]%2;
	for (n1 = 0; n1 < NSYMM; n1++) {
	  gradn[n1][ig][ih] = gradn[n1][ig][1 - ih];
	  }
      }

      
      if (nunknown == 1) {
	
	/* Special case A matrix is diagonal. */
	/* Subtract all three gradient terms from the RHS and then cancel
	 * the one unknown contribution ... works for any normal[0] */
	
	gradient_bcs6x6_coeff(pbpc->kappa0, pbpc->kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    for (ia = 0; ia < 3; ia++) {
	      dq = 0.5*(gradn[n2][ia][0] + gradn[n2][ia][1]);
	      b18[n1] -= bc[n1][n2][ia]*dq;
	    }
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[n1] += bc[n1][n2][normal[0]/2]*dq;
	  }
	  
	  b18[n1] *= bcsign[normal[0]];
	  x18[n1] = tc_a6inv[normal[0]/2][n1]*b18[n1];
	}
      }
      
      if (nunknown == 2) {
	
	if (normal[0]/2 == X && normal[1]/2 == Y) normal[2] = Z;
	if (normal[0]/2 == X && normal[1]/2 == Z) normal[2] = Y;
	if (normal[0]/2 == Y && normal[1]/2 == Z) normal[2] = X;
	
	/* Compute the RHS for two unknowns and one known */
	
	gradient_bcs6x6_coeff(pbpc->kappa0, pbpc->kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]][0] + gradn[n2][normal[2]][1]);
	    b18[n1] -= bc[n1][n2][normal[2]]*dq;
	    
	  }
	}
	
	gradient_bcs6x6_coeff(pbpc->kappa0, pbpc->kappa1, bcs[normal[1]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]][0] + gradn[n2][normal[2]][1]);
	    b18[NSYMM + n1] -= bc[n1][n2][normal[2]]*dq;
	    
	  }
	}
	
	/* Solve x = A^-1 b depending on unknown conbination */
	/* XY => ia = 0 XZ => ia = 1 YZ => ia = 2 ... */
	
	ia = normal[0]/2 + normal[1]/2 - 1;
	//assert(ia == 0 || ia == 1 || ia == 2);
	
	for (n1 = 0; n1 < 2*NSYMM; n1++) {
	  x18[n1] = 0.0;
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    x18[n1] += bcsign[normal[0]]*tc_a12inv[ia][n1][n2]*b18[n2];
	  }
	  for (n2 = NSYMM; n2 < 2*NSYMM; n2++) {
	    x18[n1] += bcsign[normal[1]]*tc_a12inv[ia][n1][n2]*b18[n2];
	  }
	}
      }
      
      if (nunknown == 3) {
	
	gradient_bcs6x6_coeff(pbpc->kappa0, pbpc->kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
	  }
	  b18[n1] *= bcsign[normal[0]];
	}
	
	gradient_bcs6x6_coeff(pbpc->kappa0, pbpc->kappa1, bcs[normal[1]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
	  }
	  b18[NSYMM + n1] *= bcsign[normal[1]];
	}
	
	gradient_bcs6x6_coeff(pbpc->kappa0, pbpc->kappa1, bcs[normal[2]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[2*NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[2*NSYMM + n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	  }
	  b18[2*NSYMM + n1] *= bcsign[normal[2]];
	}
	
	/* Solve x = A^-1 b */
	
	for (n1 = 0; n1 < 3*NSYMM; n1++) {
	  x18[n1] = 0.0;
	  for (n2 = 0; n2 < 3*NSYMM; n2++) {
	    x18[n1] += tc_a18inv[n1][n2]*b18[n2];
	  }
	}
      }
      
      /* Fix the trace (don't care about Qzz in the end) */
      
      for (n = 0; n < nunknown; n++) {
	
	tr = r3*(x18[NSYMM*n + XX] + x18[NSYMM*n + YY] + x18[NSYMM*n + ZZ]);
	x18[NSYMM*n + XX] -= tr;
	x18[NSYMM*n + YY] -= tr;
	
	/* Store missing half gradients */
	
	for (n1 = 0; n1 < NQAB; n1++) {
	  gradn[n1][normal[n]/2][normal[n] % 2] = x18[NSYMM*n + n1];
	}
      }
      
      /* The final answer is the sum of partial gradients */
      
      for (n1 = 0; n1 < NQAB; n1++) {
	del2[NQAB*index + n1] = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  grad[3*(NQAB*index + n1) + ia] =
	    0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	  del2[NQAB*index + n1] += gradn[n1][ia][0] - gradn[n1][ia][1];
	}
      }


      }

    }
 }
 
 
  return;
}


static int gradient_6x6_gpu(const double * field, double * grad,
			    double * del2, const int nextra) {

  int nlocal[3];
  int ia, n1, n2;
  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};


  double bc[6][6][3];

  double kappa0;
  double kappa1;
  double a6inv[3][6];
  double ** a12inv[3];
  double ** a18inv;

  assert(field);

  coords_nlocal(nlocal);

  kappa0 = blue_phase_kappa0();
  kappa1 = blue_phase_kappa1();

  /* Compute inverse matrices */

  util_matrix_create(12, 12, &(a12inv[0]));
  util_matrix_create(12, 12, &(a12inv[1]));
  util_matrix_create(12, 12, &(a12inv[2]));
  util_matrix_create(18, 18, &a18inv);

  for (ia = 0; ia < 3; ia++) {
    gradient_bcs6x6_coeff(kappa0, kappa1, bcs[2*ia + 1], bc); /* +ve sign */
    for (n1 = 0; n1 < NSYMM; n1++) {
      a6inv[ia][n1] = 1.0/bc[n1][n1][ia];
    }

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {
	a18inv[ia*NSYMM + n1][0*NSYMM + n2] = 0.5*(1+d_[ia][X])*bc[n1][n2][X];
	a18inv[ia*NSYMM + n1][1*NSYMM + n2] = 0.5*(1+d_[ia][Y])*bc[n1][n2][Y];
	a18inv[ia*NSYMM + n1][2*NSYMM + n2] = 0.5*(1+d_[ia][Z])*bc[n1][n2][Z];

      }
    }
  }

  for (n1 = 0; n1 < 12; n1++) {
    for (n2 = 0; n2 < 12; n2++) {
      a12inv[0][n1][n2] = a18inv[n1][n2];
      a12inv[2][n1][n2] = a18inv[6+n1][6+n2];
    }
  }

  for (n1 = 0; n1 < 6; n1++) {
    for (n2 = 0; n2 < 6; n2++) {
      a12inv[1][n1][n2] = a18inv[n1][n2];
      a12inv[1][n1][6+n2] = a18inv[n1][12+n2];
    }
  }

  for (n1 = 6; n1 < 12; n1++) {
    for (n2 = 0; n2 < 6; n2++) {
      a12inv[1][n1][n2] = a18inv[6+n1][n2];
      a12inv[1][n1][6+n2] = a18inv[6+n1][12+n2];
    }
  }

  ia = util_matrix_invert(12, a12inv[0]);
  assert(ia == 0);
  ia = util_matrix_invert(12, a12inv[1]);
  assert(ia == 0);
  ia = util_matrix_invert(12, a12inv[2]);
  assert(ia == 0);
  ia = util_matrix_invert(18, a18inv);
  assert(ia == 0);

  int nhalo=coords_nhalo();

  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;
  int nSites=Nall[X]*Nall[Y]*Nall[Z];
  
  int noffset[3];
  coords_nlocal_offset(noffset);



  //copy lattice shape constants to target ahead of execution
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));
  copyConstToTarget(&tc_nextra,&nextra, sizeof(int));
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int));
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int));
  copyConstToTarget(tc_noffset,noffset, 3*sizeof(int));

  //contiguos memory
  double a18invtmp[18][18];
  double a12invtmp[3][12][12];

  int i, j, k;
  for(i=0;i<18;i++)
    for(j=0;j<18;j++)
      a18invtmp[i][j]=a18inv[i][j];


  
  for(i=0;i<3;i++)
    for(j=0;j<12;j++)
      for(k=0;k<12;k++)
      a12invtmp[i][j][k]=a12inv[i][j][k];

  copyConstToTarget(tc_a18inv,a18invtmp, 18*18*sizeof(double));
  copyConstToTarget(tc_a12inv,a12invtmp, 3*12*12*sizeof(double));

  copyConstToTarget(tc_a6inv,a6inv, 3*6*sizeof(double));

  // initialise kernel constants on both host and target
  blue_phase_set_kernel_constants();

  // get a pointer to target copy of stucture containing kernel constants
  void* pcon=NULL;
  blue_phase_target_constant_ptr(&pcon);

  //map_t* t_map = map_->tcopy; //target copy of map structure

  double* tmpptr;
  //copyFromTarget(&tmpptr,&(t_map->status),sizeof(char*)); 
  //copyToTarget(tmpptr,map_->status,nSites*sizeof(char));


  // set up colloids such that they can be accessed from target
  // noting that each actual colloid structure stays resident on the host

  colloids_info_t* cinfo=colloids_q_cinfo();  


  // if (cinfo->map_new){
  //colloids_info_t* t_cinfo=cinfo->tcopy; //target copy of colloids_info structure     
  //colloid_t* tmpcol;
  //copyFromTarget(&tmpcol,&(t_cinfo->map_new),sizeof(colloid_t**)); 
  //copyToTarget(tmpcol,cinfo->map_new,nSites*sizeof(colloid_t*));
  //}


  //execute lattice-based operation on target
  
  gradient_6x6_gpu_lattice __targetLaunch__(nSites) (field, grad,
  						     del2, map_->tcopy,
						     (bluePhaseKernelConstants_t*) pcon, 
						     cinfo->tcopy);
  targetSynchronize();
  

  util_matrix_free(18, &a18inv);
  util_matrix_free(12, &(a12inv[2]));
  util_matrix_free(12, &(a12inv[1]));
  util_matrix_free(12, &(a12inv[0]));

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

__targetHost__ __target__ int gradient_bcs6x5_coeff(double kappa0, double kappa1, const int dn[3],
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

__targetHost__ __target__ int gradient_bcs6x6_coeff(double kappa0, double kappa1, const int dn[3],
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

__targetHost__ __target__ static void util_q5_to_qab(double q[3][3], const double * phi) {

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
