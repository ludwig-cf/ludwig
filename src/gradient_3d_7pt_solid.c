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
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "wall.h"
#include "coords.h"
#include "gradient.h"
#include "site_map.h"
#include "free_energy.h"
#include "colloids_Q_tensor.h"
#include "gradient_3d_7pt_solid.h"

/* Only tensor order parameter relevant */

#define NOP 5
double blue_phase_q0(void);

#define NGRAD_BC 6
static const int bc_[NGRAD_BC][3] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0},
				     {0,0,1}, {0,0,-1}};

double fed_surface(double q[3][3], double qs[5], double w);
void fed_q5_to_qab(double q[3][3], const double * phi);
void gradient_3d_7pt_solid_fe_s(double * fstats);


static void gradient_3d_7pt_solid_try2_wall(const int nop,
					    const double * field,
					    double * grad,
					    double * del2, const int nextra);

static void gradient_3d_7pt_solid_try2_fluid(const int nop,
					     const double * field,
					     double * grad,
					     double * del2,
					     const int nextra);

static void gradient_3d_7pt_solid_wall_y(const int nop,
					 const double * field,
					 double * grad,
					 double * del2,
					 const int nextra);
static void gradient_3d_7pt_solid_wall_z(const int nop,
					 const double * field,
					 double * grad,
					 double * del2,
					 const int nextra);

static void gradient_colloid(const int nop, const double * field,
			     double * grad, double * del2, int nextra);

static void gradient_norm1(const int index, const int norm1,
			   const int nhat1[3], const double dn1[3],
			   const double * field, double * grad,
			   double * del2);

static void gradient_norm2(const int index, const int norm1, const int norm2,
			   const int nhat1[3], const double dn1[3],
			   const int nhat2[3], const double dn2[3],
			   const double * field, double * grad,
			   double * del2);

static void gradient_norm3(const int index, const int nhatx[3],
			   const double dnx[3], const int nhaty[3],
			   const double dny[3], const int nhatz[3],
			   const double dnz[3], const double * field,
			   double * grad, double * del2);

static void gradient_bcs(double kappa0, double kappa1, const double dn[3],
			 double dq[NOP][3], double bc[NOP][NOP][3]);

int util_solve_linear_system(const int n, double (*a)[], double * xb);

static double fe_wall_[2];

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
 *****************************************************************************/

void gradient_3d_7pt_solid_d2(const int nop, const double * field,
			      double * grad, double * delsq) {
  int nextra;

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  fe_wall_[0] = 0.0;
  fe_wall_[1] = 0.0;

  gradient_3d_7pt_solid_try2_fluid(nop, field, grad, delsq, nextra);

  if (wall_at_edge(X)) {
    gradient_3d_7pt_solid_try2_wall(nop, field, grad, delsq, nextra);
  }

  if (wall_at_edge(Y)) {
    gradient_3d_7pt_solid_wall_y(nop, field, grad, delsq, nextra);
  }

  if (wall_at_edge(Z)) {
    gradient_3d_7pt_solid_wall_z(nop, field, grad, delsq, nextra);
  }

  /* Always at the moment. Might want to avoid if only flat walls */
  gradient_colloid(nop, field, grad, delsq, nextra);

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_fe_s
 *
 *****************************************************************************/

void gradient_3d_7pt_solid_fe_s(double * fstats) {

  fstats[0] = fe_wall_[0];
  fstats[1] = fe_wall_[1];

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_hs
 *
 *  TODO:
 *  This is potentially using updated values of phi.
 *
 *****************************************************************************/

void gradient_3d_7pt_solid_hs(int ic, int jc, int kc, double hs[3][3]) {

  int ia, ib, n, p;
  int index, index1;
  char status;

  double w;
  double gradf;
  double qs[5];
  double qsab[3][3], q0[3][3];
  double dn[3];
  extern double * phi_site;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      hs[ia][ib] = 0.0;
    }
  }

  index = coords_index(ic, jc, kc);
  w = colloids_q_tensor_w();

  for (p = 0; p < NGRAD_BC; p++) {

    index1 = coords_index(ic + bc_[p][X], jc + bc_[p][Y], kc + bc_[p][Z]);
    status = site_map_get_status_index(index1);
  
    if (status == BOUNDARY) {

      /* Use fluid point in the opposite direction to solid to construct
       * gradient and extrapolate to solid surface */

      index1 = coords_index(ic - bc_[p][X], jc - bc_[p][Y], kc - bc_[p][Z]);

      for (n = 0; n < NOP; n++) {
	gradf = phi_site[NOP*index1 + n] - phi_site[NOP*index + n];
	qs[n] = phi_site[NOP*index + n] - 0.5*gradf;
      }

      for (ia = 0; ia < 3; ia++) {
	dn[ia] = 1.0*bc_[p][ia];
      }

      fed_q5_to_qab(qsab, qs);
      colloids_q_boundary(dn, qsab, q0);

      for (ia = 0; ia < 3; ia++) {
	for (ib = 0 ; ib < 3; ib++) {
	  hs[ia][ib] = -w*(qsab[ia][ib] - q0[ia][ib]);
	}
      }

      if (status == COLLOID) {
	fatal("NO colloid hs yet\n");
      }

    }
  }

  return;
}

void fed_q5_to_qab(double q[3][3], const double * phi) {

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

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_try2_fluid
 *
 *****************************************************************************/

static void gradient_3d_7pt_solid_try2_fluid(const int nop,
					     const double * field,
					     double * grad,
					     double * del2,
					     const int nextra) {
  int nlocal[3];
  int nhalo;
  int n;
  int ic, jc, kc;
  int xs, ys;
  int index;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  ys = nlocal[Z] + 2*nhalo;
  xs = ys*(nlocal[Y] + 2*nhalo);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + X]
	    = 0.5*(field[nop*(index + xs) + n] - field[nop*(index - xs) + n]);
	  grad[3*(nop*index + n) + Y]
	    = 0.5*(field[nop*(index + ys) + n] - field[nop*(index - ys) + n]);
	  grad[3*(nop*index + n) + Z]
	    = 0.5*(field[nop*(index + 1) + n] - field[nop*(index - 1) + n]);
	  del2[nop*index + n]
	    = field[nop*(index + xs) + n] + field[nop*(index - xs) + n]
	    + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	    + field[nop*(index + 1)  + n] + field[nop*(index - 1)  + n]
	    - 6.0*field[nop*index + n];
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_try2_wall
 *
 *****************************************************************************/

static void gradient_3d_7pt_solid_try2_wall(const int nop,
					    const double * field,
					    double * grad,
					    double * del2,
					    const int nextra) {
  int jc, kc;
  int index;
  int nlocal[3];
  int nhat[3];

  double dn[3];

  assert(wall_at_edge(Y) == 0);
  assert(wall_at_edge(Z) == 0);

  coords_nlocal(nlocal);

  if (cart_coords(X) == 0) {

    /* Correct the lower wall */

    nhat[X] = +1;
    nhat[Y] = 0;
    nhat[Z] = 0;
    dn[X] = +1.0;
    dn[Y] = 0.0;
    dn[Z] = 0.0;

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(1, jc, kc);
	gradient_norm1(index, X, nhat, dn, field, grad, del2);
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {

    /* Correct the upper wall */

    nhat[X] = -1;
    nhat[Y] = 0;
    nhat[Z] = 0;
    dn[X] = -1.0;
    dn[Y] = 0.0;
    dn[Z] = 0.0;

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(nlocal[X], jc, kc);
	gradient_norm1(index, X, nhat, dn, field, grad, del2);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_colloid
 *
 *****************************************************************************/

static void gradient_colloid(const int nop, const double * field,
			     double * grad, double * del2, int nextra) {
  int nlocal[3];
  int ic, jc, kc;
  int index, n;

  int norm[3];
  int nhat[3][3];
  double dn[3][3];

  coords_nlocal(nlocal);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);

	for (n = 0; n < 3; n++) {
	  nhat[n][X] = 0;
	  nhat[n][Y] = 0;
	  nhat[n][Z] = 0;
	}

	n = 0;
	if (site_map_get_status(ic+1, jc, kc) == COLLOID) {
	  norm[n] = X;
	  nhat[n][X] = -1;
	  n++;
	}

	if (site_map_get_status(ic-1, jc, kc) == COLLOID) {
	  assert(n == 0);
	  norm[n] = X;
	  nhat[n][X] = +1;
	  n++;
	}

	if (site_map_get_status(ic, jc+1, kc) == COLLOID) {
	  assert(n <= 1);
	  norm[n] = Y;
	  nhat[n][Y] = -1;
	  n++;
	}

	if (site_map_get_status(ic, jc-1, kc) == COLLOID) {
	  assert(n <= 1);
	  norm[n] = Y;
	  nhat[n][Y] = 1;
	  n++;
	}

	if (site_map_get_status(ic, jc, kc+1) == COLLOID) {
	  assert(n <= 2);
	  norm[n] = Z;
	  nhat[n][Z] = -1;
	  n++;
	}

	if (site_map_get_status(ic, jc, kc-1) == COLLOID) {
	  assert(n <= 2);
	  norm[n] = Z;
	  nhat[n][Z] = 1;
	  n++;
	}

	assert(n <= 3);

	if (n == 1) {
	  colloids_q_boundary_normal(index, nhat[0], dn[0]);
	  gradient_norm1(index, norm[0], nhat[0], dn[0], field, grad, del2);
	}

	if (n == 2) {
	  colloids_q_boundary_normal(index, nhat[0], dn[0]);
	  colloids_q_boundary_normal(index, nhat[1], dn[1]);
	  gradient_norm2(index, norm[0], norm[1], nhat[0], dn[0], nhat[1],
			 dn[1], field, grad, del2);
	}

	if (n == 3) {
	  colloids_q_boundary_normal(index, nhat[0], dn[0]);
	  colloids_q_boundary_normal(index, nhat[1], dn[1]);
	  colloids_q_boundary_normal(index, nhat[2], dn[2]);
	  gradient_norm3(index, nhat[0], dn[0], nhat[1], dn[1], nhat[2], dn[2],
			 field, grad, del2);
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_norm1
 *
 *  Computes the cholesteric boundary condition for fluid sites with
 *  one normal solid neighbour.
 *
 *  The fluid site is at index;
 *  The direction of the neighbour (hence required gradient) is
 *      norm1 [ = X | Y | Z ];
 *  The integer outward normal vector is nhat1 (a lattice vector)
 *  The 'true' normal (floating point) at the half way position is
 *  dn1 (not necessarily == nhat1).
 *
 *****************************************************************************/

static void gradient_norm1(const int index, const int norm1,
			   const int nhat1[3], const double dn1[3],
			   const double * field, double * grad,
			   double * del2) {

  int ia, ib, ic, ig, n1, n2;
  int nhalo;
  int nlocal[3];
  int str[3];            /* Memory strides in each direction */
  int nsolid1;            /* Points to solid partial gradient */
  int nfluid1;            /* Points to fluid partial gradient */

  double gradn[NOP][3][2];  /* gradn[n1][ia][0] is forward gradient for n1 in
			     * direction ia , Q_{ic+1} - Q_ic etc,
			     * grad[n][ia] [1] backward gradient,
			     * Q_ic - Q_{ic-1} etc */ 
  double dq[NOP][3];        /* Gradients of the order parameter from fluid */
 
  double qs1[NOP];
  double qs1ab[3][3];
  double q01[3][3];

  double bc1[NOP][NOP][3];  /* All gradient terms in boundary conditon */
  double a[NOP][NOP];       /* Linear algebra Ax = b */
  double b[NOP];            /* b is the rhs/solution vector */
  double c1[3][3];          /* Constant terms in boundary condition */

  double kappa0, kappa1, q_0, w;

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

  nfluid1 = (1 - nhat1[norm1])/2;
  nsolid1 = (1 + nhat1[norm1])/2;

  assert(nfluid1 == 0 || nfluid1 == 1);
  assert(nsolid1 == 0 || nsolid1 == 1);
  assert(nfluid1 != nsolid1);

  for (n1 = 0; n1 < NOP; n1++) {
    for (ia = 0; ia < 3; ia++) {
      gradn[n1][ia][0] =
	field[NOP*(index + str[ia]) + n1] - field[NOP*index + n1];
      gradn[n1][ia][1] =
	field[NOP*index + n1] - field[NOP*(index - str[ia]) + n1];
    }
    qs1[n1] = field[NOP*index+n1] - 0.5*nhat1[norm1]*gradn[n1][norm1][nfluid1];
  }

  fed_q5_to_qab(qs1ab, qs1);
  colloids_q_boundary(dn1, qs1ab, q01);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      fe_wall_[nsolid1] +=
	0.5*w*(qs1ab[ia][ib] - q01[ia][ib])*(qs1ab[ia][ib] - q01[ia][ib]);
    }
  }

  /* Derivatives of Q_ab precomputed for the fluid; set unknowns to unity
   * as these multiply the coefficients in the respective terms in the
   * linear algebra problem to follow. */

  for (n1 = 0; n1 < NOP; n1++) {
    for (ia = 0; ia < 3; ia++) {
      dq[n1][ia] = grad[3*(NOP*index + n1) + ia];
    }
    dq[n1][norm1] = 1.0;
  }

  /* constant terms k1*q0*(e_agc Q_cb + e_bgc Q_Ca) n_g + w*(qs - q0)_ab */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      c1[ia][ib] = 0.0;
      for (ig = 0; ig < 3; ig++) {
	for (ic = 0; ic < 3; ic++) {
	  c1[ia][ib] += kappa1*q_0*dn1[ig]*
	    (e_[ia][ig][ic]*qs1ab[ic][ib] + e_[ib][ig][ic]*qs1ab[ic][ia]);
	}
      }
      c1[ia][ib] += w*(qs1ab[ia][ib] - q01[ia][ib]);
    }
  }

  /* Construct the linear algebra problem matrixA vector b */

  gradient_bcs(kappa0, kappa1, dn1, dq, bc1);

  for (n1 = 0; n1 < NOP; n1++) {
    b[n1] = 0.0;
    for (n2 = 0; n2 < NOP; n2++) {
      a[n1][n2] = bc1[n1][n2][norm1];
      b[n1] -= bc1[n1][n2][norm1];
      for (ia = 0; ia < 3; ia++) {
	b[n1] += bc1[n1][n2][ia];
      }
    }
  }

  b[XX] = -(b[XX] + c1[X][X]);
  b[XY] = -(b[XY] + 2.0*c1[X][Y]);
  b[XZ] = -(b[XZ] + 2.0*c1[X][Z]);
  b[YY] = -(b[YY] + c1[Y][Y]);
  b[YZ] = -(b[YZ] + 2.0*c1[Y][Z]);

  /* SOLVE LINEAR SYSTEM b <= A^{-1} b */

  util_solve_linear_system(5, a, b);

  /* This result for the solid partial gradient always has the wrong
   * sign for the final gradient calculation. */

  gradn[XX][norm1][nsolid1] = -b[XX];
  gradn[XY][norm1][nsolid1] = -b[XY];
  gradn[XZ][norm1][nsolid1] = -b[XZ];
  gradn[YY][norm1][nsolid1] = -b[YY];
  gradn[YZ][norm1][nsolid1] = -b[YZ];

  for (n1 = 0; n1 < NOP; n1++) {
    grad[3*(NOP*index + n1) + norm1] =
      0.5*(gradn[n1][norm1][0] + gradn[n1][norm1][1]);
    del2[NOP*index + n1] = gradn[n1][X][0] - gradn[n1][X][1]
      + gradn[n1][Y][0] - gradn[n1][Y][1]
      + gradn[n1][Z][0] - gradn[n1][Z][1];
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_wall_y
 *
 *****************************************************************************/

static void gradient_3d_7pt_solid_wall_y(const int nop,
					 const double * field,
					 double * grad,
					 double * del2,
					 const int nextra) {
  int nlocal[3];
  int ic, kc;
  int index;
  int nhat[3];

  double dn[3];

  assert(wall_at_edge(X) == 0);
  assert(wall_at_edge(Z) == 0);

  coords_nlocal(nlocal);

  for (index = 0; index < 3; index++) {
    nhat[index] = 0;
    dn[index] = 0.0;
  }

  if (cart_coords(Y) == 0) {

    nhat[Y] = +1;
    dn[Y] = +1.0;

    for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, 1, kc);
	gradient_norm1(index, Y, nhat, dn, field, grad, del2);
      }
    }
  }

  if (cart_coords(Y) == cart_size(Y) - 1) {

    nhat[Y] = -1;
    dn[Y] = -1.0;

    for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, nlocal[Y], kc);
	gradient_norm1(index, Y, nhat, dn, field, grad, del2);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_wall_z
 *
 *****************************************************************************/

static void gradient_3d_7pt_solid_wall_z(const int nop,
					 const double * field,
					 double * grad,
					 double * del2,
					 const int nextra) {
  int nlocal[3];
  int ia, ic, jc;
  int index;
  int nhat[3];

  double dn[3];

  coords_nlocal(nlocal);

  assert(wall_at_edge(X) == 0);
  assert(wall_at_edge(Y) == 0);

  for (ia = 0; ia < 3; ia++) {
    nhat[ia] = 0;
    dn[ia] = 0.0;
  }

  if (cart_coords(Z) == 0) {

    nhat[Z] = +1;
    dn[Z] = +1.0;

    for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

	index = coords_index(ic, jc, 1);
	gradient_norm1(index, Z, nhat, dn, field, grad, del2);
      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {

    nhat[Z] = -1;
    dn[Z] = -1.0;

    for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

	index = coords_index(ic, jc, nlocal[Z]);
	gradient_norm1(index, Z, nhat, dn, field, grad, del2);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_norm2
 *
 *****************************************************************************/

static void gradient_norm2(const int index, const int norm1, const int norm2,
			   const int nhat1[3], const double dn1[3],
			   const int nhat2[3], const double dn2[3],
			   const double * field, double * grad,
			   double * del2) {

  int ia, ib, ic, ig, n1, n2;
  int nhalo;
  int nlocal[3];
  int str[3];             /* Memory strides */
  int nsolid1, nsolid2;   /* Point to solid partial gradients */
  int nfluid1, nfluid2;   /* Point to fluid partial gradients */


  double gradn[NOP][3][2];  /* gradn[n1][ia][0] is forward gradient for n1 in
			     * direction ia , Q_{ic+1} - Q_ic etc,
			     * grad[n][ia] [1] backward gradient,
			     * Q_ic - Q_{ic-1} etc */ 
  double dq[NOP][3];        /* Gradients of the order parameter from fluid */
 
  double qs1[NOP], qs2[NOP];
  double qs1ab[3][3], qs2ab[3][3];
  double q01[3][3], q02[3][3];

  double bc1[NOP][NOP][3];   /* All gradient terms in boundary conditon */
  double a[2*NOP][2*NOP];    /* Linear algebra Ax = b */
  double b[2*NOP];           /* b is the rhs/solution vector */
  double c1[3][3];           /* Constant terms in boundary condition */
  double c2[3][3];

  double kappa0, kappa1, q_0, w;

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

  nfluid1 = (1 - nhat1[norm1])/2;
  nsolid1 = (1 + nhat1[norm1])/2;

  nfluid2 = (1 - nhat2[norm2])/2;
  nsolid2 = (1 + nhat2[norm2])/2;

  assert(nfluid1 == 0 || nfluid1 == 1);
  assert(nsolid1 == 0 || nsolid1 == 1);
  assert(nfluid1 != nsolid1);

  assert(nfluid2 == 0 || nfluid2 == 1);
  assert(nsolid2 == 0 || nsolid2 == 1);
  assert(nfluid2 != nsolid2);

  for (n1 = 0; n1 < NOP; n1++) {
    for (ia = 0; ia < 3; ia++) {
      gradn[n1][ia][0] =
	field[NOP*(index + str[ia]) + n1] - field[NOP*index + n1];
      gradn[n1][ia][1] =
	field[NOP*index + n1] - field[NOP*(index - str[ia]) + n1];
    }
    qs1[n1] = field[NOP*index+n1] - 0.5*nhat1[norm1]*gradn[n1][norm1][nfluid1];
    qs2[n1] = field[NOP*index+n1] - 0.5*nhat2[norm2]*gradn[n1][norm2][nfluid2];
  }

  fed_q5_to_qab(qs1ab, qs1);
  colloids_q_boundary(dn1, qs1ab, q01);
  fed_q5_to_qab(qs2ab, qs2);
  colloids_q_boundary(dn2, qs2ab, q02);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      fe_wall_[nsolid1] +=
	0.5*w*(qs1ab[ia][ib] - q01[ia][ib])*(qs1ab[ia][ib] - q01[ia][ib])
	+ 0.5*w*(qs2ab[ia][ib] - q02[ia][ib])*(qs2ab[ia][ib] - q02[ia][ib]);
    }
  }

  /* Derivatives of Q_ab precomputed for the fluid; set unknowns to unity
   * as these multiply the coefficients in the respective terms in the
   * linear algebra problem to follow. */

  for (n1 = 0; n1 < NOP; n1++) {
    for (ia = 0; ia < 3; ia++) {
      dq[n1][ia] = grad[3*(NOP*index + n1) + ia];
    }
    dq[n1][norm1] = 1.0;
    dq[n1][norm2] = 1.0;
  }

  /* constant terms k1*q0*(e_agc Q_cb + e_bgc Q_Ca) n_g + w*(qs - q0)_ab */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      c1[ia][ib] = 0.0;
      c2[ia][ib] = 0.0;
      for (ig = 0; ig < 3; ig++) {
	for (ic = 0; ic < 3; ic++) {
	  c1[ia][ib] += kappa1*q_0*dn1[ig]*
	    (e_[ia][ig][ic]*qs1ab[ic][ib] + e_[ib][ig][ic]*qs1ab[ic][ia]);
	  c2[ia][ib] += kappa1*q_0*dn2[ig]*
	    (e_[ia][ig][ic]*qs2ab[ic][ib] + e_[ib][ig][ic]*qs2ab[ic][ia]);
	}
      }
      c1[ia][ib] += w*(qs1ab[ia][ib] - q01[ia][ib]);
      c2[ia][ib] += w*(qs2ab[ia][ib] - q02[ia][ib]);
    }
  }

  /* Construct boundary condition terms: first normal */

  gradient_bcs(kappa0, kappa1, dn1, dq, bc1);

  for (n1 = 0; n1 < NOP; n1++) {
    b[n1] = 0.0;
    for (n2 = 0; n2 < NOP; n2++) {
      a[n1][n2] = bc1[n1][n2][norm1];
      a[n1][NOP + n2] = bc1[n1][n2][norm2];
      b[n1] -= bc1[n1][n2][norm1];
      b[n1] -= bc1[n1][n2][norm2];
      for (ia = 0; ia < 3; ia++) {
	b[n1] += bc1[n1][n2][ia];
      }
    }
  }

  b[XX] = -(b[XX] + c1[X][X]);
  b[XY] = -(b[XY] + 2.0*c1[X][Y]);
  b[XZ] = -(b[XZ] + 2.0*c1[X][Z]);
  b[YY] = -(b[YY] + c1[Y][Y]);
  b[YZ] = -(b[YZ] + 2.0*c1[Y][Z]);

  /* Second normal */

  gradient_bcs(kappa0, kappa1, dn2, dq, bc1);

  for (n1 = 0; n1 < NOP; n1++) {
    b[n1] = 0.0;
    for (n2 = 0; n2 < NOP; n2++) {
      a[NOP + n1][n2] = bc1[n1][n2][norm1];
      a[NOP + n1][NOP + n2] = bc1[n1][n2][norm2];
      b[NOP + n1] -= bc1[n1][n2][norm1];
      b[NOP + n1] -= bc1[n1][n2][norm2];
      for (ia = 0; ia < 3; ia++) {
	b[NOP + n1] += bc1[n1][n2][ia];
      }
    }
  }

  b[NOP + XX] = -(b[NOP + XX] + c2[X][X]);
  b[NOP + XY] = -(b[NOP + XY] + 2.0*c2[X][Y]);
  b[NOP + XZ] = -(b[NOP + XZ] + 2.0*c2[X][Z]);
  b[NOP + YY] = -(b[NOP + YY] + c2[Y][Y]);
  b[NOP + YZ] = -(b[NOP + YZ] + 2.0*c2[Y][Z]);

  /* SOLVE LINEAR SYSTEM b <= A^{-1} b */

  util_solve_linear_system(2*NOP, a, b);

  /* This result for the solid partial gradients always has the wrong
   * sign for the final gradient calculation. */

  gradn[XX][norm1][nsolid1] = -b[XX];
  gradn[XY][norm1][nsolid1] = -b[XY];
  gradn[XZ][norm1][nsolid1] = -b[XZ];
  gradn[YY][norm1][nsolid1] = -b[YY];
  gradn[YZ][norm1][nsolid1] = -b[YZ];

  gradn[XX][norm2][nsolid2] = -b[NOP + XX];
  gradn[XY][norm2][nsolid2] = -b[NOP + XY];
  gradn[XZ][norm2][nsolid2] = -b[NOP + XZ];
  gradn[YY][norm2][nsolid2] = -b[NOP + YY];
  gradn[YZ][norm2][nsolid2] = -b[NOP + YZ];

  for (n1 = 0; n1 < NOP; n1++) {
    grad[3*(NOP*index + n1) + norm1] =
      0.5*(gradn[n1][norm1][0] + gradn[n1][norm1][1]);
    grad[3*(NOP*index + n1) + norm2] =
      0.5*(gradn[n1][norm2][0] + gradn[n1][norm2][1]);
    del2[NOP*index + n1] = gradn[n1][X][0] - gradn[n1][X][1]
      + gradn[n1][Y][0] - gradn[n1][Y][1]
      + gradn[n1][Z][0] - gradn[n1][Z][1];
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_bcs
 *
 *****************************************************************************/

static void gradient_bcs(double kappa0, double kappa1, const double dn[3],
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
 *  gradient_norm3
 *
 *****************************************************************************/

static void gradient_norm3(const int index, const int nhatx[3],
			   const double dnx[3], const int nhaty[3],
			   const double dny[3], const int nhatz[3],
			   const double dnz[3], const double * field,
			   double * grad, double * del2) {

  int ia, ib, ic, ig, n1, n2;
  int nhalo;
  int nlocal[3];
  int str[3];
  int nsolid[3];            /* Point to solid partial gradients */
  int nfluid[3];            /* Point to fluid partial gradients */

  double gradn[NOP][3][2];  /* gradn[n1][ia][0] is forward gradient for n1 in
			     * direction ia , Q_{ic+1} - Q_ic etc,
			     * grad[n][ia] [1] backward gradient,
			     * Q_ic - Q_{ic-1} etc */ 
  double dq[NOP][3];        /* Gradients of the order parameter from fluid */
 
  double qs1[NOP], qs2[NOP], qs3[NOP];
  double qs1ab[3][3], qs2ab[3][3], qs3ab[3][3];
  double q01[3][3], q02[3][3], q03[3][3];

  double bc1[NOP][NOP][3];   /* All gradient terms in boundary conditon */
  double a[3*NOP][3*NOP];    /* Linear algebra Ax = b */
  double b[3*NOP];           /* b is the rhs/solution vector */
  double c1[3][3];           /* Constant terms in boundary condition */
  double c2[3][3];
  double c3[3][3];

  double kappa0, kappa1, q_0, w;

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

  nfluid[X] = (1 - nhatx[X])/2;
  nsolid[X] = (1 + nhatx[X])/2;

  nfluid[Y] = (1 - nhaty[Y])/2;
  nsolid[Y] = (1 + nhatx[Y])/2;

  nfluid[Z] = (1 - nhatz[Z])/2;
  nsolid[Z] = (1 + nhatz[Z])/2;

  assert(nfluid[X] == 0 || nfluid[X] == 1);
  assert(nsolid[X] == 0 || nsolid[X] == 1);
  assert(nfluid[X] != nsolid[X]);

  assert(nfluid[Y] == 0 || nfluid[Y] == 1);
  assert(nsolid[Y] == 0 || nsolid[Y] == 1);
  assert(nfluid[Y] != nsolid[Y]);

  assert(nfluid[Z] == 0 || nfluid[Z] == 1);
  assert(nsolid[Z] == 0 || nsolid[Z] == 1);
  assert(nfluid[Z] != nsolid[Z]);

  for (n1 = 0; n1 < NOP; n1++) {
    for (ia = 0; ia < 3; ia++) {
      gradn[n1][ia][0] =
	field[NOP*(index + str[ia]) + n1] - field[NOP*index + n1];
      gradn[n1][ia][1] =
	field[NOP*index + n1] - field[NOP*(index - str[ia]) + n1];
    }
    qs1[n1] = field[NOP*index+n1] - 0.5*nhatx[X]*gradn[n1][X][nfluid[X]];
    qs2[n1] = field[NOP*index+n1] - 0.5*nhaty[Y]*gradn[n1][Y][nfluid[Y]];
    qs3[n1] = field[NOP*index+n1] - 0.5*nhatz[Z]*gradn[n1][Z][nfluid[Z]];
  }

  fed_q5_to_qab(qs1ab, qs1);
  colloids_q_boundary(dnx, qs1ab, q01);
  fed_q5_to_qab(qs2ab, qs2);
  colloids_q_boundary(dny, qs2ab, q02);
  fed_q5_to_qab(qs3ab, qs3);
  colloids_q_boundary(dnz, qs3ab, q03);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      fe_wall_[0] +=
	0.5*w*(qs1ab[ia][ib] - q01[ia][ib])*(qs1ab[ia][ib] - q01[ia][ib])
	+ 0.5*w*(qs2ab[ia][ib] - q02[ia][ib])*(qs2ab[ia][ib] - q02[ia][ib])
	+ 0.5*w*(qs3ab[ia][ib] - q03[ia][ib])*(qs3ab[ia][ib] - q03[ia][ib]);
    }
  }

  /* All the gradient terms are required. */

  for (n1 = 0; n1 < NOP; n1++) {
    for (ia = 0; ia < 3; ia++) {
      dq[n1][ia] = 1.0;
    }
  }

  /* constant terms k1*q0*(e_agc Q_cb + e_bgc Q_Ca) n_g + w*(qs - q0)_ab */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      c1[ia][ib] = 0.0;
      c2[ia][ib] = 0.0;
      c3[ia][ib] = 0.0;
      for (ig = 0; ig < 3; ig++) {
	for (ic = 0; ic < 3; ic++) {
	  c1[ia][ib] += kappa1*q_0*dnx[ig]*
	    (e_[ia][ig][ic]*qs1ab[ic][ib] + e_[ib][ig][ic]*qs1ab[ic][ia]);
	  c2[ia][ib] += kappa1*q_0*dny[ig]*
	    (e_[ia][ig][ic]*qs2ab[ic][ib] + e_[ib][ig][ic]*qs2ab[ic][ia]);
	  c3[ia][ib] += kappa1*q_0*dnz[ig]*
	    (e_[ia][ig][ic]*qs3ab[ic][ib] + e_[ib][ig][ic]*qs3ab[ic][ia]);
	}
      }
      c1[ia][ib] += w*(qs1ab[ia][ib] - q01[ia][ib]);
      c2[ia][ib] += w*(qs2ab[ia][ib] - q02[ia][ib]);
      c3[ia][ib] += w*(qs3ab[ia][ib] - q03[ia][ib]);
    }
  }

  /* Construct boundary condition terms: first normal */

  gradient_bcs(kappa0, kappa1, dnx, dq, bc1);

  for (n1 = 0; n1 < NOP; n1++) {
    for (n2 = 0; n2 < NOP; n2++) {
      a[n1][n2] = bc1[n1][n2][X];
      a[n1][NOP + n2] = bc1[n1][n2][Y];
      a[n1][2*NOP + n2] = bc1[n1][n2][Z];
    }
  }

  b[XX] = -c1[X][X];
  b[XY] = -2.0*c1[X][Y];
  b[XZ] = -2.0*c1[X][Z];
  b[YY] = -c1[Y][Y];
  b[YZ] = -2.0*c1[Y][Z];

  /* Second normal */

  gradient_bcs(kappa0, kappa1, dny, dq, bc1);

  for (n1 = 0; n1 < NOP; n1++) {
    for (n2 = 0; n2 < NOP; n2++) {
      a[NOP + n1][n2] = bc1[n1][n2][X];
      a[NOP + n1][NOP + n2] = bc1[n1][n2][Y];
      a[NOP + n1][2*NOP + n2] = bc1[n1][n2][Z];
    }
  }

  b[NOP + XX] = -c2[X][X];
  b[NOP + XY] = -2.0*c2[X][Y];
  b[NOP + XZ] = -2.0*c2[X][Z];
  b[NOP + YY] = -c2[Y][Y];
  b[NOP + YZ] = -2.0*c2[Y][Z];

  /* Third normal */

  gradient_bcs(kappa0, kappa1, dnz, dq, bc1);

  for (n1 = 0; n1 < NOP; n1++) {
    for (n2 = 0; n2 < NOP; n2++) {
      a[2*NOP + n1][n2] = bc1[n1][n2][X];
      a[2*NOP + n1][NOP + n2] = bc1[n1][n2][Y];
      a[2*NOP + n1][2*NOP + n2] = bc1[n1][n2][Z];
    }
  }

  b[2*NOP + XX] = -c3[X][X];
  b[2*NOP + XY] = -2.0*c3[X][Y];
  b[2*NOP + XZ] = -2.0*c3[X][Z];
  b[2*NOP + YY] = -c3[Y][Y];
  b[2*NOP + YZ] = -2.0*c3[Y][Z];

  /* SOLVE LINEAR SYSTEM b <= A^{-1} b */

  util_solve_linear_system(3*NOP, a, b);

  /* This result for the solid partial gradients always has the wrong
   * sign for the final gradient calculation. */

  for (n1 = 0; n1 < NOP; n1++) {
    for (ia = 0; ia < 3; ia++) {
      gradn[n1][ia][nsolid[ia]] = -b[ia*NOP + n1];
    }
  }

  for (n1 = 0; n1 < NOP; n1++) {
    grad[3*(NOP*index + n1) + X] = 0.5*(gradn[n1][X][0] + gradn[n1][X][1]);
    grad[3*(NOP*index + n1) + Y] = 0.5*(gradn[n1][Y][0] + gradn[n1][Y][1]);
    grad[3*(NOP*index + n1) + Y] = 0.5*(gradn[n1][Z][0] + gradn[n1][Z][1]);

    del2[NOP*index + n1] = gradn[n1][X][0] - gradn[n1][X][1]
      + gradn[n1][Y][0] - gradn[n1][Y][1]
      + gradn[n1][Z][0] - gradn[n1][Z][1];
  }


  return;
}

/*****************************************************************************
 *
 *  util_solve_linear_system
 *
 *  Solve Ax = b for nxn A. xb RHS on entry and solution on exit.
 *  A is destroyed.
 *
 *****************************************************************************/

int util_solve_linear_system(const int n, double (*a)[n], double * xb) {

  int i, j, k;
  int ifail = 0;
  int iprow;
  int * ipivot;

  double tmp;

  ipivot = (int *) malloc(n*sizeof(int));
  if (ipivot == NULL) fatal("malloc(ipivot) failed\n");

  iprow = -1;
  for (k = 0; k < n; k++) {
    ipivot[k] = -1;
  }

  for (k = 0; k < n; k++) {

    /* Find pivot row */
    tmp = 0.0;
    for (i = 0; i < n; i++) {
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
      ifail = -1;
      info("*** IFAIL = -1\n");
      /* Must drop through to deallocate ipivot[] */
      break;
    }

    tmp = 1.0 / a[iprow][k];

    for (j = k; j < n; j++) {
      a[iprow][j] *= tmp;
    }
    xb[iprow] *= tmp;

    /* Subtract the pivot row (scaled) from remaining rows */

    for (i = 0; i < n; i++) {
      if (ipivot[i] == -1) {
	tmp = a[i][k];
	for (j = k; j < n; j++) {
	  a[i][j] -= tmp*a[iprow][j];
	}
	xb[i] -= tmp*xb[iprow];
      }
    }
  }

  /* Now do the back substitution */

  for (i = n - 1; i > -1; i--) {
    iprow = ipivot[i];
    tmp = xb[iprow];
    for (k = i + 1; k < n; k++) {
      tmp -= a[iprow][k]*xb[ipivot[k]];
    }
    xb[iprow] = tmp;
  }

  free(ipivot);

  if (ifail != 0) fatal("IFAIL FAILED\n");

  return ifail;
}
