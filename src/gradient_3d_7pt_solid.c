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

static void gradient_x_general(const int index, const int nhat[3],
			       const double dn[3],
			       const double * field, double * grad,
			       double * del2);

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
  gradient_3d_7pt_solid_try2_wall(nop, field, grad, delsq, nextra);

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

  if (wall_at_edge(X) == 0) return;

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
	gradient_x_general(index, nhat, dn, field, grad, del2);
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
	gradient_x_general(index, nhat, dn, field, grad, del2);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_x_general
 *
 *****************************************************************************/

static void gradient_x_general(const int index, const int nhat[3],
			       const double dn[3],
			       const double * field, double * grad,
			       double * del2) {
  int ia, ib, ic, ig, n;
  int nhalo;
  int nlocal[3];
  int xs, ys;
  int nsolid;            /* Points to solid partial gradient */
  int nfluid;            /* Points to fluid partial gradient */

  double gradx[NOP][2];  /* grad[n][0] is forward gradient, Q_{ic+1} - Q_ic
		            grad[n][1] backward gradient,   Q_ic - Q_{ic-1} */ 
  double qs[NOP];
  double qsab[3][3];
  double q0[3][3];

  double a[5][5];        /* Linear algebra Ax = b */
  double b[5];           /* b is the rhs/solution vector */
  double c[3][3];        /* Constant terms */

  double kappa0, kappa1, q_0, w;
  double kappa2;

  double dyqxx, dzqxx, dyqxy, dzqxy, dyqxz;
  double dzqxz, dyqyy, dzqyy, dyqyz, dzqyz;

  assert(NOP == 5);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  ys = (nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  kappa0 = fe_kappa();
  kappa1 = fe_kappa(); /* One elastic constant */ 
  kappa2 = kappa0 + kappa1;
  q_0 = blue_phase_q0();
  w = colloids_q_tensor_w();

  nfluid = (1 - nhat[X])/2;
  nsolid = (1 + nhat[X])/2;

  assert(nfluid == 0 || nfluid == 1);
  assert(nsolid == 0 || nsolid == 1);
  assert(nfluid != nsolid);

  for (n = 0; n < NOP; n++) {
    gradx[n][0] = field[NOP*(index + xs) + n] - field[NOP*index + n];
    gradx[n][1] = field[NOP*index + n] - field[NOP*(index - xs) + n];
    qs[n] = field[NOP*index + n] - 0.5*dn[X]*gradx[n][nfluid];
  }

  fed_q5_to_qab(qsab, qs);
  colloids_q_boundary(dn, qsab, q0);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      fe_wall_[nsolid] +=
	0.5*w*(qsab[ia][ib] - q0[ia][ib])*(qsab[ia][ib] - q0[ia][ib]);
    }
  }

  /* Tangential derivatives from fluid */

  dyqxx = grad[3*(NOP*index + XX) + Y];
  dzqxx = grad[3*(NOP*index + XX) + Z];
  dyqxy = grad[3*(NOP*index + XY) + Y];
  dzqxy = grad[3*(NOP*index + XY) + Z];
  dyqxz = grad[3*(NOP*index + XZ) + Y];
  dzqxz = grad[3*(NOP*index + XZ) + Z];
  dyqyy = grad[3*(NOP*index + YY) + Y];
  dzqyy = grad[3*(NOP*index + YY) + Z];
  dyqyz = grad[3*(NOP*index + YZ) + Y];
  dzqyz = grad[3*(NOP*index + YZ) + Z];

  /* constant terms k1*q0*(e_agc Q_cb + e_bgc Q_Ca) n_g + w*(qs - q0)_ab */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      c[ia][ib] = 0.0;
      for (ig = 0; ig < 3; ig++) {
	for (ic = 0; ic < 3; ic++) {
	  c[ia][ib] += kappa1*q_0*dn[ig]*
	    (e_[ia][ig][ic]*qsab[ic][ib] + e_[ib][ig][ic]*qsab[ic][ia]);
	}
      }
      c[ia][ib] += w*(qsab[ia][ib] - q0[ia][ib]);
    }
  }

  /* Construct coefficient matrix */

  a[XX][XX] =  kappa0*dn[X];
  a[XX][XY] = -kappa1*dn[Y];
  a[XX][XZ] = -kappa1*dn[Z];
  a[XX][YY] =  0.0;
  a[XX][YZ] =  0.0;

  b[XX] = -(kappa1*dn[Y]*dyqxx + kappa0*dn[X]*dyqxy
	  + kappa1*dn[Z]*dzqxx + kappa0*dn[X]*dzqxz + c[X][X]);

  a[XY][XX] =  kappa0*dn[Y];
  a[XY][XY] =  kappa2*dn[X];
  a[XY][XZ] =  0.0;
  a[XY][YY] = -kappa1*dn[Y];
  a[XY][YZ] = -kappa1*dn[Z];

  b[XY] = -(-kappa1*dn[X]*dyqxx + kappa2*dn[Y]*dyqxy - kappa1*dn[Z]*dyqxz
	    + kappa0*dn[X]*dyqyy
	    + 2.0*kappa1*dn[Z]*dzqxy + kappa0*dn[Y]*dzqxz
	    + kappa0*dn[X]*dzqyz + 2.0*c[X][Y]);

  a[XZ][XX] =  kappa2*dn[Z];
  a[XZ][XY] =  0.0;
  a[XZ][XZ] =  kappa2*dn[X];
  a[XZ][YY] =  kappa1*dn[Z];
  a[XZ][YZ] = -kappa1*dn[Y];

  b[XZ] = -(kappa0*dn[Z]*dyqxy + 2.0*kappa1*dn[Y]*dyqxz + kappa0*dn[X]*dyqyz
	    - kappa2*dn[X]*dzqxx -kappa1*dn[Y]*dzqxy + kappa2*dn[Z]*dzqxz
	    - kappa0*dn[X]*dzqyy + 2.0*c[X][Z]);

  a[YY][XX] = 0.0;
  a[YY][XY] = kappa0*dn[Y];
  a[YY][XZ] = 0.0;
  a[YY][YY] = kappa1*dn[X];
  a[YY][YZ] = 0.0;

  b[YY] = -(-kappa1*dn[X]*dyqxy + kappa0*dn[Y]*dyqyy - kappa1*dn[Z]*dyqyz
	    + kappa1*dn[Z]*dzqyy + kappa0*dn[Y]*dzqyz + c[Y][Y]);

  a[YZ][XX] = 0.0;
  a[YZ][XY] = kappa0*dn[Z];
  a[YZ][XZ] = kappa0*dn[Y];
  a[YZ][YY] = 0.0;
  a[YZ][YZ] = 2.0*kappa1*dn[X];

  b[YZ] = -(kappa1*dn[Z]*dyqxx - kappa1*dn[X]*dyqxz + kappa2*dn[Z]*dyqyy
	    + kappa2*dn[Y]*dyqyz
	    - kappa0*dn[Y]*dzqxx - kappa1*dn[X]*dzqxy - kappa2*dn[Y]*dzqyy
	    + kappa2*dn[Z]*dzqyz + 2.0*c[Y][Z]);

  /* SOLVE LINEAR SYSTEM b <- A^{-1} b */

  util_solve_linear_system(5, a, b);

  gradx[XX][nsolid] = b[XX];
  gradx[XY][nsolid] = b[XY];
  gradx[XZ][nsolid] = b[XZ];
  gradx[YY][nsolid] = b[YY];
  gradx[YZ][nsolid] = b[YZ];

  for (n = 0; n < NOP; n++) {
    grad[3*(NOP*index + n) + X] = 0.5*(dn[X]*gradx[n][0] - dn[X]*gradx[n][1]);
    del2[NOP*index + n]
      = dn[X]*gradx[n][0] + dn[X]*gradx[n][1]
      + field[NOP*(index + ys) + n] + field[NOP*(index - ys) + n]
      + field[NOP*(index + 1 ) + n] + field[NOP*(index - 1 ) + n]
      - 4.0*field[NOP*index + n];
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
  int nhalo;
  int n;
  int ic, kc;
  int index;
  int xs, ys;

  double gradm1[5], gradp1[5];  /* gradient terms */
  double kappa0, kappa1;        /* Two elastic constants */

  double qs[5];                 /* surface q extrapolated from fluid */
  double q0, w;

  double qxxx, qzxx, qxxy, qzxy, qxxz;
  double qzxz, qzyy, qxyy, qxyz, qzyz;

  if (wall_at_edge(Y) == 0) return;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  ys = (nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  assert(wall_at_edge(X) == 0);
  assert(wall_at_edge(Z) == 0);

  kappa0 = fe_kappa();
  kappa1 = kappa0; /* One elastic constant at the moment */
  q0 = 0.19635; /* KLUDGE */
  w = 0.0;

  if (cart_coords(Y) == 0) {

    /* Correct the lower wall */

    for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, 1, kc);

	for (n = 0; n < nop; n++) {
	  gradp1[n] = field[nop*(index + ys) + n] - field[nop*index + n];
	  qs[n] = field[nop*index + n] - 0.5*gradp1[n];
	}

	qxxx = grad[3*(nop*index + XX) + X];
	qzxx = grad[3*(nop*index + XX) + Z];
	qxxy = grad[3*(nop*index + XY) + X];
	qzxy = grad[3*(nop*index + XY) + Z];
	qxxz = grad[3*(nop*index + XZ) + X];
	qzxz = grad[3*(nop*index + XZ) + Z];
	qxyy = grad[3*(nop*index + YY) + X];
	qzyy = grad[3*(nop*index + YY) + Z];
	qxyz = grad[3*(nop*index + YZ) + X];
	qzyz = grad[3*(nop*index + YZ) + Z];

	gradm1[XX] = qxxy - 2.0*q0*qs[XZ];
	gradm1[XY] = (1.0/(kappa0 + kappa1))*
	  (-kappa0*(qxxx + qzxz) + kappa1*qxyy - 2.0*kappa1*q0*qs[YZ]);
	gradm1[XZ] = 0.5*(qxyz + qzxy) + q0*(2.0*qs[XX] + qs[YY]);
	gradm1[YY] = - qxxy - qzyz;
	gradm1[YZ] = (1.0/(kappa0 + kappa1))*
	  (-kappa0*(qxxz - qzxx - qzyy) + kappa1*qzyy + 2.0*kappa1*q0*qs[YZ]);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + Y] = 0.5*(gradp1[n] - gradm1[n]);
	  del2[nop*index + n]
	    = gradp1[n] + gradm1[n]
	    + field[nop*(index + xs) + n] + field[nop*(index - xs) + n]
	    + field[nop*(index + 1 ) + n] + field[nop*(index - 1 ) + n] 
	    - 4.0*field[nop*index + n];
	}

	/* Next site */
      }
    }
  }

  if (cart_coords(Y) == cart_size(Y) - 1) {

    /* Correct the upper wall */

    for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, nlocal[Y], kc);

	for (n = 0; n < nop; n++) {
	  gradm1[n] = field[nop*index + n] - field[nop*(index - ys) + n];
	  qs[n] = field[nop*index + n] + 0.5*gradm1[n];
	}

	qxxx = grad[3*(nop*index + XX) + X];
	qzxx = grad[3*(nop*index + XX) + Z];
	qxxy = grad[3*(nop*index + XY) + X];
	qzxy = grad[3*(nop*index + XY) + Z];
	qxxz = grad[3*(nop*index + XZ) + X];
	qzxz = grad[3*(nop*index + XZ) + Z];
	qxyy = grad[3*(nop*index + YY) + X];
	qzyy = grad[3*(nop*index + YY) + Z];
	qxyz = grad[3*(nop*index + YZ) + X];
	qzyz = grad[3*(nop*index + YZ) + Z];

	gradp1[XX] = qxxy - 2.0*q0*qs[XZ];
	gradp1[XY] = (1.0/(kappa0 + kappa1))*
	  (-kappa0*(qxxx + qzxz) + kappa0*qxyy - 2.0*kappa1*q0*qs[YZ]);
	gradp1[XZ] = 0.5*(qxyz + qzxy) + q0*(2.0*qs[XX] + qs[YY]);
	gradp1[YY] = - qxxy - qzyz;
	gradp1[YZ] = (1.0/(kappa0 + kappa1))*
	  (-kappa0*(qxxz - qzxx - qzyy) + kappa1*qzyy + 2.0*kappa1*q0*qs[YZ]);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + Y] = 0.5*(-gradp1[n] + gradm1[n]);
	  del2[nop*index + n]
	    = -gradp1[n] - gradm1[n]
	    + field[nop*(index + xs) + n] + field[nop*(index - xs) + n]
	    + field[nop*(index + 1 ) + n] + field[nop*(index - 1 ) + n]
	    - 4.0*field[nop*index + n];
	}

	/* Next site */
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
  int nhalo;
  int n;
  int ic, jc;
  int index;
  int xs, ys;

  double gradm1[5], gradp1[5];  /* gradient terms */
  double kappa0, kappa1;        /* Two elastic constants */

  double qs[5];                 /* surface q extrapolated from fluid */
  double q0, w;

  double qxxx, qyxx, qxxy, qyxy, qxxz;
  double qyxz, qxyy, qyyy, qxyz, qyyz;

  if (wall_at_edge(Z) == 0) return;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  ys = (nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  assert(wall_at_edge(X) == 0);
  assert(wall_at_edge(Y) == 0);

  kappa0 = fe_kappa();
  kappa1 = kappa0; /* One elastic constant at the moment */
  q0 = 0.19635; /* KLUDGE */
  w = 0.0;

  if (cart_coords(Z) == 0) {

    /* Correct the lower wall */

    for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

	index = coords_index(ic, jc, 1);

	for (n = 0; n < nop; n++) {
	  gradp1[n] = field[nop*(index + 1) + n] - field[nop*index + n];
	  qs[n] = field[nop*index + n] - 0.5*gradp1[n];
	}

	qxxx = grad[3*(nop*index + XX) + X];
	qyxx = grad[3*(nop*index + XX) + Y];
	qxxy = grad[3*(nop*index + XY) + X];
	qyxy = grad[3*(nop*index + XY) + Y];
	qxxz = grad[3*(nop*index + XZ) + X];
	qyxz = grad[3*(nop*index + XZ) + Y];
	qxyy = grad[3*(nop*index + YY) + X];
	qyyy = grad[3*(nop*index + YY) + Y];
	qxyz = grad[3*(nop*index + YZ) + X];
	qyyz = grad[3*(nop*index + YZ) + Y];

	gradm1[XX] = qxxz + 2.0*q0*qs[XY];
	gradm1[XY] = 0.5*(qxyz + qyxz) - q0*(qs[XX] - qs[YY]);
	gradm1[XZ] = (1.0/(kappa0 + kappa1))*
	  (-kappa0*(qxxx + qyxy) + kappa1*(- qxxx - qxyy)
	   + 2.0*kappa1*q0*qs[YZ]);
	gradm1[YY] = qyyz - 2.0*q0*qs[XY];
	gradm1[YZ] = (1.0/(kappa0 + kappa1))*
	  (-kappa0*(qxxy + qyyy) + kappa1*(-qyxx - qyyy)
	   + 2.0*kappa1*q0*qs[XZ]);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + Z] = 0.5*(gradp1[n] - gradm1[n]);
	  del2[nop*index + n]
	    = gradp1[n] + gradm1[n]
	    + field[nop*(index + xs) + n] + field[nop*(index - xs) + n]
	    + field[nop*(index + ys) + n] + field[nop*(index - ys) + n] 
	    - 4.0*field[nop*index + n];
	}

	/* Next site */
      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {

    /* Correct the upper wall */

    for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

	index = coords_index(ic, jc, nlocal[Z]);

	for (n = 0; n < nop; n++) {
	  gradm1[n] = field[nop*index + n] - field[nop*(index - 1) + n];
	  qs[n] = field[nop*index + n] + 0.5*gradm1[n];
	}

	qxxx = grad[3*(nop*index + XX) + X];
	qyxx = grad[3*(nop*index + XX) + Y];
	qxxy = grad[3*(nop*index + XY) + X];
	qyxy = grad[3*(nop*index + XY) + Y];
	qxxz = grad[3*(nop*index + XZ) + X];
	qyxz = grad[3*(nop*index + XZ) + Y];
	qxyy = grad[3*(nop*index + YY) + X];
	qyyy = grad[3*(nop*index + YY) + Y];
	qxyz = grad[3*(nop*index + YZ) + X];
	qyyz = grad[3*(nop*index + YZ) + Y];

	gradp1[XX] = qxxz + 2.0*q0*qs[XY];
	gradp1[XY] = 0.5*(qxyz + qyxz) - q0*(qs[XX] - qs[YY]);
	gradp1[XZ] = (1.0/(kappa0 + kappa1))*
	  (-kappa0*(qxxx + qyxy) + kappa1*(- qxxx - qxyy)
	   + 2.0*kappa1*q0*qs[YZ]);
	gradp1[YY] = qyyz - 2.0*q0*qs[XY];
	gradp1[YZ] = (1.0/(kappa0 + kappa1))*
	  (-kappa0*(qxxy + qyyy) + kappa1*(-qyxx - qyyy)
	   + 2.0*kappa1*q0*qs[XZ]);

	for (n = 0; n < nop; n++) {
	  grad[3*(nop*index + n) + Z] = 0.5*(-gradp1[n] + gradm1[n]);
	  del2[nop*index + n]
	    = -gradp1[n] - gradm1[n]
	    + field[nop*(index + xs) + n] + field[nop*(index - xs) + n]
	    + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	    - 4.0*field[nop*index + n];
	}

	/* Next site */
      }
    }
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
