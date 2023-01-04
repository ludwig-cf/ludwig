/*****************************************************************************
 *
 *  lc_anchoring_impl.h
 *
 *  Inline routines for liquid crystal anchoring.
 *
 *  Normal anchoring and fixed anchoring arre rather similar.
 *  Planar anchoring follows Fournier and Galatola
 *  Europhys. Lett. 72, 403 (2005).
 *
 *  This file is intended to be inlined via an include statement
 *  as it is repeated in a number of places, and performance is
 *  sensitive to contractions involving the permutation tensor...
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LC_ANCHORING_IMPL_C
#define LUDWIG_LC_ANCHORING_IMPL_C

#include <assert.h>

#include "coords.h"

/*****************************************************************************
 *
 *  lc_anchoring_kappa1_ct
 *
 *  The constant term in kappa1 in the anchoring boundary term as a function
 *  of the preferred anchoring unit vector nhat, and the surface Q_ab.
 *
 *    c_ab = - kappa1 q_0 nhat_g (e_agh Q_hb + e_bgh Q_ha)
 *
 *  The permutation tensor e_abc is expanded as this is a performance
 *  sensitive operation (no question about it).
 *
 *  Arguments kappa1 and q0 are the elastic constant and the pitch
 *  wavevector, respectively.
 *
 *****************************************************************************/

__host__ __device__
static inline void lc_anchoring_kappa1_ct(double kappa1, double q0,
					  const double nhat[3],
					  const double qs[3][3],
					  double c[3][3]) {

  double kq = -kappa1*q0;

  c[X][X] = kq*(nhat[Y]*( qs[Z][X] +qs[Z][X]) + nhat[Z]*(-qs[Y][X] -qs[Y][X]));
  c[X][Y] = kq*(nhat[X]*(          -qs[Z][X]) + nhat[Y]*( qs[Z][Y]          )
	  +     nhat[Z]*(           qs[X][X]) + nhat[Z]*(-qs[Y][Y]          ));
  c[X][Z] = kq*(nhat[X]*(           qs[Y][X]) + nhat[Y]*(          -qs[X][X])
	  +     nhat[Y]*( qs[Z][Z]          ) + nhat[Z]*(-qs[Y][Z]          ));

  c[Y][X] = kq*(nhat[X]*(-qs[Z][X]          ) + nhat[Y]*(           qs[Z][Y])
	  +     nhat[Z]*( qs[X][X]          ) + nhat[Z]*(          -qs[Y][Y]));
  c[Y][Y] = kq*(nhat[X]*(-qs[Z][Y] -qs[Z][Y]) + nhat[Z]*( qs[X][Y] +qs[X][Y]));
  c[Y][Z] = kq*(nhat[X]*(           qs[Y][Y]) + nhat[X]*(-qs[Z][Z]          )
	  +     nhat[Y]*(          -qs[X][Y]) + nhat[Z]*( qs[X][Z]          ));

  c[Z][X] = kq*(nhat[X]*( qs[Y][X]          ) + nhat[Y]*(-qs[X][X]          )
	  +     nhat[Y]*(           qs[Z][Z]) + nhat[Z]*(          -qs[Y][Z]));
  c[Z][Y] = kq*(nhat[X]*( qs[Y][Y]          ) + nhat[X]*(          -qs[Z][Z])
	  +     nhat[Y]*(-qs[X][Y]          ) + nhat[Z]*(           qs[X][Z]));
  c[Z][Z] = kq*(nhat[X]*( qs[Y][Z] +qs[Y][Z]) + nhat[Y]*(-qs[X][Z] -qs[X][Z]));

  return;
}

/****************************************************************************
 *
 *  lc_anchoring_fixed_q0
 *
 *  Preferred anchoring q0 for fixed anchoring unit director nfix:
 *
 *    Q_ab = 0.5*amp*(3.0 nfix_a nfix_b - delta_ab)
 *
 ****************************************************************************/ 

__host__ __device__
static inline void lc_anchoring_fixed_q0(const double nhat[3],
					 double amp,
					 double q0[3][3]) {

  q0[X][X] = 0.5*amp*(3.0*nhat[X]*nhat[X] - 1.0);
  q0[X][Y] = 0.5*amp*(3.0*nhat[X]*nhat[Y] - 0.0);
  q0[X][Z] = 0.5*amp*(3.0*nhat[X]*nhat[Z] - 0.0);

  q0[Y][X] = 0.5*amp*(3.0*nhat[Y]*nhat[X] - 0.0);
  q0[Y][Y] = 0.5*amp*(3.0*nhat[Y]*nhat[Y] - 1.0);
  q0[Y][Z] = 0.5*amp*(3.0*nhat[Y]*nhat[Z] - 0.0);

  q0[Z][X] = 0.5*amp*(3.0*nhat[Z]*nhat[X] - 0.0);
  q0[Z][Y] = 0.5*amp*(3.0*nhat[Z]*nhat[Y] - 0.0);
  q0[Z][Z] = 0.5*amp*(3.0*nhat[Z]*nhat[Z] - 1.0);

  return;
}

/*****************************************************************************
 *
 *  lc_anchoring_fixed_ct
 *
 *  Compute the constant term in the boundary condition equation.
 *  This is
 *
 *    -kappa1 q0 nhat_g ( e_agh Q_hb + e_bgh Q_ha) - w1 (Q_ab - Q^0_ab)
 *
 *  The Q_ab is the surface Q tensor: argument qs.
 *
 *****************************************************************************/

__host__ __device__
static inline void lc_anchoring_fixed_ct(const lc_anchoring_param_t * anch,
					 const double qs[3][3],
					 const double nhat[3],
					 double kappa1,
					 double q0,
					 double amp,
					 double ct[3][3]) {
  double qfix[3][3] = {0};

  assert(anch->type == LC_ANCHORING_FIXED);

  lc_anchoring_kappa1_ct(kappa1, q0, nhat, qs, ct);
  lc_anchoring_fixed_q0(anch->nfix, amp, qfix);

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      ct[ia][ib] += -anch->w1*(qs[ia][ib] - qfix[ia][ib]);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lc_anchoring_normal_q0
 *
 *  Prefered Q_ab for unit nhat outward normal.
 *
 *****************************************************************************/

__host__ __device__
static inline void lc_anchoring_normal_q0(const double nhat[3],
					  double amp,
					  double q0[3][3]) {
  /* These are the same. */
  lc_anchoring_fixed_q0(nhat, amp, q0);

  return;
}

/*****************************************************************************
 *
 *  lc_anchoring_normal_ct
 *
 *****************************************************************************/

__host__ __device__
static inline void lc_anchoring_normal_ct(const lc_anchoring_param_t * anch,
					  const double qs[3][3],
					  const double nhat[3],
					  double kappa1,
					  double q0,
					  double amp,
					  double ct[3][3]) {

  double qnormal[3][3] = {0};  /* Preferred Q at boundary */

  assert(anch->type == LC_ANCHORING_NORMAL);

  lc_anchoring_kappa1_ct(kappa1, q0, nhat, qs, ct);
  lc_anchoring_normal_q0(nhat, amp, qnormal);

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      ct[ia][ib] += -anch->w1*(qs[ia][ib] - qnormal[ia][ib]);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lc_anchoring_planar_qtilde
 *
 *  Following Fournier and Galatola, this is
 *
 *    Q~_ab = Q_ab + (1/2) A d_ab
 *
 *****************************************************************************/

__host__ __device__
static inline void lc_anchoring_planar_qtilde(double a0,
					      const double qs[3][3],
					      double qtilde[3][3]) {

  qtilde[X][X] = qs[X][X] + 0.5*a0;
  qtilde[X][Y] = qs[X][Y];
  qtilde[X][Z] = qs[X][Z];
  qtilde[Y][X] = qs[Y][X];
  qtilde[Y][Y] = qs[Y][Y] + 0.5*a0;
  qtilde[Y][Z] = qs[Y][Z];
  qtilde[Z][X] = qs[Z][X];
  qtilde[Z][Y] = qs[Z][Y];
  qtilde[Z][Z] = qs[Z][Z] + 0.5*a0;

  return;
}

/*****************************************************************************
 *
 *  lc_anchoring_planar_ct
 *
 *  The planar (or homoetropic) boundary condition as described by
 *  Fournier and Galatola.
 *
 *  This is the contribution to the constant term in the boundary
 *  condition equation.
 *
 *****************************************************************************/

__host__ __device__
static inline void lc_anchoring_planar_ct(const lc_anchoring_param_t * anchor,
					  const double qs[3][3],
					  const double nhat[3],
					  double kappa1,
					  double q0,
					  double amp,
					  double ct[3][3]) {

  double qtilde[3][3] = {0};

  assert(anchor->type == LC_ANCHORING_PLANAR);

  lc_anchoring_kappa1_ct(kappa1, q0, nhat, qs, ct);
  lc_anchoring_planar_qtilde(amp, qs, qtilde);

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      double qtperp = 0.0;
      double qt2    = 0.0;     /* Q~^2 */
      double s0     = 1.5*amp; /* Fournier and Galatola S_0 = (3/2)A */
      for (int ig = 0; ig < 3; ig++) {
	for (int ih = 0; ih < 3; ih++) {
	  double dag = 1.0*(ia == ig);
	  double dhb = 1.0*(ih == ib);
	  double pag = dag - nhat[ia]*nhat[ig];
	  double phb = dhb - nhat[ih]*nhat[ib];
	  qtperp += pag*qtilde[ig][ih]*phb;
	  qt2 += qtilde[ig][ih]*qtilde[ig][ih];
	}
      }
      /* Contribute to surface terms ... */
      ct[ia][ib] += -anchor->w1*(qtilde[ia][ib] - qtperp);
      ct[ia][ib] += -2.0*anchor->w2*(qt2 - s0*s0)*qtilde[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lc_anchoring_coefficents
 *
 *  This computes the coefficients in the gradient terms of the boundary
 *  conidtion equation which are related to the two elastic constants
 *  and the unit outward normal (only). The outward normal is the
 *  integer unit vector dn.
 *
 *  There are six equations each with eighteen terms d_gamma Q_ab.
 *  The eighteen derivatives are the bc[][6][3] read Q_ab,gamma.
 *
 *****************************************************************************/

__host__ __device__
static inline void lc_anchoring_coefficients(double kappa0,
					     double kappa1,
					     const int dn[3],
					     double bc[NSYMM][NSYMM][3]) {
  double kappa2 = kappa0 + kappa1;

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

  return;
}

#endif
