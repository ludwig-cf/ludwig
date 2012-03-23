/*****************************************************************************
 *
 *  blue_phase.c
 *
 *  Routines related to blue phase liquid crystal free energy
 *  and molecular field.
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
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "phi.h"
#include "phi_gradients.h"
#include "blue_phase.h"
#include "ran.h"

static double q0_;        /* Pitch = 2pi / q0_ */
static double a0_;        /* Bulk free energy parameter A_0 */
static double gamma_;     /* Controls magnitude of order */
static double kappa0_;    /* Elastic constant \kappa_0 */
static double kappa1_;    /* Elastic constant \kappa_1 */

static double xi_;        /* effective molecular aspect ratio (<= 1.0) */
static double redshift_;  /* redshift parameter */
static double rredshift_; /* reciprocal */
static double zeta_;      /* Apolar activity parameter \zeta */

static int redshift_update_ = 0; /* Dynamic cubic redshift update */
static int output_to_file_  = 1; /* To stdout or "free_energy.dat" */
static double amplitude_ = 0.0;  /* Magnitude of order (uniaxial) */
static double epsilon_ = 0.0;    /* Dielectric anisotropy (e/12pi) */

static double electric_[3] = {0.0, 0.0, 0.0}; /* Electric field */

static const double redshift_min_ = 0.00000000001; 

/*****************************************************************************
 *
 *  blue_phase_set_free_energy_parameters
 *
 *  Enforces the 'one constant approximation' kappa0 = kappa1 = kappa
 *
 *  Note that these values remain unchanged throughout. Redshifted
 *  values are computed separately as needed.
 *
 *****************************************************************************/

void blue_phase_set_free_energy_parameters(double a0, double gamma,
					   double kappa, double q0) {
  a0_ = a0;
  gamma_ = gamma;
  kappa0_ = kappa;
  kappa1_ = kappa;
  q0_ = q0;

  /* Anchoring boundary conditions require kappa0 from free energy */
  fe_kappa_set(kappa0_);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_amplitude
 *
 *****************************************************************************/

double blue_phase_amplitude(void) {

  return amplitude_;
}

/*****************************************************************************
 *
 *  blue_phase_amplitude_set
 *
 *****************************************************************************/

void blue_phase_amplitude_set(const double a) {

  amplitude_ = a;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_amplitude_calculate
 *
 *****************************************************************************/

double blue_phase_amplitude_calculate(void) {
  double amplitude;
  
  amplitude = 2.0/3.0*(0.25 + 0.75*sqrt(1.0 - 8.0/(3.0*gamma_)));

  return amplitude;
}

/*****************************************************************************
 *
 *  blue_phase_set_xi
 *
 *  Set the molecular aspect ratio.
 *
 *****************************************************************************/

void blue_phase_set_xi(double xi) {

  xi_ = xi;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_get_xi
 *
 *****************************************************************************/

double blue_phase_get_xi(void) {

  return xi_;
}

/*****************************************************************************
 *
 *  blue_phase_set_zeta
 *
 *  Set the activity parameter.
 *
 *****************************************************************************/

void blue_phase_set_zeta(double zeta) {

  zeta_ = zeta;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_get_zeta
 *
 *****************************************************************************/

double blue_phase_get_zeta(void) {

  return zeta_;
}

/*****************************************************************************
 *
 *  blue_phase_set_gamma
 *
 *  Set the gamma_ parameter.
 *
 *****************************************************************************/

void blue_phase_set_gamma(double gamma) {

  gamma_ = gamma;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_get_gamma
 *
 *****************************************************************************/

double blue_phase_get_gamma(void) {

  return gamma_;
}

/*****************************************************************************
 *
 *  blue_phase_free_energy_density
 *
 *  Return the free energy density at lattice site index.
 *
 *****************************************************************************/

double blue_phase_free_energy_density(const int index) {

  double e;
  double q[3][3];
  double dq[3][3][3];

  phi_get_q_tensor(index, q);
  phi_gradients_tensor_gradient(index, dq);
  
  e = blue_phase_compute_fed(q, dq);

  return e;
}

/*****************************************************************************
 *
 *  blue_phase_compute_fed
 *
 *  Compute the free energy density as a function of q and the q gradient
 *  tensor dq.
 *
 *****************************************************************************/

double blue_phase_compute_fed(double q[3][3], double dq[3][3][3]) {

  int ia, ib, ic, id;
  double q0;
  double kappa0, kappa1;
  double q2, q3;
  double dq0, dq1;
  double sum;
  double efield;

  q0 = q0_*rredshift_;
  kappa0 = kappa0_*redshift_*redshift_;
  kappa1 = kappa1_*redshift_*redshift_;

  q2 = 0.0;

  /* Q_ab^2 */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += q[ia][ib]*q[ia][ib];
    }
  }

  /* Q_ab Q_bc Q_ca */

  q3 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	/* We use here the fact that q[ic][ia] = q[ia][ic] */
	q3 += q[ia][ib]*q[ib][ic]*q[ia][ic];
      }
    }
  }

  /* (d_b Q_ab)^2 */

  dq0 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    sum = 0.0;
    for (ib = 0; ib < 3; ib++) {
      sum += dq[ib][ia][ib];
    }
    dq0 += sum*sum;
  }

  /* (e_acd d_c Q_db + 2q_0 Q_ab)^2 */
  /* With symmetric Q_db write Q_bd */

  dq1 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum += e_[ia][ic][id]*dq[ic][ib][id];
	}
      }
      sum += 2.0*q0*q[ia][ib];
      dq1 += sum*sum;
    }
  }

  /* Electric field term (epsilon_ includes the factor 1/12pi) */

  efield = 0.0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      efield += electric_[ia]*q[ia][ib]*electric_[ib];
    }
  }

  sum = 0.5*a0_*(1.0 - r3_*gamma_)*q2 - r3_*a0_*gamma_*q3 +
    0.25*a0_*gamma_*q2*q2 + 0.5*kappa0*dq0 + 0.5*kappa1*dq1 - epsilon_*efield;;

  return sum;
}

/*****************************************************************************
 *
 *  blue_phase_molecular_field
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *  Note this is only valid in the one-constant approximation at
 *  the moment (kappa0 = kappa1 = kappa).
 *
 *****************************************************************************/

void blue_phase_molecular_field(int index, double h[3][3]) {

  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  assert(kappa0_ == kappa1_);

  phi_get_q_tensor(index, q);
  phi_gradients_tensor_gradient(index, dq);
  phi_gradients_tensor_delsq(index, dsq);

  blue_phase_compute_h(q, dq, dsq, h);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_compute_h
 *
 *  Compute the molcular field h from q, the q gradient tensor dq, and
 *  the del^2 q tensor.
 *
 *****************************************************************************/

void blue_phase_compute_h(double q[3][3], double dq[3][3][3],
			  double dsq[3][3], double h[3][3]) {
  int ia, ib, ic, id;

  double q0;              /* Redshifted value */
  double kappa0, kappa1;  /* Redshifted values */
  double q2;
  double e2;
  double eq;
  double sum;

  q0 = q0_*rredshift_;
  kappa0 = kappa0_*redshift_*redshift_;
  kappa1 = kappa1_*redshift_*redshift_;

  /* From the bulk terms in the free energy... */

  q2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += q[ia][ib]*q[ia][ib];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	sum += q[ia][ic]*q[ib][ic];
      }
      h[ia][ib] = -a0_*(1.0 - r3_*gamma_)*q[ia][ib]
	+ a0_*gamma_*(sum - r3_*q2*d_[ia][ib]) - a0_*gamma_*q2*q[ia][ib];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  eq = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	eq += e_[ib][ic][ia]*dq[ib][ic][ia];
      }
    }
  }

  /* d_c Q_db written as d_c Q_bd etc */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum +=
	    (e_[ia][ic][id]*dq[ic][ib][id] + e_[ib][ic][id]*dq[ic][ia][id]);
	}
      }
      h[ia][ib] += kappa1*dsq[ia][ib]
	- 2.0*kappa1*q0*sum + 4.0*r3_*kappa1*q0*eq*d_[ia][ib]
	- 4.0*kappa1*q0*q0*q[ia][ib];
    }
  }

  /* Electric field term */

  e2 = 0.0;
  for (ia = 0; ia < 3; ia++) {
    e2 += electric_[ia]*electric_[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] +=  epsilon_*(electric_[ia]*electric_[ib] - r3_*d_[ia][ib]*e2);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_chemical_stress
 *
 *  Return the stress sth[3][3] at lattice site index.
 *
 *****************************************************************************/

void blue_phase_chemical_stress(int index, double sth[3][3]) {

  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  phi_get_q_tensor(index, q);
  phi_gradients_tensor_gradient(index, dq);
  phi_gradients_tensor_delsq(index, dsq);

  blue_phase_compute_h(q, dq, dsq, h);
  blue_phase_compute_stress(q, dq, h, sth);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_compute_stress
 *
 *  Compute the stress as a function of the q tensor, the q tensor
 *  gradient and the molecular field.
 *
 *  Note the definition here has a minus sign included to allow
 *  computation of the force as minus the divergence (which often
 *  appears as plus in the liquid crystal literature). This is a
 *  separate operation at the end to avoid confusion.
 *
 *****************************************************************************/

void blue_phase_compute_stress(double q[3][3], double dq[3][3][3],
			       double h[3][3], double sth[3][3]) {
  int ia, ib, ic, id, ie;
  double q0;
  double kappa0;
  double kappa1;
  double qh;
  double p0;

  q0 = q0_*rredshift_;
  kappa0 = kappa0_*redshift_*redshift_;
  kappa1 = kappa1_*redshift_*redshift_;
  
  /* We have ignored the rho T term at the moment, assumed to be zero
   * (in particular, it has no divergence if rho = const). */

  p0 = 0.0 - blue_phase_compute_fed(q, dq);

  /* The contraction Q_ab H_ab */

  qh = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      qh += q[ia][ib]*h[ia][ib];
    }
  }

  /* The term in the isotropic pressure, plus that in qh */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] = -p0*d_[ia][ib] + 2.0*xi_*(q[ia][ib] + r3_*d_[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi and molecular field */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] +=
	  -xi_*h[ia][ic]*(q[ib][ic] + r3_*d_[ib][ic])
	  -xi_*(q[ia][ic] + r3_*d_[ia][ic])*h[ib][ic];
      }
    }
  }

  /* Dot product term d_a Q_cd . dF/dQ_cd,b */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sth[ia][ib] +=
	    - kappa0*dq[ia][ib][ic]*dq[id][ic][id]
	    - kappa1*dq[ia][ic][id]*dq[ib][ic][id]
	    + kappa1*dq[ia][ic][id]*dq[ic][ib][id];

	  for (ie = 0; ie < 3; ie++) {
	    sth[ia][ib] +=
	      -2.0*kappa1*q0*dq[ia][ic][id]*e_[ib][ic][ie]*q[id][ie];
	  }
	}
      }
    }
  }

  /* The antisymmetric piece q_ac h_cb - h_ac q_cb. We can
   * rewrite it as q_ac h_bc - h_ac q_bc. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] += q[ia][ic]*h[ib][ic] - h[ia][ic]*q[ib][ic];
      }
    }
  }

  /* This is the minus sign. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	sth[ia][ib] = -sth[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_O8M_init
 *
 *  BP I using the current free energy parameter q0_
 *
 *****************************************************************************/

void blue_phase_O8M_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double r2;
  double cosx, cosy, cosz, sinx, siny, sinz;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  r2 = sqrt(2.0);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	cosx = cos(r2*q0_*x);
	cosy = cos(r2*q0_*y);
	cosz = cos(r2*q0_*z);
	sinx = sin(r2*q0_*x);
	siny = sin(r2*q0_*y);
	sinz = sin(r2*q0_*z);

	q[X][X] = amplitude_*(-2.0*cosy*sinz +    sinx*cosz + cosx*siny);
	q[X][Y] = amplitude_*(  r2*cosy*cosz + r2*sinx*sinz - sinx*cosy);
	q[X][Z] = amplitude_*(  r2*cosx*cosy + r2*sinz*siny - cosx*sinz);
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude_*(-2.0*sinx*cosz +    siny*cosx + cosy*sinz);
	q[Y][Z] = amplitude_*(  r2*cosz*cosx + r2*siny*sinx - siny*cosz);
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	phi_set_q_tensor(index, q);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_O2_init
 *
 *  This initialisation is for BP II.
 *
 *****************************************************************************/

void blue_phase_O2_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude_*(cos(2.0*q0_*z) - cos(2.0*q0_*y));
	q[X][Y] = amplitude_*sin(2.0*q0_*z);
	q[X][Z] = amplitude_*sin(2.0*q0_*y);
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude_*(cos(2.0*q0_*x) - cos(2.0*q0_*z));
	q[Y][Z] = amplitude_*sin(2.0*q0_*x);
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	phi_set_q_tensor(index, q);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_H2D_init
 *
 *  This initialisation is for 2D hexagonal BP.
 *
 *****************************************************************************/

void blue_phase_H2D_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude_*(-1.5*cos(q0_*x)*cos(q0_*sqrt(3.0)*y));
	q[X][Y] = amplitude_*(-0.5*sqrt(3.0)*sin(q0_*x)*sin(q0_*sqrt(3.0)*y));
	q[X][Z] = amplitude_*(sqrt(3.0)*cos(q0_*x)*sin(q0_*sqrt(3.0)*y));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude_*(-cos(2.0*q0_*x)-0.5*cos(q0_*x)*cos(q0_*sqrt(3.0)*y));
	q[Y][Z] = amplitude_*(-sin(2.0*q0_*x)-sin(q0_*x)*cos(q0_*sqrt(3.0)*y));
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	phi_set_q_tensor(index, q);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_H3DA_init
 *
 *  This initialisation is for 3D hexagonal BP A.
 *
 *****************************************************************************/

void blue_phase_H3DA_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude_*(-1.5*cos(q0_*x)*cos(q0_*sqrt(3.0)*y)+0.25*cos(q0_*N_total(X)/N_total(Z)*z)); 
	q[X][Y] = amplitude_*(-0.5*sqrt(3.0)*sin(q0_*x)*sin(q0_*sqrt(3.0)*y)+0.25*sin(q0_*N_total(X)/N_total(Z)*z));
	q[X][Z] = amplitude_*(sqrt(3.0)*cos(q0_*x)*sin(q0_*sqrt(3.0)*y));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude_*(-cos(2.0*q0_*x)-0.5*cos(q0_*x)*cos(q0_*sqrt(3.0)*y)-0.25*cos(q0_*N_total(X)/N_total(Z)*z));
	q[Y][Z] = amplitude_*(-sin(2.0*q0_*x)-sin(q0_*x)*cos(q0_*sqrt(3.0)*y));
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	phi_set_q_tensor(index, q);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_H3DB_init
 *
 *  This initialisation is for 3D hexagonal BP B.
 *
 *****************************************************************************/

void blue_phase_H3DB_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude_*(1.5*cos(q0_*x)*cos(q0_*sqrt(3.0)*y)+0.25*cos(q0_*N_total(X)/N_total(Z)*z)); 
	q[X][Y] = amplitude_*(0.5*sqrt(3.0)*sin(q0_*x)*sin(q0_*sqrt(3.0)*y)+0.25*sin(q0_*N_total(X)/N_total(Z)*z));
	q[X][Z] = amplitude_*(-sqrt(3.0)*cos(q0_*x)*sin(q0_*sqrt(3.0)*y));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude_*(cos(2.0*q0_*x)+0.5*cos(q0_*x)*cos(q0_*sqrt(3.0)*y)-0.25*cos(q0_*N_total(X)/N_total(Z)*z));
	q[Y][Z] = amplitude_*(sin(2.0*q0_*x)+sin(q0_*x)*cos(q0_*sqrt(3.0)*y));
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	phi_set_q_tensor(index, q);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_O5_init
 *
 *  This initialisation is for O5.
 *
 *****************************************************************************/

void blue_phase_O5_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude_*
            (2.0*cos(sqrt(2.0)*q0_*y)*cos(sqrt(2.0)*q0_*z)-
                 cos(sqrt(2.0)*q0_*x)*cos(sqrt(2.0)*q0_*z)-
                 cos(sqrt(2.0)*q0_*x)*cos(sqrt(2.0)*q0_*y)); 
	q[X][Y] = amplitude_*
            (sqrt(2.0)*cos(sqrt(2.0)*q0_*y)*sin(sqrt(2.0)*q0_*z)-
             sqrt(2.0)*cos(sqrt(2.0)*q0_*x)*sin(sqrt(2.0)*q0_*z)-
             sin(sqrt(2.0)*q0_*x)*sin(sqrt(2.0)*q0_*y));
	q[X][Z] = amplitude_*
            (sqrt(2.0)*cos(sqrt(2.0)*q0_*x)*sin(sqrt(2.0)*q0_*y)-
             sqrt(2.0)*cos(sqrt(2.0)*q0_*z)*sin(sqrt(2.0)*q0_*y)-
             sin(sqrt(2.0)*q0_*x)*sin(sqrt(2.0)*q0_*z));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude_*
            (2.0*cos(sqrt(2.0)*q0_*x)*cos(sqrt(2.0)*q0_*z)-
                 cos(sqrt(2.0)*q0_*y)*cos(sqrt(2.0)*q0_*x)-
                 cos(sqrt(2.0)*q0_*y)*cos(sqrt(2.0)*q0_*z));
	q[Y][Z] = amplitude_*
            (sqrt(2.0)*cos(sqrt(2.0)*q0_*z)*sin(sqrt(2.0)*q0_*x)-
             sqrt(2.0)*cos(sqrt(2.0)*q0_*y)*sin(sqrt(2.0)*q0_*x)-
             sin(sqrt(2.0)*q0_*y)*sin(sqrt(2.0)*q0_*z));
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	phi_set_q_tensor(index, q);

      }
    }
  }

  return;
}
/*****************************************************************************
 *
 *  blue_phase_DTC_init
 *
 *  This initialisation is with double twist cylinders.
 *
 *****************************************************************************/

void blue_phase_DTC_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = -amplitude_*(cos(2*q0_*y));
	q[X][Y] = 0.0;
	q[X][Z] = amplitude_*sin(2.0*q0_*y);
	q[Y][X] = q[X][Y];
	q[Y][Y] = -amplitude_*(cos(2.0*q0_*x));
	q[Y][Z] = -amplitude_*sin(2.0*q0_*x);
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	phi_set_q_tensor(index, q);

      }
    }
  }

  return;
}


/*****************************************************************************
 *
 *  blue_phase_BPIII_init
 *
 *  This initialisation is with Blue Phase III, randomly positioned
 *  and oriented DTC-cylinders in isotropic (0) or cholesteric (1) environment.
 *
 *  NOTE: The rotations are not rigorously implemented; no cross-boundary 
 *        communication is performed. 
 *        Hence, the decomposition must consist of sufficiently large volumes.
 *        
 *****************************************************************************/

void blue_phase_BPIII_init(const double specs[3]) {

  int ic, jc, kc;
  int ir, jr, kr; 	/* indices for rotated output */
  int ia, ib, ik, il, is, it, in;
  int nlocal[3];
  int noffset[3];
  int index;
  double q[3][3], q0[3][3], qr[3][3];
  double x, y, z;
  double *a, *b;	/* rotation angles */
  double *C;     	/* global coordinates of DTC-centres */
  int N=2, R=3, ENV=1; 	/* default no. & radius & environment */ 
  double rc[3];	  	/* distance DTC-centre - site */ 
  double rc_r[3]; 	/* rotated vector */ 
  double Mx[3][3], My[3][3]; /* rotation matrices */
  double phase1, phase2;
  double n[3]={0.0,0.0,0.0};

  N = (int) specs[0];
  R = (int) specs[1];
  ENV = (int) specs[2];

  a = (double*)calloc(N, sizeof(double));
  b = (double*)calloc(N, sizeof(double));
  C = (double*)calloc(3*N, sizeof(double));

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

    /* Initialise random rotation angles and centres in serial */
    /* to get the same random numbers on all processes */
    for(in = 0; in < N; in++){

      a[in] = 2.0*pi_ * ran_serial_uniform();
      b[in] = 2.0*pi_ * ran_serial_uniform();
      C[3*in]   = N_total(X) * ran_serial_uniform(); 
      C[3*in+1] = N_total(Y) * ran_serial_uniform(); 
      C[3*in+2] = N_total(Z) * ran_serial_uniform(); 

    }

  /* Setting environment configuration */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (ENV == 0){

	  phase1 = pi_*(0.5 - ran_parallel_uniform());
	  phase2 = pi_*(0.5 - ran_parallel_uniform());

	  n[X] = cos(phase1)*sin(phase2);
	  n[Y] = sin(phase1)*sin(phase2);
	  n[Z] = cos(phase2);

	  blue_phase_q_uniaxial(n, q);

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      q[ia][ib] *= 1.0e-6;
	    }
	  }

	}
	if (ENV == 1){

	  /* cholesteric helix along y-direction */
	  n[X] = cos(q0_*y);
	  n[Y] = 0.0;
	  n[Z] = -sin(q0_*y);

	  blue_phase_q_uniaxial(n, q);

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      q[ia][ib] *= amplitude_;
	    }
	  }
	  
	}

	index = coords_index(ic, jc, kc);
	phi_set_q_tensor(index, q);

      }
    }
  }

  /* Replace configuration inside DTC-domains */
  /* by sweeping through all local sites */
  for(in = 0; in<N; in++){

    blue_phase_M_rot(Mx,0,a[in]);
    blue_phase_M_rot(My,1,b[in]);

    for (ic = 1; ic <= nlocal[X]; ic++) {
      x = noffset[X] + ic;
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	y = noffset[Y] + jc;
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  z = noffset[Z] + kc;

	  rc[X] = x - C[3*in];
	  rc[Y] = y - C[3*in+1];
	  rc[Z] = z - C[3*in+2];

	  /* If current site is in ROI perform double */
	  /* rotation around local x- and y-axis */
	  if(rc[0]*rc[0] + rc[1]*rc[1] + rc[2]*rc[2] < R*R){

	    for(ia=0; ia<3; ia++){
	      rc_r[ia] = 0.0;
	      for(ik=0; ik<3; ik++){
		for(il=0; il<3; il++){
		  rc_r[ia] += My[ia][ik] * Mx[ik][il] * rc[il];
		}
	      }
	    }

	    /* DTC symmetric wrt local z-axis */
	    q0[X][X] = -amplitude_*(cos(2*q0_*rc[Y]));
	    q0[X][Y] = 0.0;
	    q0[X][Z] = amplitude_*sin(2.0*q0_*rc[Y]);
	    q0[Y][X] = q[X][Y];
	    q0[Y][Y] = -amplitude_*(cos(2.0*q0_*rc[X]));
	    q0[Y][Z] = -amplitude_*sin(2.0*q0_*rc[X]);
	    q0[Z][X] = q[X][Z];
	    q0[Z][Y] = q[Y][Z];
	    q0[Z][Z] = - q[X][X] - q[Y][Y];

	    /* Transform order parameter tensor */ 
/***************************************************************
* NOTE: This has been commented out as a similar rotation of the
*       order parameter leads to considerable instabilities in 
*       the calculation of the gradients.
*       BPIII emerges more reliably from an unrotated OP.
***************************************************************/
/*
            for (ia=0; ia<3; ia++){
              for (ib=0; ib<3; ib++){
                qr[ia][ib] = 0.0;
                for (ik=0; ik<3; ik++){
                  for (il=0; il<3; il++){
		    for (is=0; is<3; is++){
		      for (it=0; it<3; it++){

			qr[ia][ib] += My[ia][is] * Mx[is][ik] * \
				q0[ik][il] * Mx[it][il] * My[ib][it];

		      }
		    }
                  }
                }
              }
            }
*/
            /* Determine local output index */
            ir = (int)(C[3*in] + rc_r[X] - noffset[X]);
            jr = (int)(C[3*in+1] + rc_r[Y] - noffset[Y]);
            kr = (int)(C[3*in+2] + rc_r[Z] - noffset[Z]);

	    /* Replace if index is in local domain */
	    if((1 <= ir && ir <= nlocal[X]) &&  
	       (1 <= jr && jr <= nlocal[Y]) &&  
               (1 <= kr && kr <= nlocal[Z]))
	    {

	      /* see comment above */
/*
	      q[X][X] = qr[X][X];
	      q[X][Y] = qr[X][Y];
	      q[X][Z] = qr[X][Z];
	      q[Y][X] = q[X][Y];
	      q[Y][Y] = qr[Y][Y];
	      q[Y][Z] = qr[Y][Z];
	      q[Z][X] = q[X][Z];
	      q[Z][Y] = q[Y][Z];
	      q[Z][Z] = - q[X][X] - q[Y][Y];
*/

	      q[X][X] = q0[X][X];
	      q[X][Y] = q0[X][Y];
	      q[X][Z] = q0[X][Z];
	      q[Y][X] = q[X][Y];
	      q[Y][Y] = q0[Y][Y];
	      q[Y][Z] = q0[Y][Z];
	      q[Z][X] = q[X][Z];
	      q[Z][Y] = q[Y][Z];
	      q[Z][Z] = - q[X][X] - q[Y][Y];

	      index = coords_index(ir, jr, kr);
	      phi_set_q_tensor(index, q);
	    }

	  }

	}
      }
    }

  }

  phi_halo();

  free(a);
  free(b);
  free(C);

  return;
}


/*****************************************************************************
 *
 *  blue_phase_twist_init
 *
 *  Initialise a uniaxial helix in the indicated helical axis.
 *  Uses the current free energy parameters
 *     q0_ (P=2pi/q0)
 *     amplitude
 *
 *****************************************************************************/

void blue_phase_twist_init(const int helical_axis) {
  
  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double n[3];
  double q[3][3];
  double x, y, z;
 
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  assert(helical_axis == X || helical_axis == Y || helical_axis == Z);
 
  n[X] = 0.0;
  n[Y] = 0.0;
  n[Z] = 0.0;
 
  for (ic = 1; ic <= nlocal[X]; ic++) {

    if (helical_axis == X) {
      x = noffset[X] + ic;
      n[Y] = cos(q0_*x);
      n[Z] = sin(q0_*x);
    }

    for (jc = 1; jc <= nlocal[Y]; jc++) {

      if (helical_axis == Y) {
	y = noffset[Y] + jc;
	n[X] = cos(q0_*y);
	n[Z] = -sin(q0_*y);
      }

      for (kc = 1; kc <= nlocal[Z]; kc++) {
	
	index = coords_index(ic, jc, kc);

	if (helical_axis == Z) {
	  z = noffset[Z] + kc;
	  n[X] = cos(q0_*z);
	  n[Y] = sin(q0_*z);
	}

	blue_phase_q_uniaxial(n, q);
	phi_set_q_tensor(index, q);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_q_uniaxial
 *
 *  For given director n we return
 *
 *     Q_ab = (1/2) A (3 n_a n_b - d_ab)
 *
 *  where A gives the maximum amplitude of order on diagonalisation.
 *
 *  Note this is slightly different  from the definition in
 *  Wright and Mermin (Eq. 4.3) where
 *
 *     Q_ab = (1/3) gamma (3 n_a n_b - d_ab)
 *
 *  and the magnitude of order is then (2/3) gamma.
 *
 *****************************************************************************/

void blue_phase_q_uniaxial(const double n[3], double q[3][3]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q[ia][ib] = 0.5*amplitude_*(3.0*n[ia]*n[ib] - d_[ia][ib]);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_nematic_init
 *
 *  Initialise a uniform uniaxial nematic.
 *
 *  The inputs are the amplitude A and the vector n_a (which we explicitly
 *  convert to a unit vector here).
 *
 *****************************************************************************/

void blue_phase_nematic_init(const double n[3]) {

  int ic, jc, kc;
  int nlocal[3];
  int ia, index;

  double nhat[3];
  double q[3][3];

  assert(modulus(n) > 0.0);
  coords_nlocal(nlocal);

  for (ia = 0; ia < 3; ia++) {
    nhat[ia] = n[ia] / modulus(n);
  }

  blue_phase_q_uniaxial(nhat, q);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	phi_set_q_tensor(index, q);
      }
    }
  }
  return;
}

/*****************************************************************************
 *
 *  blue_phase_chi_edge
 *  Setting  chi edge disclination
 *  Using the current free energy parameter q0_ (P=2pi/q0)
 *****************************************************************************/

void blue_phase_chi_edge(int N, double z0, double x0) {
  
  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double n[3];
  double theta;
  
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;
	
	index = coords_index(ic, jc, kc);

	theta = 1.0*N/2.0*atan2((1.0*z-z0),(1.0*x-x0)) + q0_*(z-z0);	
	n[X] = cos(theta);
	n[Y] = sin(theta);
	n[Z] = 0.0;

	blue_phase_q_uniaxial(n, q);
	phi_set_q_tensor(index, q);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_set_random_q_init
 *  Setting q tensor to isotropic in chosen area of the simulation box
 * -Juho 12/11/09
 *****************************************************************************/

void blue_set_random_q_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int index;
  int ia, ib;

  double n[3];
  double q[3][3];
  double phase1, phase2;
  
  coords_nlocal(nlocal);
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	phase1 = pi_*(0.5 - ran_parallel_uniform());
	phase2 = pi_*(0.5 - ran_parallel_uniform());
	
	n[X] = cos(phase1)*sin(phase2);
	n[Y] = sin(phase1)*sin(phase2);
	n[Z] = cos(phase2);

	for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
		q[ia][ib] *= 0.00001;
	    }
	}

	blue_phase_q_uniaxial(n, q);
	phi_set_q_tensor(index, q);
      }
    }
  }
  
  return;
}

/*****************************************************************************
 *
 *  blue_set_random_q_rectangle_init
 *  Setting q tensor to isotropic in chosen area of the simulation box
 * 
 *****************************************************************************/

void blue_set_random_q_rectangle_init(const double xmin, const double xmax,
				      const double ymin, const double ymax,
				      const double zmin, const double zmax) {

  int i, j, k;
  int nlocal[3];
  int offset[3];
  int index;
  int ia, ib;

  double n[3];
  double q[3][3];
  double phase1, phase2;
  double amplitude_original;
  double amplitude_local;

  coords_nlocal(nlocal);
  coords_nlocal_offset(offset);
  
  /* get the original amplitude 
   * and set the new amplitude for
   * the local operation 
   */
  amplitude_original = blue_phase_amplitude();
  amplitude_local = 0.00001;
  blue_phase_amplitude_set(amplitude_local);
  
  for (i = 1; i<=N_total(X); i++) {
    for (j = 1; j<=N_total(Y); j++) {
      for (k = 1; k<=N_total(Z); k++) {

	if((i>xmin) && (i<xmax) &&
	   (j>ymin) && (j<ymax) &&
	   (k>zmin) && (k<zmax))
	  {
	    phase1 = pi_*(0.5 - ran_serial_uniform());
	    phase2 = pi_*(0.5 - ran_serial_uniform());
	    
	    /* Only set values if within local box */
	    if((i>offset[X]) && (i<=offset[X] + nlocal[X]) &&
	       (j>offset[Y]) && (j<=offset[Y] + nlocal[Y]) &&
	       (k>offset[Z]) && (k<=offset[Z] + nlocal[Z]))
	      {
		index = coords_index(i-offset[X], j-offset[Y], k-offset[Z]);
	      
		n[X] = cos(phase1)*sin(phase2);
		n[Y] = sin(phase1)*sin(phase2);
		n[Z] = cos(phase2);

		for (ia = 0; ia < 3; ia++) {
		  for (ib = 0; ib < 3; ib++) {
		    q[ia][ib] *= 0.00001;
		  }
		}

		blue_phase_q_uniaxial(n, q);
		phi_set_q_tensor(index, q);
	      }
	  }
      }
    }
  }

  /* set the amplitude to the original value */
  blue_phase_amplitude_set(amplitude_original);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_chirality
 *
 *  Return the chirality, which is defined here as
 *         sqrt(108 \kappa_0 q_0^2 / A_0 \gamma)
 *
 *  Not dependent on the redshift.
 *
 *****************************************************************************/

double blue_phase_chirality(void) {

  double chirality;

  chirality = sqrt(108.0*kappa0_*q0_*q0_ / (a0_*gamma_));

  return chirality;
}

/*****************************************************************************
 *
 *  blue_phase_reduced_temperature
 *
 *  Return the the reduced temperature defined here as
 *       27*(1 - \gamma/3) / \gamma
 *
 *****************************************************************************/

double blue_phase_reduced_temperature(void) {

  double tau;

  tau = 27.0*(1.0 - r3_*gamma_) / gamma_;

  return tau;
}

/*****************************************************************************
 *
 *  blue_phase_dimensionless_field_strength
 *
 *  Return the dimensionless field strength which is
 *      e^2 = (27 epsilon / 32 pi A_O gamma) E_a E_a
 *
 *****************************************************************************/

double blue_phase_dimensionless_field_strength(void) {

  int ia;
  double e;
  double fieldsq;

  fieldsq = 0.0;
  for (ia = 0; ia < 3; ia++) {
    fieldsq += electric_[ia]*electric_[ia];
  }

  /* Remember epsilon is stored with factor (1/12pi) */ 

  e = sqrt(27.0*(12.0*pi_*epsilon_)*fieldsq/(32.0*pi_*a0_*gamma_));

  return e;
}

/*****************************************************************************
 *
 *  blue_phase_redshift
 *
 *  Return the redshift parameter.
 *
 *****************************************************************************/

double blue_phase_redshift(void) {

  return redshift_;
}

/*****************************************************************************
 *
 *  blue_phase_redshift_set
 *
 *  Set the redshift parameter.
 *
 *****************************************************************************/

void blue_phase_redshift_set(const double redshift) {

  assert(fabs(redshift) >= redshift_min_);
  redshift_ = redshift;
  rredshift_ = 1.0/redshift_;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_redshift_update_set
 *
 *  At the moment the 'token' is on/off.
 *
 *****************************************************************************/

void blue_phase_redshift_update_set(int update) {

  redshift_update_ = update;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_redshift_compute
 *
 *  Redshift adjustment. If this is required at all, it should be
 *  done at every timestep. It gives rise to an Allreduce.
 *
 *  The redshift calculation uses the unredshifted values of the
 *  free energy parameters kappa0_, kappa1_ and q0_.
 *
 *  The term quadratic in gradients may be written F_ddQ
 *
 *     (1/2) [ kappa1 (d_a Q_bc)^2 - kappa1 (d_a Q_bc d_b Q_ac)
 *           + kappa0 (d_b Q_ab)^2 ]
 *
 *  The linear term is F_dQ
 *
 *     2 q0 kappa1 Q_ab e_acg d_c Q_gb
 *
 *  The new redshift is computed as - F_dQ / 2 F_ddQ
 *
 *****************************************************************************/

void blue_phase_redshift_compute(void) {

  int ic, jc, kc, index;
  int ia, ib, id, ig;
  int nlocal[3];

  double q[3][3], dq[3][3][3];

  double dq0, dq1, dq2, dq3, sum;
  double egrad_local[2], egrad[2];    /* Gradient terms for redshift calc. */
  double rnew;

  if (redshift_update_ == 0) return;

  coords_nlocal(nlocal);

  egrad_local[0] = 0.0;
  egrad_local[1] = 0.0;

  /* Accumulate the sums (all fluid) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	phi_get_q_tensor(index, q);
	phi_gradients_tensor_gradient(index, dq);

	/* kaapa0 (d_b Q_ab)^2 */

	dq0 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  sum = 0.0;
	  for (ib = 0; ib < 3; ib++) {
	    sum += dq[ib][ia][ib];
	  }
	  dq0 += sum*sum;
	}

	/* kappa1 (e_agd d_g Q_db + 2q_0 Q_ab)^2 */

	dq1 = 0.0;
	dq2 = 0.0;
	dq3 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    sum = 0.0;
	    for (ig = 0; ig < 3; ig++) {
	      dq1 += dq[ia][ib][ig]*dq[ia][ib][ig];
	      dq2 += dq[ia][ib][ig]*dq[ib][ia][ig];
	      for (id = 0; id < 3; id++) {
		sum += e_[ia][ig][id]*dq[ig][id][ib];
	      }
	    }
	    dq3 += q[ia][ib]*sum;
	  }
	}

	/* linear gradient and square gradient terms */

	egrad_local[0] += 2.0*q0_*kappa1_*dq3;
	egrad_local[1] += 0.5*(kappa1_*dq1 - kappa1_*dq2 + kappa0_*dq0);

      }
    }
  }

  /* Allreduce the gradient results, and compute a new redshift (we
   * keep the old one if problematic). */

  MPI_Allreduce(egrad_local, egrad, 2, MPI_DOUBLE, MPI_SUM, cart_comm());

  rnew = redshift_;
  if (egrad[1] != 0.0) rnew = -0.5*egrad[0]/egrad[1];
  if (fabs(rnew) < redshift_min_) rnew = redshift_;

  blue_phase_redshift_set(rnew);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_stats
 *
 *  This computes statistics for the free energy, and for the
 *  thermodynamic stress, if required. Remember that all the
 *  components of the stress have an additional minus sign cf.
 *  what may be expected.
 *
 *****************************************************************************/

void blue_phase_stats(int nstep) {

  int ic, jc, kc, index;
  int ia, ib, id, ig;
  int nlocal[3];

  double q0, kappa0, kappa1;
  double q[3][3], dq[3][3][3], dsq[3][3], h[3][3], sth[3][3];

  double q2, q3, dq0, dq1, sum;

  double elocal[14], etotal[14];        /* Free energy contributions etc */
  double rv;

  FILE * fp_output;

  coords_nlocal(nlocal);
  rv = 1.0/(L(X)*L(Y)*L(Z));

  /* Use current redshift. */

  q0 = q0_*rredshift_;
  kappa0 = kappa0_*redshift_*redshift_;
  kappa1 = kappa1_*redshift_*redshift_;

  for (ia = 0; ia < 14; ia++) {
    elocal[ia] = 0.0;
  }

  /* Accumulate the sums (all fluid) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	phi_get_q_tensor(index, q);
	phi_gradients_tensor_gradient(index, dq);
	phi_gradients_tensor_delsq(index, dsq);
  
	blue_phase_compute_h(q, dq, dsq, h);
	blue_phase_compute_stress(q, dq, h, sth);

	q2 = 0.0;

	/* Q_ab^2 */

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    q2 += q[ia][ib]*q[ia][ib];
	  }
	}

	/* Q_ab Q_bd Q_da */

	q3 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    for (id = 0; id < 3; id++) {
	      /* We use here the fact that q[id][ia] = q[ia][id] */
	      q3 += q[ia][ib]*q[ib][id]*q[ia][id];
	    }
	  }
	}

	/* (d_b Q_ab)^2 */

	dq0 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  sum = 0.0;
	  for (ib = 0; ib < 3; ib++) {
	    sum += dq[ib][ia][ib];
	  }
	  dq0 += sum*sum;
	}

	/* (e_agd d_g Q_db + 2q_0 Q_ab)^2 */

	dq1 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    sum = 0.0;
	    for (ig = 0; ig < 3; ig++) {
	      for (id = 0; id < 3; id++) {
		sum += e_[ia][ig][id]*dq[ig][id][ib];
	      }
	    }
	    sum += 2.0*q0*q[ia][ib];
	    dq1 += sum*sum;
	  }
	}

	/* Contributions bulk, kappa0, and kappa1 */

	elocal[0] += 0.5*a0_*(1.0 - r3_*gamma_)*q2;
	elocal[1] += -r3_*a0_*gamma_*q3;
	elocal[2] += 0.25*a0_*gamma_*q2*q2;
	elocal[3] += 0.5*kappa0*dq0;
	elocal[4] += 0.5*kappa1*dq1;

	/* Nine compoenents of stress */

	elocal[5]  += sth[X][X];
	elocal[6]  += sth[X][Y];
	elocal[7]  += sth[X][Z];
	elocal[8]  += sth[Y][X];
	elocal[9]  += sth[Y][Y];
	elocal[10] += sth[Y][Z];
	elocal[11] += sth[Z][X];
	elocal[12] += sth[Z][Y];
	elocal[13] += sth[Z][Z];
      }
    }
  }

  /* Results to standard out */

  MPI_Reduce(elocal, etotal, 14, MPI_DOUBLE, MPI_SUM, 0, cart_comm());

  for (ia = 0; ia < 14; ia++) {
    etotal[ia] *= rv;
  }

   if (output_to_file_ == 1) {

     /* Note that the reduction is to rank 0 in the Cartesian communicator */
     if (cart_rank() == 0) {

       fp_output = fopen("free_energy.dat", "a");
       if (fp_output == NULL) fatal("fopen(free_energy.dat) failed\n");

       /* timestep, total FE, gradient FE, redhsift */
       fprintf(fp_output, "%d %12.6le %12.6le %12.6le ", nstep, 
	       etotal[0] + etotal[1] + etotal[2] + etotal[3] + etotal[4],
	       etotal[3] + etotal[4], redshift_);
       /* Stress xx, xy, xz, ... */
       fprintf(fp_output, "%12.6le %12.6le %12.6le ",
	       etotal[5], etotal[6], etotal[7]);
       fprintf(fp_output, "%12.6le %12.6le %12.6le ",
	       etotal[8], etotal[9], etotal[10]);
       fprintf(fp_output, "%12.6le %12.6le %12.6le\n",
	       etotal[11], etotal[12], etotal[13]);
       
       fclose(fp_output);
     }
   }
   else {

     /* To standard output we send
      * 1. three terms in the bulk free energy
      * 2. two terms in distortion + current redshift
      * 3. total bulk, total distortion, and the grand total */

     info("\n");
     info("[fed1]%14d %14.7e %14.7e %14.7e\n", nstep, etotal[0],
	  etotal[1], etotal[2]);
     info("[fed2]%14d %14.7e %14.7e %14.7e\n", nstep, etotal[3], etotal[4],
	  redshift_);
     info("[fed3]%14d %14.7e %14.7e %14.7e\n", nstep,
	  etotal[0] + etotal[1] + etotal[2],
	  etotal[3] + etotal[4],
	  etotal[0] + etotal[1] + etotal[2] + etotal[3] + etotal[4]);
   }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_q0
 *
 *  Return the pitch wavenumber (unredshifted).
 *
 *****************************************************************************/

double blue_phase_q0(void) {

  return q0_;
}

/*****************************************************************************
 *
 *  blue_phase_electric_field_set
 *
 *****************************************************************************/

void blue_phase_electric_field_set(const double e[3]) {

  electric_[X] = e[X];
  electric_[Y] = e[Y];
  electric_[Z] = e[Z];

  return;
}

/*****************************************************************************
 *
 *  blue_phase_dielectric_anisotropy_set
 *
 *  Include the factor 1/12pi appearing in the free energy.
 *
 *****************************************************************************/

void blue_phase_dielectric_anisotropy_set(double e) {

  epsilon_ = (1.0/(12.0*pi_))*e;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_set_active_region_gamma_zeta
 *
 *  Set the parameters gamma_ and zeta_ for inside and outside 
 *
 *  the active region.
 *****************************************************************************/

void blue_phase_set_active_region_gamma_zeta(const int index) {
  
  double zeta_inside=0.0;
  double zeta_outside=0.0;
  
  double gamma_inside=3.0;
  double gamma_outside=2.4;

  /* check if we are inside/outside the active region */

  if ( coords_active_region(index) > 0.5 ){
    /*inside*/
    blue_phase_set_zeta(zeta_inside);
    blue_phase_set_gamma(gamma_inside);
  }
  else {
    /*outside*/
    blue_phase_set_zeta(zeta_outside);
    blue_phase_set_gamma(gamma_outside);
  }
  return;
}

/****************************************************************************
 *
 *  M_rot
 *
 *  Matrix for rotation around specified axis
 *
 ****************************************************************************/
void blue_phase_M_rot(double M[3][3], int dim, double alpha){

  if(dim==0){
    M[0][0] = 1.0;
    M[0][1] = 0.0;
    M[0][2] = 0.0;
    M[1][0] = 0.0;
    M[1][1] = cos(alpha);
    M[1][2] = -sin(alpha);
    M[2][0] = 0.0;
    M[2][1] = sin(alpha);
    M[2][2] = cos(alpha);
  }

  if(dim==1){
    M[0][0] = cos(alpha);
    M[0][1] = 0.0;
    M[0][2] = -sin(alpha);
    M[1][0] = 0.0;
    M[1][1] = 1.0;
    M[1][2] = 0.0;
    M[2][0] = sin(alpha);
    M[2][1] = 0.0;
    M[2][2] = cos(alpha);
  }

  if(dim==2){
    M[0][0] = cos(alpha);
    M[0][1] = -sin(alpha);
    M[0][2] = 0.0;
    M[1][0] = sin(alpha);
    M[1][1] = cos(alpha);
    M[1][2] = 0.0;
    M[2][0] = 0.0;
    M[2][1] = 0.0;
    M[2][2] = 1.0;
  }

  return;
}
