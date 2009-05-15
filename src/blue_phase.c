/*****************************************************************************
 *
 *  blue_phase.c
 *
 *  Routines related to blue phase liquid crystals.
 *
 *  $Id: blue_phase.c,v 1.1 2009-05-15 18:30:12 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "lattice.h"
#include "free_energy.h"

/* Gamma = 0.3, k0 = 0.01, k1 = 0.01, a0 = 0.014384711 gamma = 3.1764706
 * q0 = numhalftwists*nunitcell*sqrt(2)*Pi/Ly 
 * with number of half twists in unit cell = 1.0,
 * number unit cells in pitch direction = 8 if Ly = 128
 * xi = 0.7 */

static double q0_;        /* Pitch = 2pi / q0_ */
static double a0_;        /* A constant */
static double gamma_;     /* Controls magnitude of order */
static double kappa0_;    /* Elastic constant \kappa_0 */
static double kappa1_;    /* Elastic constant \kappa_1 */

static double p0_;
static double xi_;        /* effective molecular aspect ratio (<= 1.0) */
static double Gamma_;     /* Collective rotational diffusion constant */

static const double r3 = (1.0/3.0);

/*****************************************************************************
 *
 *  blue_phase_free_energy_density
 *
 *  Return the free energy density at lattice site index.
 *
 *****************************************************************************/

double blue_phase_free_energy_density(const int index) {

  int ia, ib, ic, id;
  double q[3][3];
  double dq[3][3][3];
  double q2, q3;
  double dq0, dq1;
  double sum;

  phi_get_q_tensor(index, q);
  phi_get_q_gradient_tensor(index, dq);

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
      sum += dq[ib][ia][ib]*dq[ib][ia][ib];
    }
    dq0 += sum*sum;
  }

  /* (e_acd d_c Q_db + 2q_0 Q_ab)^2 */

  dq1 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum += e_[ia][ic][id]*dq[ic][id][ib];
	}
      }
      sum += 2.0*q0_*q[ia][ib];
      dq1 += sum*sum;
    }
  }

  sum = 0.5*a0_*(1.0 - r3*gamma_)*q2 - r3*a0_*gamma_*q3
    + 0.25*a0_*gamma_*q2*q2 + 0.5*kappa0_*dq0 + 0.5*kappa1_*dq1;

  return sum;
}

/*****************************************************************************
 *
 *  blue_phase_molecular_field
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *****************************************************************************/

void blue_phase_molecular_field(int index, double h[3][3]) {

  int ia, ib, ic, id;

  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  double q2;
  double eq;
  double sum;

  phi_get_q_tensor(index, q);
  phi_get_q_gradient_tensor(index, dq);
  phi_get_q_delsq_tensor(index, dsq);

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
      h[ia][ib] = -a0_*(1.0 - r3*gamma_)*q[ia][ib]
	+ a0_*gamma_*(sum - r3*d_[ia][ib]*q2) - a0_*gamma_*q2*q[ia][ib];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  eq = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	assert(e_[ia][ib][ic] == e_[ib][ic][ia]);
	eq += e_[ib][ic][ia]*dq[ib][ic][ia];
      }
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum +=
	    (e_[ia][ic][id]*dq[ic][id][ib] + e_[ib][ic][id]*dq[ic][id][ia]);
	}
      }
      h[ia][ib] += -2.0*kappa1_*q0_*sum + 4.0*r3*kappa1_*q0_*d_[ia][ib]*eq
	- 4.0*kappa1_*q0_*q0_*q[ia][ib] + kappa0_*dsq[ia][ib];
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

  int ia, ib, ic, id, ie;

  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  /* double dfdq[3][3][3];*/
  double qh;

  qh = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      qh += q[ia][ib]*h[ia][ib];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] = -p0_*d_[ia][ib] + 2.0*xi_*(q[ia][ib] + r3*d_[ia][ib])*qh;
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] +=
	  -xi_*h[ia][ic]*(q[ic][ib] + r3*d_[ic][ib])
	  -xi_*(q[ia][ic] + r3*d_[ia][ic])*h[ic][ib]
	  + q[ia][ic]*h[ic][ib] - h[ia][ic]*q[ic][ib];
      }
    }
  }

  /* kappa0_ (?) and kappa1_ terms */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sth[ia][ib] +=
	    - kappa1_*dq[ia][ic][id]*dq[ib][ic][id]
	    + kappa1_*dq[ia][ic][id]*dq[ic][id][ib];
	  for (ie = 0; ie < 3; ie++) {
	    sth[ia][ib] +=
	      -2.0*kappa1_*q0_*dq[ia][ic][id]*e_[ie][ib][ic]*q[ie][id];
	  }
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_q_update
 *
 *  Update q via Euler forward step.
 *
 *****************************************************************************/

void blue_phase_update_q(void) {

  int ic, jc, kc;
  int ia, ib, id;
  int index;
  int nlocal[3];

  double q[3][3];
  double w[3][3];
  double d[3][3];
  double h[3][3];
  double s[3][3];
  double omega[3][3];
  double trace_qw;

  const double dt = 1.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc < nlocal[Z]; kc++) {

	/* Velocity gradient tensor, symmetric and antisymmetric parts */

	phi_get_q_tensor(index, q);
	hydrodynamics_velocity_gradient_tensor(ic, jc, kc, w);

	trace_qw = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  trace_qw += q[ia][ia]*w[ia][ia];
	  for (ib = 0; ib < 3; ib++) {
	    d[ia][ib]     = 0.5*(w[ia][ib] + w[ib][ia]);
	    omega[ia][ib] = 0.5*(w[ia][ib] - w[ib][ia]);
	  }
	}

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    s[ia][ib] = -2.0*xi_*(q[ia][ib] + r3*d_[ia][ib])*trace_qw;
	    for (id = 0; id < 3; id++) {
	      s[ia][ib] +=
		(xi_*d[ia][id] + omega[ia][id])*(q[id][ib] + r3*d_[id][ib])
	      + (q[ia][id] + r3*d_[ia][id])*(xi_*d[id][ib] - omega[id][ib]);
	    }
	  }
	}

	/* No advective piece yet. */

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    q[ia][ib] += dt*(s[ia][ib] + Gamma_*h[ia][ib]);
	  }
	}

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_O8M_init
 *
 *****************************************************************************/

void blue_phase_O8M_struct(void) {



  return;
}

