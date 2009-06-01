/*****************************************************************************
 *
 *  blue_phase.c
 *
 *  Routines related to blue phase liquid crystals.
 *
 *  $Id: blue_phase.c,v 1.2 2009-06-01 16:50:07 kevin Exp $
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
#include "phi.h"

/* Gamma = 0.3, k0 = 0.01, k1 = 0.01, a0 = 0.014384711 gamma = 3.1764706
 * q0 = numhalftwists*nunitcell*sqrt(2)*Pi/Ly 
 * with number of half twists in unit cell = 1.0,
 * number unit cells in pitch direction = 8 if Ly = 128
 * xi = 0.7 */

static double q0_;        /* Pitch = 2pi / q0_ */
static double a0_;        /* Bulk free energy parameter A_0 */
static double gamma_;     /* Controls magnitude of order */
static double kappa0_;    /* Elastic constant \kappa_0 */
static double kappa1_;    /* Elastic constant \kappa_1 */

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
  double tmp;

  phi_get_q_tensor(index, q);
  phi_get_q_gradient_tensor(index, dq);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	dq[ia][ib][ic] *= 3.0;
      }
    }
  }

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

  dq1 = 0.0;
  tmp = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum += e_[ia][ic][id]*dq[ic][id][ib];
	  tmp += e_[ia][ic][id]*dq[ic][id][ib]*q[ia][ib];
	}
      }
      sum += 2.0*q0_*q[ia][ib];
      dq1 += sum*sum;
    }
  }

  sum = 0.5*a0_*(1.0 - r3*gamma_)*q2 - r3*a0_*gamma_*q3
    + 0.25*a0_*gamma_*q2*q2 + 0.5*kappa0_*dq0 + 0.5*kappa1_*dq1;

  info("%13.6e %13.6e %13.6e %13.6e %13.6e\n", 
       0.5*a0_*(1.0 - r3*gamma_)*q2, -r3*a0_*gamma_*q3, 0.25*a0_*gamma_*q2*q2,
       2.0*kappa0_*q0_*tmp, 0.5*kappa1_*dq1 - 2.0*kappa1_*q0_*q0_*q2);

  info("%2d %2d %2d %13.6e %13.6e %13.6e %13.6e %13.6e\n",1,1,1, 
       dq[X][X][X], dq[X][X][Y], dq[X][X][Z], dq[X][Y][Y], dq[X][Y][Z]);
  info("%2d %2d %2d %13.6e %13.6e %13.6e %13.6e %13.6e\n",1,1,1, 
       dq[Y][X][X], dq[Y][X][Y], dq[Y][X][Z], dq[Y][Y][Y], dq[Y][Y][Z]);
  info("%2d %2d %2d %13.6e %13.6e %13.6e %13.6e %13.6e\n",1,1,1, 
       dq[Z][X][X], dq[Z][X][Y], dq[Z][X][Z], dq[Z][Y][Y], dq[Z][Y][Z]);

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

  /* CORRECTION for match against LCMain */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      dsq[ia][ib] *= 1.5;
      for (ic = 0; ic < 3; ic++) {
	dq[ia][ib][ic] *= 3.0;
      }
    }
  }

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
	+ a0_*gamma_*(sum - r3*q2*d_[ia][ib]) - a0_*gamma_*q2*q[ia][ib];
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

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum +=
	    (e_[ia][ic][id]*dq[ic][id][ib] + e_[ib][ic][id]*dq[ic][id][ia]);
	}
      }
      h[ia][ib] += kappa1_*dsq[ia][ib]
	- 2.0*kappa1_*q0_*sum + 4.0*r3*kappa1_*q0_*eq*d_[ia][ib]
	- 4.0*kappa1_*q0_*q0_*q[ia][ib];
    }
  }

  info("%13.6e %13.6e %13.6e %13.6e %13.6e\n", 
       h[X][X], h[X][Y], h[X][Z], h[Y][Y], h[Y][Z]);

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
  double p0 = 0.0;             /* isotropic pressure */
  double qh;

  assert(0); /* Sort out the isotropic pressure */

  phi_get_q_tensor(index, q);
  phi_get_q_gradient_tensor(index, dq);

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
      sth[ia][ib] = -p0*d_[ia][ib] + 2.0*xi_*(q[ia][ib] + r3*d_[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] +=
	  -xi_*h[ia][ic]*(q[ic][ib] + r3*d_[ic][ib])
	  -xi_*(q[ia][ic] + r3*d_[ia][ic])*h[ic][ib];
      }
    }
  }

  /* Dot product term d_a Q_cd . dF/dQ_cd,b */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sth[ia][ib] +=
	    - kappa0_*dq[ia][ic][ib]*dq[id][ic][id]
	    - kappa1_*dq[ia][ic][id]*dq[ib][ic][id]
	    + kappa1_*dq[ia][ic][id]*dq[ic][ib][id];

	  for (ie = 0; ie < 3; ie++) {
	    sth[ia][ib] +=
	      -2.0*kappa1_*q0_*dq[ia][ic][id]*e_[ib][ic][ie]*q[id][ie];
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
	sth[ia][ib] += q[ia][ic]*h[ic][ib] - h[ia][ic]*q[ic][ib];
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

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc < nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	phi_get_q_tensor(index, q);

	/* Velocity gradient tensor, symmetric and antisymmetric parts */

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

void blue_phase_O8M_init(void) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double r2;
  double cosx, cosy, cosz, sinx, siny, sinz;

  const double amplitude = -0.2;

  get_N_local(nlocal);
  get_N_offset(noffset);
  r2 = sqrt(2.0);

  /* Test value of q0_ = number half twists * number unit cells * pi
   *                     * sqrt(2.0 / L_y to match LCMain  */
  q0_ = 1.0*8.0*r2*4.0*atan(1.0)/L(Y);
  gamma_ = 3.1764706;
  kappa0_ = 0.01;
  kappa1_ = 0.01;
  a0_ = 0.014384711;

  /* The minus 1 in the coordinate positions here is to match with
   * the LC hybrid code. */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic - 1;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc - 1;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc - 1;

	index = get_site_index(ic, jc, kc);

	cosx = cos(r2*q0_*x);
	cosy = cos(r2*q0_*y);
	cosz = cos(r2*q0_*z);
	sinx = sin(r2*q0_*x);
	siny = sin(r2*q0_*y);
	sinz = sin(r2*q0_*z);

	q[X][X] = amplitude*(-2.0*cosy*sinz +    sinx*cosz + cosx*siny);
	q[X][Y] = amplitude*(  r2*cosy*cosz + r2*sinx*sinz - sinx*cosy);
	q[X][Z] = amplitude*(  r2*cosx*cosy + r2*sinz*siny - cosx*sinz);
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude*(-2.0*sinx*cosz +    siny*cosx + cosy*sinz);
	q[Y][Z] = amplitude*(  r2*cosz*cosx + r2*siny*sinx - siny*cosz);
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	phi_set_q_tensor(index, q);

      }
    }
  }

  phi_halo();
  phi_gradients_compute();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic - 1;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc - 1;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc - 1;

	index = get_site_index(ic, jc, kc);


	  info("%2d %2d %2d ", ic, jc, kc);
	  /* blue_phase_free_energy_density(index);*/
	  blue_phase_molecular_field(index, q);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_chirality
 *
 *  Return the chirality, which is defined here as
 *         sqrt(108 \kappa_0 q_0^2 / A_0 \gamma)
 *
 *****************************************************************************/

double blue_phase_chirality(void) {

  double chirality;

  chirality = sqrt(108.0*kappa0_*q0_*q0_ / a0_*gamma_);

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

  tau = 27.0*(1.0 - r3*gamma_) / gamma_;

  return tau;
}
