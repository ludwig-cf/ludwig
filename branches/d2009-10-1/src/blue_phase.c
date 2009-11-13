/*****************************************************************************
 *
 *  blue_phase.c
 *
 *  Routines related to blue phase liquid crystal free energy
 *  and molecular field.
 *
 *  $Id: blue_phase.c,v 1.5.4.1 2009-11-13 17:23:11 jlintuvu Exp $
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
#include "phi.h"
#include "blue_phase.h"
#include "ran.h"

static double q0_;        /* Pitch = 2pi / q0_ */
static double a0_;        /* Bulk free energy parameter A_0 */
static double gamma_;     /* Controls magnitude of order */
static double kappa0_;    /* Elastic constant \kappa_0 */
static double kappa1_;    /* Elastic constant \kappa_1 */

static double xi_;        /* effective molecular aspect ratio (<= 1.0) */

static const double r3 = (1.0/3.0);

/*****************************************************************************
 *
 *  blue_phase_set_free_energy_parameters
 *
 *  Enforces the 'one constant approximation' kappa0 = kappa1 = kappa
 *
 *****************************************************************************/

void blue_phase_set_free_energy_parameters(double a0, double gamma,
					   double kappa, double q0) {
  a0_ = a0;
  gamma_ = gamma;
  kappa0_ = kappa;
  kappa1_ = kappa;
  q0_ = q0;

  return;
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
  phi_get_q_gradient_tensor(index, dq);

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
  double q2, q3;
  double dq0, dq1;
  double sum;

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
  phi_get_q_gradient_tensor(index, dq);
  phi_get_q_delsq_tensor(index, dsq);

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

  double q2;
  double eq;
  double sum;

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
  phi_get_q_gradient_tensor(index, dq);
  phi_get_q_delsq_tensor(index, dsq);

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
 *****************************************************************************/

void blue_phase_compute_stress(double q[3][3], double dq[3][3][3],
			       double h[3][3], double sth[3][3]) {

  int ia, ib, ic, id, ie;
  double qh;
  double p0;

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
      sth[ia][ib] = -p0*d_[ia][ib] + 2.0*xi_*(q[ia][ib] + r3*d_[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi and molecular field */

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
 *  blue_phase_O8M_init
 *
 *  Using the current free energy parameter q0_
 *
 *****************************************************************************/

void blue_phase_O8M_init(double amplitude) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double r2;
  double cosx, cosy, cosz, sinx, siny, sinz;

  get_N_local(nlocal);
  get_N_offset(noffset);

  r2 = sqrt(2.0);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

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

  return;
}

/*****************************************************************************
 *
 *  blue_phase_twist_init
 *  Setting a uniform twist along z-axis [cholesteric phase]
 *  Using the current free energy parameter q0_ (P=2pi/q0)
 * -Juho 12/11/09
 *****************************************************************************/

void blue_phase_twist_init(double amplitude){
  
  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double r2;
  double cosxy, cosz, sinxy,sinz;
  
  get_N_local(nlocal);
  get_N_offset(noffset);
  
  /* this corresponds to a 90 degree angle between the z-axis */
  cosz=0.0;
  sinz=1.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;
	
	index = get_site_index(ic, jc, kc);

	cosxy=cos(q0_*z);
	sinxy=sin(q0_*z);
	
	q[X][X] = amplitude*(3.0/2.0*sinz*sinz*cosxy*cosxy - 1.0/2.0);
	q[X][Y] = 3.0/2.0*amplitude*(sinz*sinz*cosxy*sinxy);
	q[X][Z] = 3.0/2.0*amplitude*(sinz*cosz*cosxy);
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude*(3.0/2.0*sinz*sinz*sinxy*sinxy - 1.0/2.0);
	q[Y][Z] = 3.0/2.0*amplitude*(sinz*cosz*sinxy);
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
 *  blue_set_random_q_init
 *  Setting q tensor to isotropic in chosen area of the simulation box
 * -Juho 12/11/09
 *****************************************************************************/

void blue_set_random_q_init(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax){
  
  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  
  double phase1,phase2;
  double amplitude,Pi;

  get_N_local(nlocal);
  get_N_offset(noffset);

  /* set amplitude to something small */
  amplitude = 0.0000001;
  
  Pi = atan(1.0)*4.0;
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    if(x < xmin || x > xmax)continue;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      if(y < ymin || y > ymax)continue;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;
	if(z < zmin || z > zmax)continue;

	index = get_site_index(ic, jc, kc);
	
	phase1= 2.0/5.0*Pi*(0.5-ran_parallel_uniform());
	phase2= Pi/2.0+Pi/5.0*(0.5-ran_parallel_uniform());
	
	q[X][X] = amplitude* (3.0/2.0*sin(phase2)*sin(phase2)*cos(phase1)*cos(phase1)-1.0/2.0);
	q[X][Y] = 3.0*amplitude/2.0*(sin(phase2)*sin(phase2)*cos(phase1)*sin(phase1));
	q[X][Z] = 3.0*amplitude/2.0*(sin(phase2)*cos(phase2)*cos(phase1));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude*(3.0/2.0*sin(phase2)*sin(phase2)*sin(phase1)*sin(phase1)-1.0/2.0);
	q[Y][Z] = 3.0*amplitude/2.0*(sin(phase2)*cos(phase2)*sin(phase1));
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
 *  blue_phase_chirality
 *
 *  Return the chirality, which is defined here as
 *         sqrt(108 \kappa_0 q_0^2 / A_0 \gamma)
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

  tau = 27.0*(1.0 - r3*gamma_) / gamma_;

  return tau;
}
