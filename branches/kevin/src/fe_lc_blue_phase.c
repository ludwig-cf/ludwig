/*****************************************************************************
 *
 *  fe_lc_blue_phase.c
 *
 *  Routines related to blue phase liquid crystal free energy
 *  and molecular field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2015 The University of Edinburgh
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "util.h"
#include "fe_s.h"
#include "fe_lc_blue_phase.h"

struct fe_lcbp_s {
  fe_t super;                     /* Implements fe_t */
  fe_lcbp_param_t * param;        /* Parameters */
  field_t * q;                    /* Tensor order parameter Q_ab */
  field_grad_t * dq;              /* Gradient d_a Q_bc */
  fe_lcbp_t * target;             /* Target copy */
};

static fe_vtable_t fe_lcbp_vtable = {
  (fe_free_ft)    fe_lcbp_free,
  (fe_fed_ft)     fe_lcbp_fed,
  (fe_mu_ft)      NULL,
  (fe_str_ft)     fe_lcbp_stress,
  (fe_mu_solv_ft) NULL,
  (fe_hvector_ft) NULL,
  (fe_htensor_ft) fe_lcbp_mol_field
};


/* To prevent numerical catastrophe, we impose a minimum redshift.
 * However, one should probably not be flirting with this value at
 * all in general usage. */

static const double redshift_min_ = 0.00000000001;

/*****************************************************************************
 *
 *  fe_lcbp_create
 *
 *****************************************************************************/

__host__ int fe_lcbp_create(field_t * q, field_grad_t * dq, fe_lcbp_t ** p) {

  fe_lcbp_t * obj = NULL;

  assert(q);
  assert(dq);

  obj = (fe_lcbp_t *) calloc(1, sizeof(fe_lcbp_t));
  if (obj == NULL) fatal("calloc(fe_lcbp_t) failed\n");

  obj->param = (fe_lcbp_param_t *) calloc(1, sizeof(fe_lcbp_param_t));
  if (obj->param == NULL) fatal("calloc(fe_lcbp_param_t) failed\n");

  obj->super.vtable = &fe_lcbp_vtable;
  obj->q = q;
  obj->dq = dq;

  *p = obj;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_free
 *
 *****************************************************************************/

__host__ int fe_lcbp_free(fe_lcbp_t * fe) {

  assert(fe);

  free(fe->param);
  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_param_set
 *
 *  The caller is responsible for all values.
 *
 *  Note that these values can remain unchanged throughout. Redshifted
 *  values are computed separately as needed.
 *
 *****************************************************************************/

__host__ int fe_lcbp_param_set(fe_lcbp_t * fe, fe_lcbp_param_t values) {

  assert(fe);

  *fe->param = values;

  /* The convention here is to non-dimensionalise the dielectric
   * anisotropy by factor (1/12pi) which appears in free energy. */

  fe->param->epsilon *= (1.0/(12.0*pi_));

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_param
 *
 *****************************************************************************/

__host__ __device__ int fe_lcbp_param(fe_lcbp_t * fe, fe_lcbp_param_t * vals) {

  assert(fe);
  assert(vals);

  *vals = *fe->param;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_fed
 *
 *  Return the free energy density at lattice site index.
 *
 *****************************************************************************/

__host__ __device__ int fe_lcbp_fed(fe_lcbp_t * fe, int index, double * fed) {

  double q[3][3];
  double dq[3][3][3];

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);

  fe_lcbp_compute_fed(fe, q, dq, fed);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_compute_fed
 *
 *  Compute the free energy density as a function of q and the q gradient
 *  tensor dq.
 *
 *****************************************************************************/

__host__ __device__ int fe_lcbp_compute_fed(fe_lcbp_t * fe, double q[3][3],
					    double dq[3][3][3], double * fed) {

  int ia, ib, ic, id;
  double a0, q0;
  double gamma;
  double kappa0, kappa1;
  double q2, q3;
  double dq0, dq1;
  double sum;
  double efield;

  assert(fe);
  assert(fed);

  q0 = fe->param->q0*fe->param->rredshift;
  kappa0 = fe->param->kappa0*fe->param->redshift*fe->param->redshift;
  kappa1 = fe->param->kappa1*fe->param->redshift*fe->param->redshift;

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
      efield += fe->param->electric[ia]*q[ia][ib]*fe->param->electric[ib];
    }
  }

  a0 = fe->param->a0;
  gamma = fe->param->gamma;

  *fed = 0.5*a0*(1.0 - r3_*gamma)*q2 - r3_*a0*gamma*q3 + 0.25*a0*gamma*q2*q2
    + 0.5*kappa0*dq0 + 0.5*kappa1*dq1
    - fe->param->epsilon*efield;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_stress
 *
 *  Return the stress sth[3][3] at lattice site index.
 *
 *****************************************************************************/

__host__ __device__ int fe_lcbp_stress(fe_lcbp_t * fe, int index,
				       double sth[3][3]) {

  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  assert(fe);

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);
  field_grad_tensor_delsq(fe->dq, index, dsq);

  fe_lcbp_compute_h(fe, q, dq, dsq, h);
  fe_lcbp_compute_stress(fe, q, dq, h, sth);

  return 0;
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

__host__ __device__ int fe_lcbp_compute_stress(fe_lcbp_t * fe, double q[3][3],
					       double dq[3][3][3],
					       double h[3][3],
					       double sth[3][3]) {
  int ia, ib, ic, id, ie;
  double fed;
  double q0;
  double kappa0;
  double kappa1;
  double qh;
  double p0;

  assert(fe);

  q0 = fe->param->q0*fe->param->rredshift;
  kappa0 = fe->param->kappa0*fe->param->redshift*fe->param->redshift;
  kappa1 = fe->param->kappa1*fe->param->redshift*fe->param->redshift;

  /* We have ignored the rho T term at the moment, assumed to be zero
   * (in particular, it has no divergence if rho = const). */

  fe_lcbp_compute_fed(fe, q, dq, &fed);
  p0 = 0.0 - fed;

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
      sth[ia][ib] = -p0*d_[ia][ib]
	+ 2.0*fe->param->xi*(q[ia][ib] + r3_*d_[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi and molecular field */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] +=
	  -fe->param->xi*h[ia][ic]*(q[ib][ic] + r3_*d_[ib][ic])
	  -fe->param->xi*(q[ia][ic] + r3_*d_[ia][ic])*h[ib][ic];
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

  /* Additional active stress -zeta*(q_ab - 1/3 d_ab) */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] -= fe->param->zeta*(q[ia][ib] + r3_*d_[ia][ib]);
    }
  }

  /* This is the minus sign. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	sth[ia][ib] = -sth[ia][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_mol_field
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *  Note this is only valid in the one-constant approximation at
 *  the moment (kappa0 = kappa1 = kappa).
 *
 *****************************************************************************/

__host__ __device__ int fe_lcbp_mol_field(fe_lcbp_t * fe, int index,
					  double h[3][3]) {

  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  assert(fe);
  assert(fe->param->kappa0 == fe->param->kappa1);

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);
  field_grad_tensor_delsq(fe->dq, index, dsq);

  fe_lcbp_compute_h(fe, q, dq, dsq, h);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_compute_h
 *
 *  Compute the molcular field h from q, the q gradient tensor dq, and
 *  the del^2 q tensor.
 *
 *****************************************************************************/

__host__ __device__
int fe_lcbp_compute_h(fe_lcbp_t * fe, double q[3][3], double dq[3][3][3],
		      double dsq[3][3], double h[3][3]) {

  int ia, ib, ic, id;

  double q0;              /* Redshifted value */
  double kappa0, kappa1;  /* Redshifted values */
  double q2;
  double e2;
  double eq;
  double sum;

  assert(fe);

  q0 = fe->param->q0*fe->param->rredshift;
  kappa0 = fe->param->kappa0*fe->param->redshift*fe->param->redshift;
  kappa1 = fe->param->kappa1*fe->param->redshift*fe->param->redshift;

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
      h[ia][ib] = -fe->param->a0*(1.0 - r3_*fe->param->gamma)*q[ia][ib]
	+ fe->param->a0*fe->param->gamma*(sum - r3_*q2*d_[ia][ib])
	- fe->param->a0*fe->param->gamma*q2*q[ia][ib];
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
      h[ia][ib] += kappa0*dsq[ia][ib]
	- 2.0*kappa1*q0*sum + 4.0*r3_*kappa1*q0*eq*d_[ia][ib]
	- 4.0*kappa1*q0*q0*q[ia][ib];
    }
  }

  /* Electric field term */

  e2 = 0.0;
  for (ia = 0; ia < 3; ia++) {
    e2 += fe->param->electric[ia]*fe->param->electric[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] +=  fe->param->epsilon
	*(fe->param->electric[ia]*fe->param->electric[ib] - r3_*d_[ia][ib]*e2);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_amplitude_compute
 *
 *  Scalar order parameter in the nematic state, minimum of bulk free energy 
 *
 *****************************************************************************/

__host__ __device__ int fe_lcbp_amplitude_compute(fe_lcbp_t * fe, double * a) {

  assert(fe);
  assert(a);
  
  *a = (2.0/3.0)*(0.25 + 0.75*sqrt(1.0 - 8.0/(3.0*fe->param->gamma)));

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_compute_bulk_fed
 *
 *  Compute the bulk free energy density as a function of q.
 *
 *  Note: This function contains also the part quadratic in q 
 *        which is normally part of the gradient free energy. 
 *
 *****************************************************************************/

__host__ __device__
int fe_lcbp_compute_bulk_fed(fe_lcbp_t * fe, double q[3][3], double * fed) {

  int ia, ib, ic;
  double q0;
  double kappa1;
  double q2, q3;

  assert(fe);

  q0 = fe->param->q0*fe->param->rredshift;
  kappa1 = fe->param->kappa1*fe->param->redshift*fe->param->redshift;

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

  *fed = 0.5*fe->param->a0*(1.0 - r3_*fe->param->gamma)*q2
    - r3_*fe->param->a0*fe->param->gamma*q3
    + 0.25*fe->param->a0*fe->param->gamma*q2*q2;

  /* Add terms quadratic in q from gradient free energy */ 

  *fed += 0.5*kappa1*4.0*q0*q0*q2;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_compute_gradient_fed
 *
 *  Compute the gradient contribution to the free energy density 
 *  as a function of q and the q gradient tensor dq.
 *
 *  Note: The part quadratic in q has been added to the bulk free energy.
 *
 *****************************************************************************/

__host__ __device__
int fe_lcbp_compute_gradient_fed(fe_lcbp_t * fe, double q[3][3],
				 double dq[3][3][3], double * fed) {

  int ia, ib, ic, id;
  double q0;
  double kappa0, kappa1;
  double dq0, dq1;
  double q2;
  double sum;

  assert(fe);

  q0 = fe->param->q0*fe->param->rredshift;
  kappa0 = fe->param->kappa0*fe->param->redshift*fe->param->redshift;
  kappa1 = fe->param->kappa1*fe->param->redshift*fe->param->redshift;

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
  q2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      sum = 0.0;
  
      q2 += q[ia][ib]*q[ia][ib];

      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum += e_[ia][ic][id]*dq[ic][ib][id];
	}
      }
      sum += 2.0*q0*q[ia][ib];
      dq1 += sum*sum;
    }
  }

  /* Subtract part that is quadratic in q */
  dq1 -= 4.0*q0*q0*q2;

  *fed = 0.5*kappa0*dq0 + 0.5*kappa1*dq1;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_chirality
 *
 *  Return the chirality, which is defined here as
 *         sqrt(108 \kappa_0 q_0^2 / A_0 \gamma)
 *
 *  Not dependent on the redshift.
 *
 *****************************************************************************/

__host__ __device__ int fe_lcbp_chirality(fe_lcbp_t * fe, double * chirality) {

  double a0;
  double gamma;
  double kappa0;
  double q0;

  assert(fe);
  assert(chirality);

  a0     = fe->param->a0;
  gamma  = fe->param->gamma;
  kappa0 = fe->param->kappa0;
  q0     = fe->param->q0;

  *chirality = sqrt(108.0*kappa0*q0*q0 / (a0*gamma));

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_reduced_temperature
 *
 *  Return the the reduced temperature defined here as
 *       27*(1 - \gamma/3) / \gamma
 *
 *****************************************************************************/

__host__ __device__
int fe_lcbp_reduced_temperature(fe_lcbp_t * fe, double * tau) {

  double gamma;

  assert(fe);
  assert(tau);

  gamma = fe->param->gamma;
  *tau = 27.0*(1.0 - r3_*gamma) / gamma;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_dimensionless_field_strength
 *
 *  Return the dimensionless field strength which is
 *      e^2 = (27 epsilon / 32 pi A_O gamma) E_a E_a
 *
 *****************************************************************************/

__host__ __device__
int fe_lcbp_dimensionless_field_strength(fe_lcbp_t * fe, double * ered) {

  int ia;
  double a0;
  double gamma;
  double epsilon;
  double fieldsq;

  assert(fe);

  fieldsq = 0.0;
  for (ia = 0; ia < 3; ia++) {
    fieldsq += fe->param->electric[ia]*fe->param->electric[ia];
  }

  /* Remember epsilon is stored with factor (1/12pi) */ 

  a0 = fe->param->a0;
  gamma = fe->param->gamma;
  epsilon = 12.0*pi_*fe->param->epsilon;

  *ered = sqrt(27.0*epsilon*fieldsq/(32.0*pi_*a0*gamma));

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_redshift
 *
 *  Return the redshift parameter.
 *
 *****************************************************************************/

__host__ __device__ int fe_lcbp_redshift(fe_lcbp_t * fe, double * redshift) {

  assert(fe);
  assert(redshift);

  *redshift = fe->param->redshift;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lcbp_redshift_set
 *
 *  Set the redshift parameter.
 *
 *****************************************************************************/

__host__ __device__
int fe_lcbp_redshift_set(fe_lcbp_t * fe,  double redshift) {

  assert(fe);
  assert(fabs(redshift) >= redshift_min_);

  fe->param->redshift = redshift;
  fe->param->rredshift = 1.0/redshift;

  return 0;
}
