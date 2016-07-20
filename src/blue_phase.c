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
 *  (c) 2011-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "blue_phase.h"
#include "io_harness.h"
#include "leesedwards.h"
#include "physics.h"

static __constant__ fe_lc_param_t const_param;

/* To prevent numerical catastrophe, we impose a minimum redshift.
 * However, one should probably not be flirting with this value at
 * all in general usage. */

#define FE_REDSHIFT_MIN 0.00000000001

static fe_vt_t fe_hvt = {
  (fe_free_ft)      fe_lc_free,
  (fe_target_ft)    fe_lc_target,
  (fe_fed_ft)       fe_lc_fed,
  (fe_mu_ft)        NULL,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_lc_stress,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   fe_lc_mol_field,
  (fe_htensor_v_ft) fe_lc_compute_h_v
};

static __constant__ fe_vt_t fe_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_lc_fed,
  (fe_mu_ft)        NULL,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_lc_stress,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   fe_lc_mol_field,
  (fe_htensor_v_ft) fe_lc_compute_h_v
};

/*****************************************************************************
 *
 *  fe_lc_create
 *
 *****************************************************************************/

__host__ int fe_lc_create(field_t * q, field_grad_t * dq, fe_lc_t ** pobj) {

  int ndevice;
  fe_lc_t * fe = NULL;

  assert(q);
  assert(dq);
  assert(pobj);

  fe = (fe_lc_t *) calloc(1, sizeof(fe_lc_t));
  if (fe == NULL) fatal("calloc(fe_lc_t) failed\n");

  fe->param = (fe_lc_param_t *) calloc(1, sizeof(fe_lc_param_t));
  if (fe->param == NULL) fatal("");

  fe->q = q;
  fe->dq = dq;

  /* free energy interface functions */
  fe->super.func = &fe_hvt;
  fe->super.id = FE_LC;

  /* Allocate device memory, or alias */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe->target = fe;
  }
  else {
    fe_lc_param_t * tmp;
    fe_vt_t * vt;
    targetCalloc((void **) &fe->target, sizeof(fe_lc_t));
    targetConstAddress(&tmp, const_param);
    copyToTarget(&fe->target->param, tmp, sizeof(fe_lc_param_t *));
    targetConstAddress(&vt, fe_dvt);
    copyToTarget(&fe->target->super.func, &vt, sizeof(fe_vt_t *));
    assert(0); /* Requires a test */
  }

  *pobj = fe;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_free
 *
 *****************************************************************************/

__host__ int fe_lc_free(fe_lc_t * fe) {

  int ndevice;

  assert(fe);

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) targetFree(fe->target);

  free(fe->param);
  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_target
 *
 *  Commit the parameters as a kernel call may be imminent.
 *
 *****************************************************************************/

__host__ int fe_lc_target(fe_lc_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  fe_lc_param_commit(fe);
  *target = (fe_t *) fe->target;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_param_commit
 *
 *****************************************************************************/

__host__ int fe_lc_param_commit(fe_lc_t * fe) {

  assert(fe);

  copyConstToTarget(&const_param, fe->param, sizeof(fe_lc_param_t));

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_param_set
 *
 *  The caller is responsible for all values.
 *
 *  Note that these values can remain unchanged throughout. Redshifted
 *  values are computed separately as needed.
 *
 *****************************************************************************/

__host__ int fe_lc_param_set(fe_lc_t * fe, fe_lc_param_t values) {

  PI_DOUBLE(pi);

  assert(fe);

  *fe->param = values;

  /* The convention here is to non-dimensionalise the dielectric
   * anisotropy by factor (1/12pi) which appears in free energy. */

  fe->param->epsilon *= (1.0/(12.0*pi));

  /* Must compute reciprocal of redshift */
  fe_lc_redshift_set(fe, fe->param->redshift);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_param
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_param(fe_lc_t * fe, fe_lc_param_t * vals) {

  assert(fe);
  assert(vals);

  *vals = *fe->param;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_fed
 *
 *  Return the free energy density at lattice site index.
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_fed(fe_lc_t * fe, int index, double * fed) {

  double q[3][3];
  double dq[3][3][3];

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);

  fe_lc_compute_fed(fe, fe->param->gamma, q, dq, fed);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_compute_fed
 *
 *  Compute the free energy density as a function of q and the q gradient
 *  tensor dq.
 *
 *  NOTE: gamma is potentially gamma(r) so does not come from the
 *        fe->param
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_compute_fed(fe_lc_t * fe, double gamma,
					  double q[3][3],
					  double dq[3][3][3], double * fed) {

  int ia, ib, ic, id;
  double a0, q0;
  double kappa0, kappa1;
  double q2, q3;
  double dq0, dq1;
  double sum;
  double efield;
  const double r3 = 1.0/3.0;
  LEVI_CIVITA_CHAR(e);

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
	  sum += e[ia][ic][id]*dq[ic][ib][id];
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

  *fed = 0.5*a0*(1.0 - r3*gamma)*q2 - r3*a0*gamma*q3 + 0.25*a0*gamma*q2*q2
    + 0.5*kappa0*dq0 + 0.5*kappa1*dq1
    - fe->param->epsilon*efield;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_stress
 *
 *  Return the stress sth[3][3] at lattice site index.
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_stress(fe_lc_t * fe, int index,
				     double sth[3][3]) {

  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  assert(fe);

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);
  field_grad_tensor_delsq(fe->dq, index, dsq);

  fe_lc_compute_h(fe, fe->param->gamma, q, dq, dsq, h);
  fe_lc_compute_stress(fe, q, dq, h, sth);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_compute_stress
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

__host__ __device__ int fe_lc_compute_stress(fe_lc_t * fe, double q[3][3],
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
  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d);
  LEVI_CIVITA_CHAR(e);

  assert(fe);

  q0 = fe->param->q0*fe->param->rredshift;
  kappa0 = fe->param->kappa0*fe->param->redshift*fe->param->redshift;
  kappa1 = fe->param->kappa1*fe->param->redshift*fe->param->redshift;

  /* We have ignored the rho T term at the moment, assumed to be zero
   * (in particular, it has no divergence if rho = const). */

  fe_lc_compute_fed(fe, fe->param->gamma, q, dq, &fed);
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
      sth[ia][ib] = -p0*d[ia][ib]
	+ 2.0*fe->param->xi*(q[ia][ib] + r3*d[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi and molecular field */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] +=
	  -fe->param->xi*h[ia][ic]*(q[ib][ic] + r3*d[ib][ic])
	  -fe->param->xi*(q[ia][ic] + r3*d[ia][ic])*h[ib][ic];
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
	      -2.0*kappa1*q0*dq[ia][ic][id]*e[ib][ic][ie]*q[id][ie];
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
      sth[ia][ib] -= fe->param->zeta*(q[ia][ib] + r3*d[ia][ib]);
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
 *  fe_lc_mol_field
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *  Note this is only valid in the one-constant approximation at
 *  the moment (kappa0 = kappa1 = kappa).
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_mol_field(fe_lc_t * fe, int index,
					double h[3][3]) {

  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  assert(fe);
  assert(fe->param->kappa0 == fe->param->kappa1);

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);
  field_grad_tensor_delsq(fe->dq, index, dsq);

  fe_lc_compute_h(fe, fe->param->gamma, q, dq, dsq, h);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_compute_h
 *
 *  Compute the molcular field h from q, the q gradient tensor dq, and
 *  the del^2 q tensor.
 *
 *  NOTE: gamma is potentially gamma(r) so does not come from the
 *        fe->param
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_compute_h(fe_lc_t * fe, double gamma, double q[3][3],
		    double dq[3][3][3],
		    double dsq[3][3], double h[3][3]) {

  int ia, ib, ic, id;

  double q0;              /* Redshifted value */
  double kappa0, kappa1;  /* Redshifted values */
  double q2;
  double e2;
  double eq;
  double sum;
  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d);
  LEVI_CIVITA_CHAR(e);

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
      h[ia][ib] = -fe->param->a0*(1.0 - r3*gamma)*q[ia][ib]
	+ fe->param->a0*gamma*(sum - r3*q2*d[ia][ib])
	- fe->param->a0*gamma*q2*q[ia][ib];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  eq = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	eq += e[ib][ic][ia]*dq[ib][ic][ia];
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
	    (e[ia][ic][id]*dq[ic][ib][id] + e[ib][ic][id]*dq[ic][ia][id]);
	}
      }
      h[ia][ib] += kappa0*dsq[ia][ib]
	- 2.0*kappa1*q0*sum + 4.0*r3*kappa1*q0*eq*d[ia][ib]
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
	*(fe->param->electric[ia]*fe->param->electric[ib] - r3*d[ia][ib]*e2);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_compute_bulk_fed
 *
 *  Compute the bulk free energy density as a function of q.
 *
 *  Note: This function contains also the part quadratic in q 
 *        which is normally part of the gradient free energy. 
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_compute_bulk_fed(fe_lc_t * fe, double q[3][3], double * fed) {

  int ia, ib, ic;
  double q0;
  double kappa1;
  double q2, q3;
  const double r3 = 1.0/3.0;

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

  *fed = 0.5*fe->param->a0*(1.0 - r3*fe->param->gamma)*q2
    - r3*fe->param->a0*fe->param->gamma*q3
    + 0.25*fe->param->a0*fe->param->gamma*q2*q2;

  /* Add terms quadratic in q from gradient free energy */ 

  *fed += 0.5*kappa1*4.0*q0*q0*q2;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_compute_gradient_fed
 *
 *  Compute the gradient contribution to the free energy density 
 *  as a function of q and the q gradient tensor dq.
 *
 *  Note: The part quadratic in q has been added to the bulk free energy.
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_compute_gradient_fed(fe_lc_t * fe, double q[3][3],
			       double dq[3][3][3], double * fed) {

  int ia, ib, ic, id;
  double q0;
  double kappa0, kappa1;
  double dq0, dq1;
  double q2;
  double sum;
  LEVI_CIVITA_CHAR(e);

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
	  sum += e[ia][ic][id]*dq[ic][ib][id];
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
 *  fe_lc_chirality
 *
 *  Return the chirality, which is defined here as
 *         sqrt(108 \kappa_0 q_0^2 / A_0 \gamma)
 *
 *  Not dependent on the redshift.
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_chirality(fe_lc_t * fe, double * chirality) {

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
 *  fe_lc_reduced_temperature
 *
 *  Return the the reduced temperature defined here as
 *       27*(1 - \gamma/3) / \gamma
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_reduced_temperature(fe_lc_t * fe, double * tau) {

  double gamma;

  assert(fe);
  assert(tau);

  gamma = fe->param->gamma;
  *tau = 27.0*(1.0 - gamma/3.0) / gamma;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_dimensionless_field_strength
 *
 *  Return the dimensionless field strength which is
 *      e^2 = (27 epsilon / 32 pi A_O gamma) E_a E_a
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_dimensionless_field_strength(fe_lc_t * fe, double * ered) {

  int ia;
  double a0;
  double gamma;
  double epsilon;
  double fieldsq;
  PI_DOUBLE(pi);

  assert(fe);

  fieldsq = 0.0;
  for (ia = 0; ia < 3; ia++) {
    fieldsq += fe->param->electric[ia]*fe->param->electric[ia];
  }

  /* Remember epsilon is stored with factor (1/12pi) */ 

  a0 = fe->param->a0;
  gamma = fe->param->gamma;
  epsilon = 12.0*pi*fe->param->epsilon;

  *ered = sqrt(27.0*epsilon*fieldsq/(32.0*pi*a0*gamma));

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_redshift
 *
 *  Return the redshift parameter.
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_redshift(fe_lc_t * fe, double * redshift) {

  assert(fe);
  assert(redshift);

  *redshift = fe->param->redshift;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_redshift_set
 *
 *  Set the redshift parameter (host only).
 *
 *****************************************************************************/

__host__
int fe_lc_redshift_set(fe_lc_t * fe,  double redshift) {

  assert(fe);
  assert(fabs(redshift) >= FE_REDSHIFT_MIN);

  fe->param->redshift = redshift;
  fe->param->rredshift = 1.0/redshift;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_amplitude_compute
 *
 *  Scalar order parameter in the nematic state, minimum of bulk free energy 
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_amplitude_compute(fe_lc_param_t * param, double * a) {

  assert(a);
  
  *a = (2.0/3.0)*(0.25 + 0.75*sqrt(1.0 - 8.0/(3.0*param->gamma)));

  return 0;
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

__host__ __target__
int fe_lc_q_uniaxial(fe_lc_param_t * param, const double n[3], double q[3][3]) {

  int ia, ib;
  KRONECKER_DELTA_CHAR(d);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q[ia][ib] = 0.5*param->amplitude0*(3.0*n[ia]*n[ib] - d[ia][ib]);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_redshift_compute
 *
 *  Redshift adjustment. If this is required at all, it should be
 *  done at every timestep. It gives rise to an Allreduce.
 *
 *  The redshift calculation uses the unredshifted values of the
 *  free energy parameters kappa0, kappa1 and q0.
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

__host__ int fe_lc_redshift_compute(fe_lc_t * fe) {

  int ic, jc, kc, index;
  int ia, ib, id, ig;
  int nlocal[3];

  double q[3][3], dq[3][3][3];

  double dq0, dq1, dq2, dq3, sum;
  double egrad_local[2], egrad[2];    /* Gradient terms for redshift calc. */
  double rnew;
  LEVI_CIVITA_CHAR(e);

  if (fe->param->is_redshift_updated == 0) return 0;

  coords_nlocal(nlocal);

  egrad_local[0] = 0.0;
  egrad_local[1] = 0.0;

  /* Accumulate the sums (all fluid) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	field_tensor(fe->q, index, q);
	field_grad_tensor_grad(fe->dq, index, dq);

	/* kappa0 (d_b Q_ab)^2 */

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
		sum += e[ia][ig][id]*dq[ig][id][ib];
	      }
	    }
	    dq3 += q[ia][ib]*sum;
	  }
	}

	/* linear gradient and square gradient terms */

	egrad_local[0] += 2.0*fe->param->q0*fe->param->kappa1*dq3;
	egrad_local[1] += 0.5*(fe->param->kappa1*dq1 - fe->param->kappa1*dq2
			       + fe->param->kappa0*dq0);

      }
    }
  }

  /* Allreduce the gradient results, and compute a new redshift (we
   * keep the old one if problematic). */

  MPI_Allreduce(egrad_local, egrad, 2, MPI_DOUBLE, MPI_SUM, cart_comm());

  rnew = fe->param->redshift;
  if (egrad[1] != 0.0) rnew = -0.5*egrad[0]/egrad[1];
  if (fabs(rnew) < FE_REDSHIFT_MIN) rnew = fe->param->redshift;

  fe_lc_redshift_set(fe, rnew);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_scalar_ops
 *
 *  For symmetric traceless q[3][3], return the associated scalar
 *  order parameter, biaxial order parameter and director:
 *
 *  qs[0]  scalar order parameter: largest eigenvalue
 *  qs[1]  director[X] (associated eigenvector)
 *  qs[2]  director[Y]
 *  qs[3]  director[Z]
 *  qs[4]  biaxial order parameter b = sqrt(1 - 6 (Tr(QQQ))^2 / Tr(QQ)^3)
 *         related to the two largest eigenvalues...
 *
 *  If we write Q = ((s, 0, 0), (0, t, 0), (0, 0, -s -t)) then
 *
 *    Tr(QQ)  = s^2 + t^2 + (s + t)^2
 *    Tr(QQQ) = 3 s t (s + t)
 *
 *  If no diagonalisation is possible, all the results are set to zero.
 *
 *****************************************************************************/

__host__ int fe_lc_scalar_ops(double q[3][3], double qs[NQAB]) {

  int ifail;
  double eigenvalue[3];
  double eigenvector[3][3];
  double s, t;
  double q2, q3;

  ifail = util_jacobi_sort(q, eigenvalue, eigenvector);

  qs[0] = 0.0; qs[1] = 0.0; qs[2] = 0.0; qs[3] = 0.0; qs[4] = 0.0;

  if (ifail == 0) {

    qs[0] = eigenvalue[0];
    qs[1] = eigenvector[X][0];
    qs[2] = eigenvector[Y][0];
    qs[3] = eigenvector[Z][0];

    s = eigenvalue[0];
    t = eigenvalue[1];

    q2 = s*s + t*t + (s + t)*(s + t);
    q3 = 3.0*s*t*(s + t);
    qs[4] = sqrt(1 - 6.0*q3*q3 / (q2*q2*q2));
  }

  return ifail;
}




#ifdef OLD_SHIT

__targetHost__ __target__ void fed_loop_unrolled(double sum[VVL], double dq[3][3][3][VVL],
				double q[3][3][VVL],
				double dq1[VVL],
				  bluePhaseKernelConstants_t* pbpc);


/* Vectorized version */
__host__ __target__
void blue_phase_compute_fed_vec(double sum[VVL], double q[3][3][VVL], 
				double dq[3][3][3][VVL],
				bluePhaseKernelConstants_t* pbpc) {

  int iv=0;
  int ia, ib, ic;
  double q2[VVL], q3[VVL];
  double dq0[VVL], dq1[VVL];
  double efield[VVL];

  __targetILP__(iv) q2[iv] = 0.0;

  /* Q_ab^2 */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv)  q2[iv] += q[ia][ib][iv]*q[ia][ib][iv];
    }
  }

  /* Q_ab Q_bc Q_ca */

  __targetILP__(iv) q3[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	/* We use here the fact that q[ic][ia] = q[ia][ic] */
	__targetILP__(iv)  q3[iv] += q[ia][ib][iv]*q[ib][ic][iv]*q[ia][ic][iv];
      }
    }
  }

  /* (d_b Q_ab)^2 */

  __targetILP__(iv)  dq0[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    __targetILP__(iv)  sum[iv] = 0.0;
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv)  sum[iv] += dq[ib][ia][ib][iv];
    }
    __targetILP__(iv)  dq0[iv] += sum[iv]*sum[iv];
  }

  /* (e_acd d_c Q_db + 2q_0 Q_ab)^2 */
  /* With symmetric Q_db write Q_bd */

  __targetILP__(iv)  dq1[iv] = 0.0;

  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */
  /*     __targetILP__(iv)  sum[iv] = 0.0; */
  /*     for (ic = 0; ic < 3; ic++) { */
  /* 	for (id = 0; id < 3; id++) { */
  /* 	  __targetILP__(iv)  sum[iv] += pbpc->e_[ia][ic][id]*dq[ic][ib][id][iv]; */
  /* 	} */
  /*     } */
  /*     __targetILP__(iv)  sum[iv] += 2.0*pbpc->q0*q[ia][ib][iv]; */
  /*     __targetILP__(iv)  dq1[iv] += sum[iv]*sum[iv]; */
  /*   } */
  /* } */

  fed_loop_unrolled(sum,dq, q, dq1,pbpc);


  /* Electric field term (epsilon_ includes the factor 1/12pi) */

  __targetILP__(iv)  efield[iv] = 0.0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) {
	efield[iv] += pbpc->e0[ia]*q[ia][ib][iv]*pbpc->e0[ib];
      }
    }
  }

  __targetILP__(iv) {
    sum[iv] = 0.5*pbpc->a0_*(1.0 - pbpc->r3_*pbpc->gamma_)*q2[iv]
      - pbpc->r3_*pbpc->a0_*pbpc->gamma_*q3[iv]
      + 0.25*pbpc->a0_*pbpc->gamma_*q2[iv]*q2[iv]
      + 0.5*pbpc->kappa0*dq0[iv]
      + 0.5*pbpc->kappa1*dq1[iv]
      - pbpc->epsilon_*efield[iv];
  }

  return;
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

__targetHost__ double blue_phase_compute_bulk_fed(double q[3][3]) {

  int ia, ib, ic;
  double q0;
  double kappa1;
  double q2, q3;
  double sum;

  q0 = q0_*rredshift_;
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

  sum = 0.5*a0_*(1.0 - r3_*gamma_)*q2 - r3_*a0_*gamma_*q3 + 0.25*a0_*gamma_*q2*q2;

  /* Add terms quadratic in q from gradient free energy */ 
  sum += 0.5*kappa1*4.0*q0*q0*q2;

  return sum;
}

/*****************************************************************************
 *
 *  blue_phase_compute_gradient_fed
 *
 *  Compute the gradient contribution to the free energy density 
 *  as a function of q and the q gradient tensor dq.
 *
 *  Note: The part quadratic in q has been added to the bulk free energy.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_compute_gradient_fed(double q[3][3], 
						      double dq[3][3][3]) {

  int ia, ib, ic, id;
  double q0;
  double kappa0, kappa1;
  double dq0, dq1;
  double q2;
  double sum;

  q0 = q0_*rredshift_;
  kappa0 = kappa0_*redshift_*redshift_;
  kappa1 = kappa1_*redshift_*redshift_;

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

  sum = 0.5*kappa0*dq0 + 0.5*kappa1*dq1;

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

__targetHost__ void blue_phase_molecular_field(int index, double h[3][3]) {


  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  void* pcon = NULL;

  assert(kappa0_ == kappa1_);

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);
  field_grad_tensor_delsq(grad_q_, index, dsq);

  blue_phase_set_kernel_constants();
  blue_phase_host_constant_ptr(&pcon);
 
  blue_phase_compute_h(q, dq, dsq, h, (bluePhaseKernelConstants_t*) pcon);

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

__targetHost__ __target__ void blue_phase_compute_h(double q[3][3], 
						    double dq[3][3][3],
						    double dsq[3][3], 
						    double h[3][3],
						    bluePhaseKernelConstants_t* pbpc) {
  int ia, ib, ic, id;

  double q2;
  double e2;
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
      h[ia][ib] = -pbpc->a0_*(1.0 - pbpc->r3_*pbpc->gamma_)*q[ia][ib]
	+ pbpc->a0_*pbpc->gamma_*(sum - pbpc->r3_*q2*pbpc->d_[ia][ib]) - pbpc->a0_*pbpc->gamma_*q2*q[ia][ib];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  eq = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	eq += pbpc->e_[ib][ic][ia]*dq[ib][ic][ia];
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
	    (pbpc->e_[ia][ic][id]*dq[ic][ib][id] + pbpc->e_[ib][ic][id]*dq[ic][ia][id]);
	}
      }
      
      h[ia][ib] += pbpc->kappa0*dsq[ia][ib]
	- 2.0*pbpc->kappa1*pbpc->q0*sum + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq*pbpc->d_[ia][ib]
	- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[ia][ib];
    }
  }

  /* Electric field term */

  e2 = 0.0;
  for (ia = 0; ia < 3; ia++) {
    e2 += pbpc->e0[ia]*pbpc->e0[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] +=  pbpc->epsilon_*(pbpc->e0[ia]*pbpc->e0[ib] - pbpc->r3_*pbpc->d_[ia][ib]*e2);
    }
  }

  return;
}


__targetHost__ __target__ void h_loop_unrolled(double sum[VVL], double dq[3][3][3][VVL],
				double dsq[3][3][VVL],
				double q[3][3][VVL],
				double h[3][3][VVL],
				double eq[VVL],
				bluePhaseKernelConstants_t* pbpc);


/* vectorised version of above */
__targetHost__ __target__ void blue_phase_compute_h_vec(double q[3][3][VVL], 
						    double dq[3][3][3][VVL],
						    double dsq[3][3][VVL], 
						    double h[3][3][VVL],
						    bluePhaseKernelConstants_t* pbpc) {

  int iv=0;
  int ia, ib, ic;

  double q2[VVL];
  double e2[VVL];
  double eq[VVL];
  double sum[VVL];

  /* From the bulk terms in the free energy... */

  __targetILP__(iv) q2[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) q2[iv] += q[ia][ib][iv]*q[ia][ib][iv];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) sum[iv] = 0.0;
      for (ic = 0; ic < 3; ic++) {
	__targetILP__(iv) sum[iv] += q[ia][ic][iv]*q[ib][ic][iv];
      }
      __targetILP__(iv) h[ia][ib][iv] = -pbpc->a0_*(1.0 - pbpc->r3_*pbpc->gamma_)*q[ia][ib][iv]
	+ pbpc->a0_*pbpc->gamma_*(sum[iv] - pbpc->r3_*q2[iv]*pbpc->d_[ia][ib]) - pbpc->a0_*pbpc->gamma_*q2[iv]*q[ia][ib][iv];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  __targetILP__(iv) eq[iv] = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	__targetILP__(iv) eq[iv] += pbpc->e_[ib][ic][ia]*dq[ib][ic][ia][iv];
      }
    }
  }

  /* d_c Q_db written as d_c Q_bd etc */
  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */
  /*     __targetILP__(iv) sum[iv] = 0.0; */
  /*     for (ic = 0; ic < 3; ic++) { */
  /* 	for (id = 0; id < 3; id++) { */
  /* 	  __targetILP__(iv) sum[iv] += */
  /* 	    (pbpc->e_[ia][ic][id]*dq[ic][ib][id][iv] + pbpc->e_[ib][ic][id]*dq[ic][ia][id][iv]); */
  /* 	} */
  /*     } */
      
  /*     __targetILP__(iv) h[ia][ib][iv] += pbpc->kappa0*dsq[ia][ib][iv] */
  /* 	- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[ia][ib] */
  /* 	- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[ia][ib][iv]; */
  /*   } */
  /* } */

  h_loop_unrolled(sum,dq,dsq,q,h,eq,pbpc);

  /* Electric field term */

  __targetILP__(iv) e2[iv] = 0.0;
  for (ia = 0; ia < 3; ia++) {
    __targetILP__(iv) e2[iv] += pbpc->e0[ia]*pbpc->e0[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) h[ia][ib][iv] +=  pbpc->epsilon_*(pbpc->e0[ia]*pbpc->e0[ib] - pbpc->r3_*pbpc->d_[ia][ib]*e2[iv]);
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


__targetHost__ void blue_phase_chemical_stress(int index, double sth[3][3]) {

  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  void * pcon = NULL;

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);
  field_grad_tensor_delsq(grad_q_, index, dsq);

  blue_phase_set_kernel_constants();
  blue_phase_host_constant_ptr(&pcon);

  blue_phase_compute_h(q, dq, dsq, h, (bluePhaseKernelConstants_t*) pcon);
  blue_phase_compute_stress(q, dq, h, sth, (bluePhaseKernelConstants_t*) pcon);

  return;
}




/*targetDP development version */

 __target__ void blue_phase_chemical_stress_dev(int index, field_t* t_q, field_grad_t* t_q_grad, double* t_pth, void* pcon, int calledFromPhiForceStress) { 


  if (calledFromPhiForceStress != 1) {
#ifndef __NVCC__
    fatal("Error: in porting to targetDP we are assuming that blue_phase_chemical_stress is only called from phi_force_stress\n");
#endif  
  }
 
  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  double sth_loc[3][3];

  int ia, ib;

  bluePhaseKernelConstants_t* pbpc= (bluePhaseKernelConstants_t*) pcon;

  q[X][X] = t_q->data[addr_qab(tc_nSites, index, XX)];
  q[X][Y] = t_q->data[addr_qab(tc_nSites, index, XY)];
  q[X][Z] = t_q->data[addr_qab(tc_nSites, index, XZ)];
  q[Y][X] = q[X][Y];
  q[Y][Y] = t_q->data[addr_qab(tc_nSites, index, YY)];
  q[Y][Z] = t_q->data[addr_qab(tc_nSites, index, YZ)];
  q[Z][X] = q[X][Z];
  q[Z][Y] = q[Y][Z];
  q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];

  for (ia = 0; ia < NVECTOR; ia++) {
    dq[ia][X][X] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3, index, XX, ia)];
    dq[ia][X][Y] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3, index, XY, ia)];
    dq[ia][X][Z] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3, index, XZ, ia)];
    dq[ia][Y][X] = dq[ia][X][Y];
    dq[ia][Y][Y] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3, index, YY, ia)];
    dq[ia][Y][Z] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3, index, YZ, ia)];
    dq[ia][Z][X] = dq[ia][X][Z];
    dq[ia][Z][Y] = dq[ia][Y][Z];
    dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
  }

  dsq[X][X] = t_q_grad->delsq[addr_rank1(tc_nSites, NQAB, index, XX)];
  dsq[X][Y] = t_q_grad->delsq[addr_rank1(tc_nSites, NQAB, index, XY)];
  dsq[X][Z] = t_q_grad->delsq[addr_rank1(tc_nSites, NQAB, index, XZ)];
  dsq[Y][X] = dsq[X][Y];
  dsq[Y][Y] = t_q_grad->delsq[addr_rank1(tc_nSites, NQAB, index, YY)];
  dsq[Y][Z] = t_q_grad->delsq[addr_rank1(tc_nSites, NQAB, index, YZ)];
  dsq[Z][X] = dsq[X][Z];
  dsq[Z][Y] = dsq[Y][Z];
  dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];

  blue_phase_compute_h(q, dq, dsq, h, pbpc);
  blue_phase_compute_stress(q, dq, h, sth_loc, pbpc);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      t_pth[addr_rank2(tc_nSites,3,3,index,ia,ib)] = sth_loc[ia][ib];
    }
  }

  return;
}

/* vectorised version of the above */

__target__
void blue_phase_chemical_stress_dev_vec(int baseIndex,
					field_t* t_q,
					field_grad_t* t_q_grad,
					double* t_pth,
					void* pcon,
					int calledFromPhiForceStress) { 

   int iv=0;

  if (calledFromPhiForceStress != 1) {
#ifndef __NVCC__
    fatal("Error: in porting to targetDP we are assuming that blue_phase_chemical_stress is only called from phi_force_stress\n");
#endif  
  }
 
  double q[3][3][VVL];
  double h[3][3][VVL];
  double dq[3][3][3][VVL];
  double dsq[3][3][VVL];

  int ia;

  bluePhaseKernelConstants_t* pbpc= (bluePhaseKernelConstants_t*) pcon;

  __targetILP__(iv) q[X][X][iv] = t_q->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)];
  __targetILP__(iv) q[X][Y][iv] = t_q->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)];
  __targetILP__(iv) q[X][Z][iv] = t_q->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)];
  __targetILP__(iv) q[Y][X][iv] = q[X][Y][iv];
  __targetILP__(iv) q[Y][Y][iv] = t_q->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)];
  __targetILP__(iv) q[Y][Z][iv] = t_q->data[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)];
  __targetILP__(iv) q[Z][X][iv] = q[X][Z][iv];
  __targetILP__(iv) q[Z][Y][iv] = q[Y][Z][iv];
  __targetILP__(iv) q[Z][Z][iv] = 0.0 - q[X][X][iv] - q[Y][Y][iv];

  for (ia = 0; ia < NVECTOR; ia++) {
    __targetILP__(iv) dq[ia][X][X][iv] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3,baseIndex+iv,XX,ia)];
    __targetILP__(iv) dq[ia][X][Y][iv] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3,baseIndex+iv,XY,ia)];
    __targetILP__(iv) dq[ia][X][Z][iv] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3,baseIndex+iv,XZ,ia)];
    __targetILP__(iv) dq[ia][Y][X][iv] = dq[ia][X][Y][iv];
    __targetILP__(iv) dq[ia][Y][Y][iv] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3,baseIndex+iv,YY,ia)];
    __targetILP__(iv) dq[ia][Y][Z][iv] = t_q_grad->grad[addr_rank2(tc_nSites,NQAB,3,baseIndex+iv,YZ,ia)];
    __targetILP__(iv) dq[ia][Z][X][iv] = dq[ia][X][Z][iv];
    __targetILP__(iv) dq[ia][Z][Y][iv] = dq[ia][Y][Z][iv];
    __targetILP__(iv) dq[ia][Z][Z][iv] = 0.0 - dq[ia][X][X][iv] - dq[ia][Y][Y][iv];
  }


  __targetILP__(iv) dsq[X][X][iv] = t_q_grad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XX)];
  __targetILP__(iv) dsq[X][Y][iv] = t_q_grad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XY)];
  __targetILP__(iv) dsq[X][Z][iv] = t_q_grad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,XZ)];
  __targetILP__(iv) dsq[Y][X][iv] = dsq[X][Y][iv];
  __targetILP__(iv) dsq[Y][Y][iv] = t_q_grad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YY)];
  __targetILP__(iv) dsq[Y][Z][iv] = t_q_grad->delsq[addr_rank1(tc_nSites,NQAB,baseIndex+iv,YZ)];
  __targetILP__(iv) dsq[Z][X][iv] = dsq[X][Z][iv];
  __targetILP__(iv) dsq[Z][Y][iv] = dsq[Y][Z][iv];
  __targetILP__(iv) dsq[Z][Z][iv] = 0.0 - dsq[X][X][iv] - dsq[Y][Y][iv];

  blue_phase_compute_h_vec(q, dq, dsq, h, pbpc);
  blue_phase_compute_stress_vec(q, dq, h, t_pth, pbpc,baseIndex);

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

__host__ __target__
void blue_phase_compute_stress(double q[3][3], double dq[3][3][3],
			       double h[3][3], double sth[3][3], 
			       bluePhaseKernelConstants_t* pbpc) {
  int ia, ib, ic, id, ie;

  double qh;
  double p0;

  /* We have ignored the rho T term at the moment, assumed to be zero
     (in particular, it has no divergence if rho = const). */

  p0 = 0.0 - blue_phase_compute_fed(q, dq, pbpc);

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
      sth[ia][ib] = -p0*pbpc->d_[ia][ib] + 2.0*pbpc->xi_*(q[ia][ib]
  						 + pbpc->r3_*pbpc->d_[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi and molecular field */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
  	sth[ia][ib] +=
  	  -pbpc->xi_*h[ia][ic]*(q[ib][ic] + pbpc->r3_*pbpc->d_[ib][ic])
  	  -pbpc->xi_*(q[ia][ic] + pbpc->r3_*pbpc->d_[ia][ic])*h[ib][ic];
      }
    }
  }

  /* Dot product term d_a Q_cd . dF/dQ_cd,b */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      for (ic = 0; ic < 3; ic++) {
  	for (id = 0; id < 3; id++) {
  	  sth[ia][ib] +=
  	    - pbpc->kappa0*dq[ia][ib][ic]*dq[id][ic][id]
  	    - pbpc->kappa1*dq[ia][ic][id]*dq[ib][ic][id]
  	    + pbpc->kappa1*dq[ia][ic][id]*dq[ic][ib][id];

  	  for (ie = 0; ie < 3; ie++) {
  	    sth[ia][ib] +=
  	      -2.0*pbpc->kappa1*pbpc->q0*dq[ia][ic][id]*pbpc->e_[ib][ic][ie]*q[id][ie];
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
      sth[ia][ib] -= pbpc->zeta_*(q[ia][ib] + pbpc->r3_*pbpc->d_[ia][ib]);
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


__targetHost__ __target__ void stress_body_unrolled(double dq[3][3][3][VVL],
				double q[3][3][VVL],
				double h[3][3][VVL],
				     double* sth,
				double qh[VVL],
				double p0[VVL],
				     bluePhaseKernelConstants_t* pbpc, int baseIndex);

/* vectorised version of above */

__host__ __target__
void blue_phase_compute_stress_vec(double q[3][3][VVL],
				   double dq[3][3][3][VVL],
				   double h[3][3][VVL], double* sth, 
				   bluePhaseKernelConstants_t* pbpc,
				   int baseIndex) {
  int ia, ib;
  int iv=0;
  double qh[VVL];
  double p0[VVL];

  /* We have ignored the rho T term at the moment, assumed to be zero
     (in particular, it has no divergence if rho = const). */

  blue_phase_compute_fed_vec(p0, q, dq, pbpc);
  __targetILP__(iv) p0[iv] = 0. - p0[iv]; 

  /* The contraction Q_ab H_ab */

  __targetILP__(iv) qh[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) qh[iv] += q[ia][ib][iv]*h[ia][ib][iv];
    }
  }

  /* The term in the isotropic pressure, plus that in qh */

  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */
  /*     __targetILP__(iv) sth[ia][ib][iv] = -p0[iv]*pbpc->d_[ia][ib] + 2.0*pbpc->xi_*(q[ia][ib][iv] */
  /* 						 + pbpc->r3_*pbpc->d_[ia][ib])*qh[iv]; */
  /*   } */
  /* } */

  /* /\* Remaining two terms in xi and molecular field *\/ */

  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */
  /*     for (ic = 0; ic < 3; ic++) { */
  /* 	__targetILP__(iv) sth[ia][ib][iv] += */
  /* 	  -pbpc->xi_*h[ia][ic][iv]*(q[ib][ic][iv] + pbpc->r3_*pbpc->d_[ib][ic]) */
  /* 	  -pbpc->xi_*(q[ia][ic][iv] + pbpc->r3_*pbpc->d_[ia][ic])*h[ib][ic][iv]; */
  /*     } */
  /*   } */
  /* } */

  /* /\* Dot product term d_a Q_cd . dF/dQ_cd,b *\/ */

  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */

  /*     for (ic = 0; ic < 3; ic++) { */
  /* 	for (id = 0; id < 3; id++) { */
  /* 	  __targetILP__(iv) sth[ia][ib][iv] += */
  /* 	    - pbpc->kappa0*dq[ia][ib][ic][iv]*dq[id][ic][id][iv] */
  /* 	    - pbpc->kappa1*dq[ia][ic][id][iv]*dq[ib][ic][id][iv] */
  /* 	    + pbpc->kappa1*dq[ia][ic][id][iv]*dq[ic][ib][id][iv]; */

  /* 	  for (ie = 0; ie < 3; ie++) { */
  /* 	    __targetILP__(iv) sth[ia][ib][iv] += */
  /* 	      -2.0*pbpc->kappa1*pbpc->q0*dq[ia][ic][id][iv]*pbpc->e_[ib][ic][ie]*q[id][ie][iv]; */
  /* 	  } */
  /* 	} */
  /*     } */
  /*   } */
  /* } */

  /* /\* The antisymmetric piece q_ac h_cb - h_ac q_cb. We can */
  /*  * rewrite it as q_ac h_bc - h_ac q_bc. *\/ */

  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */
  /*     for (ic = 0; ic < 3; ic++) { */
  /* 	__targetILP__(iv) sth[ia][ib][iv] += q[ia][ic][iv]*h[ib][ic][iv] - h[ia][ic][iv]*q[ib][ic][iv]; */
  /*     } */
  /*   } */
  /* } */

  /* /\* Additional active stress -zeta*(q_ab - 1/3 d_ab) *\/ */

  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */
  /*     __targetILP__(iv) sth[ia][ib][iv] -= pbpc->zeta_*(q[ia][ib][iv] + pbpc->r3_*pbpc->d_[ia][ib]); */
  /*   } */
  /* } */

  /* /\* This is the minus sign. *\/ */

  /* for (ia = 0; ia < 3; ia++) { */
  /*   for (ib = 0; ib < 3; ib++) { */
  /* 	__targetILP__(iv) sth[ia][ib][iv] = -sth[ia][ib][iv]; */
  /*   } */
  /* } */

  stress_body_unrolled( dq, q, h, sth, qh, p0, pbpc, baseIndex);

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

__targetHost__ double blue_phase_chirality(void) {

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

__targetHost__ double blue_phase_reduced_temperature(void) {

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

__targetHost__ double blue_phase_dimensionless_field_strength(void) {

  int ia;
  double e;
  double fieldsq;
  double e0[3];

  physics_e0(e0);

  fieldsq = 0.0;
  for (ia = 0; ia < 3; ia++) {
    fieldsq += e0[ia]*e0[ia];
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

__targetHost__ double blue_phase_redshift(void) {

  return redshift_;
}

/*****************************************************************************
 *
 *  blue_phase_rredshift
 *
 *  Return the reciprocal redshift parameter.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_rredshift(void) {

  return rredshift_;
}
/*****************************************************************************
 *
 *  blue_phase_kappa0
 *
 *  Return the first elastic constant.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_kappa0(void) {

  return kappa0_;
}

/*****************************************************************************
 *
 *  blue_phase_kappa1
 *
 *  Return the second elastic constant.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_kappa1(void) {

  return kappa1_;
}

/*****************************************************************************
 *
 *  blue_phase_a0
 *
 *  Return the bulk free energy constant.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_a0(void) {

  return a0_;
}

/*****************************************************************************
 *
 *  blue_phase_gamma
 *
 *  Return the inversed effective temperature.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_gamma(void) {

  return gamma_;
}

/*****************************************************************************
 *
 *  blue_phase_q0
 *
 *  Return the pitch wavenumber (unredshifted).
 *
 *****************************************************************************/

__targetHost__ double blue_phase_q0(void) {

  return q0_;
}

/*****************************************************************************
 *
 *  blue_phase_redshift_set
 *
 *  Set the redshift parameter.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_redshift_set(const double redshift) {

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

__targetHost__ void blue_phase_redshift_update_set(int update) {

  redshift_update_ = update;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_dielectric_anisotropy_set
 *
 *  Include the factor 1/12pi appearing in the free energy.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_dielectric_anisotropy_set(double e) {

  epsilon_ = (1.0/(12.0*pi_))*e;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_dielectric_anisotropy
 *
 *
 *****************************************************************************/


__targetHost__ double blue_phase_dielectric_anisotropy(void) {

  return epsilon_;
}


/*****************************************************************************
 *
 *  blue_phase_set_active_region_gamma_zeta
 *
 *  Set the parameters gamma_ and zeta_ for inside and outside 
 *
 *  the active region.
 *****************************************************************************/

__targetHost__ void blue_phase_set_active_region_gamma_zeta(const int index) {
  
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

/*****************************************************************************
 *
 *  fed_io_info
 *
 *****************************************************************************/

__targetHost__ int fed_io_info(io_info_t ** info) {

  assert(info);

  *info = io_info_fed;

  return 0;
}

/*****************************************************************************
 *
 *  fed_io_info_set
 *
 *****************************************************************************/

__targetHost__ int fed_io_info_set(int grid[3], int form_out) {

  const char * name = "Free energy density";
  const char * stubname = "fed";

  assert(io_info_fed == NULL);
  
  io_info_fed = io_info_create_with_grid(grid);

  if (io_info_fed == NULL) fatal("io_info_create(fed) failed\n");
  
  io_info_set_name(io_info_fed, name);
  io_info_write_set(io_info_fed, IO_FORMAT_BINARY, fed_write);
  io_info_write_set(io_info_fed, IO_FORMAT_ASCII, fed_write_ascii);
  io_info_set_bytesize(io_info_fed, 3*sizeof(double));
 
  io_info_format_out_set(io_info_fed, form_out);
  io_info_metadata_filestub_set(io_info_fed, stubname);

  return 0;
}

/*****************************************************************************
 *
 *  fed_write_ascii
 *
 *  The "self" pointer is not required here.
 *
 *****************************************************************************/

static int fed_write_ascii(FILE * fp, int index, void * self) {


  int n;

  double q[3][3], dq[3][3][3];
  double fed[3];
  void * pcon = NULL;

  assert(fp);

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);

  blue_phase_set_kernel_constants();
  blue_phase_host_constant_ptr(&pcon);

  fed[0] = blue_phase_compute_fed(q, dq, (bluePhaseKernelConstants_t*) pcon);
  fed[1] = blue_phase_compute_bulk_fed(q);
  fed[2] = blue_phase_compute_gradient_fed(q, dq);

  n = fprintf(fp, "%22.15e  %22.15e  %22.15e\n", fed[0], fed[1], fed[2]);
  if (n < 0) fatal("fprintf(fed) failed at index %d\n", index);

  return n;
}

/*****************************************************************************
 *
 *  fed_write
 *
 *  The "self" object is not required.
 *
 *****************************************************************************/

static int fed_write(FILE * fp, int index, void * self) {

  int n;

  double q[3][3], dq[3][3][3];
  double fed[3];
  void * pcon = NULL;

  assert(fp);

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);

  blue_phase_set_kernel_constants();
  blue_phase_host_constant_ptr(&pcon);

  fed[0] = blue_phase_compute_fed(q, dq, (bluePhaseKernelConstants_t*) pcon);
  fed[1] = blue_phase_compute_bulk_fed(q);
  fed[2] = blue_phase_compute_gradient_fed(q, dq);

  n = fwrite(fed, sizeof(double), 3, fp);
  if (n != 3) fatal("fwrite(fed) failed at index %d\n", index);

  return n;
}

/* Unrolled kernels: thes get much beter performance since he multiplications
 by 0 and repeated loading of duplicate coefficients have been eliminated */


__targetHost__ __target__ void h_loop_unrolled(double sum[VVL], double dq[3][3][3][VVL],
				double dsq[3][3][VVL],
				double q[3][3][VVL],
				double h[3][3][VVL],
				double eq[VVL],
				bluePhaseKernelConstants_t* pbpc){

  int iv=0;

__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[1][0][2][iv] + dq[1][0][2][iv];
__targetILP__(iv) sum[iv] += -dq[2][0][1][iv] + -dq[2][0][1][iv];
__targetILP__(iv) h[0][0][iv] += pbpc->kappa0*dsq[0][0][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[0][0]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[0][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][0][2][iv];
__targetILP__(iv) sum[iv] += dq[1][1][2][iv] ;
__targetILP__(iv) sum[iv] += dq[2][0][0][iv];
__targetILP__(iv) sum[iv] += -dq[2][1][1][iv] ;
__targetILP__(iv) h[0][1][iv] += pbpc->kappa0*dsq[0][1][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[0][1]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[0][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][0][1][iv];
__targetILP__(iv) sum[iv] += -dq[1][0][0][iv];
__targetILP__(iv) sum[iv] += dq[1][2][2][iv] ;
__targetILP__(iv) sum[iv] += -dq[2][2][1][iv] ;
__targetILP__(iv) h[0][2][iv] += pbpc->kappa0*dsq[0][2][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[0][2]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[0][2][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][0][2][iv] ;
__targetILP__(iv) sum[iv] += dq[1][1][2][iv];
__targetILP__(iv) sum[iv] += dq[2][0][0][iv] ;
__targetILP__(iv) sum[iv] += -dq[2][1][1][iv];
__targetILP__(iv) h[1][0][iv] += pbpc->kappa0*dsq[1][0][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[1][0]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[1][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][1][2][iv] + -dq[0][1][2][iv];
__targetILP__(iv) sum[iv] += dq[2][1][0][iv] + dq[2][1][0][iv];
__targetILP__(iv) h[1][1][iv] += pbpc->kappa0*dsq[1][1][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[1][1]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[1][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][1][1][iv];
__targetILP__(iv) sum[iv] += -dq[0][2][2][iv] ;
__targetILP__(iv) sum[iv] += -dq[1][1][0][iv];
__targetILP__(iv) sum[iv] += dq[2][2][0][iv] ;
__targetILP__(iv) h[1][2][iv] += pbpc->kappa0*dsq[1][2][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[1][2]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[1][2][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][0][1][iv] ;
__targetILP__(iv) sum[iv] += -dq[1][0][0][iv] ;
__targetILP__(iv) sum[iv] += dq[1][2][2][iv];
__targetILP__(iv) sum[iv] += -dq[2][2][1][iv];
__targetILP__(iv) h[2][0][iv] += pbpc->kappa0*dsq[2][0][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[2][0]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[2][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][1][1][iv] ;
__targetILP__(iv) sum[iv] += -dq[0][2][2][iv];
__targetILP__(iv) sum[iv] += -dq[1][1][0][iv] ;
__targetILP__(iv) sum[iv] += dq[2][2][0][iv];
__targetILP__(iv) h[2][1][iv] += pbpc->kappa0*dsq[2][1][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[2][1]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[2][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][2][1][iv] + dq[0][2][1][iv];
__targetILP__(iv) sum[iv] += -dq[1][2][0][iv] + -dq[1][2][0][iv];
__targetILP__(iv) h[2][2][iv] += pbpc->kappa0*dsq[2][2][iv]
- 2.0*pbpc->kappa1*pbpc->q0*sum[iv] + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq[iv]*pbpc->d_[2][2]
- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[2][2][iv];

}


__targetHost__ __target__ void fed_loop_unrolled(double sum[VVL], double dq[3][3][3][VVL],
				double q[3][3][VVL],
				double dq1[VVL],
				bluePhaseKernelConstants_t* pbpc){


  int iv=0;

__targetILP__(iv) sum[iv] = 0.0;
  __targetILP__(iv) sum[iv] += dq[1][0][2][iv];

 __targetILP__(iv) sum[iv] -= dq[2][0][1][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[0][0][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];

__targetILP__(iv) sum[iv] = 0.0;
  __targetILP__(iv) sum[iv] += dq[1][1][2][iv];

 __targetILP__(iv) sum[iv] -= dq[2][1][1][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[0][1][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];

__targetILP__(iv) sum[iv] = 0.0;
  __targetILP__(iv) sum[iv] += dq[1][2][2][iv];

 __targetILP__(iv) sum[iv] -= dq[2][2][1][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[0][2][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];

__targetILP__(iv) sum[iv] = 0.0;
 __targetILP__(iv) sum[iv] -= dq[0][0][2][iv];

  __targetILP__(iv) sum[iv] += dq[2][0][0][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[1][0][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];

__targetILP__(iv) sum[iv] = 0.0;
 __targetILP__(iv) sum[iv] -= dq[0][1][2][iv];

  __targetILP__(iv) sum[iv] += dq[2][1][0][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[1][1][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];

__targetILP__(iv) sum[iv] = 0.0;
 __targetILP__(iv) sum[iv] -= dq[0][2][2][iv];

  __targetILP__(iv) sum[iv] += dq[2][2][0][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[1][2][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];

__targetILP__(iv) sum[iv] = 0.0;
  __targetILP__(iv) sum[iv] += dq[0][0][1][iv];

 __targetILP__(iv) sum[iv] -= dq[1][0][0][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[2][0][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];

__targetILP__(iv) sum[iv] = 0.0;
  __targetILP__(iv) sum[iv] += dq[0][1][1][iv];

 __targetILP__(iv) sum[iv] -= dq[1][1][0][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[2][1][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];

__targetILP__(iv) sum[iv] = 0.0;
  __targetILP__(iv) sum[iv] += dq[0][2][1][iv];

 __targetILP__(iv) sum[iv] -= dq[1][2][0][iv];

__targetILP__(iv) sum[iv] += 2.0*pbpc->q0*q[2][2][iv];

__targetILP__(iv) dq1[iv] += sum[iv]*sum[iv];


}

__targetHost__ __target__ void stress_body_unrolled(double dq[3][3][3][VVL],
				double q[3][3][VVL],
				double h[3][3][VVL],
				     double* sth,
				double qh[VVL],
				double p0[VVL],
				     bluePhaseKernelConstants_t* pbpc, int baseIndex){

  int iv=0;

double sthtmp[VVL];
double xiloc=pbpc->xi_;
double r3loc=pbpc->r3_;
double kappa0loc=pbpc->kappa0;
double kappa1loc=pbpc->kappa1;
double q0loc=pbpc->q0;
double zetaloc=pbpc->zeta_;
__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[0][0][iv]+ r3loc)*qh[iv] -p0[iv] ;

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][0][iv]*(q[0][0][iv] + r3loc)   -xiloc*(q[0][0][iv]    +r3loc)*h[0][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][1][iv]*(q[0][1][iv])   -xiloc*(q[0][1][iv]    )*h[0][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][2][iv]*(q[0][2][iv])   -xiloc*(q[0][2][iv]    )*h[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[0][0][0][iv]*dq[0][0][0][iv]+ kappa1loc*dq[0][0][0][iv]*dq[0][0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[0][0][1][iv]*dq[0][0][1][iv]+ kappa1loc*dq[0][0][1][iv]*dq[0][0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[0][0][2][iv]*dq[0][0][2][iv]+ kappa1loc*dq[0][0][2][iv]*dq[0][0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[0][1][0][iv]*dq[0][1][0][iv]+ kappa1loc*dq[0][1][0][iv]*dq[1][0][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][1][0][iv]*q[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[0][1][1][iv]*dq[0][1][1][iv]+ kappa1loc*dq[0][1][1][iv]*dq[1][0][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][1][1][iv]*q[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[0][1][2][iv]*dq[0][1][2][iv]+ kappa1loc*dq[0][1][2][iv]*dq[1][0][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][1][2][iv]*q[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[0][2][0][iv]*dq[0][2][0][iv]+ kappa1loc*dq[0][2][0][iv]*dq[2][0][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][2][0][iv]*q[0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[0][2][1][iv]*dq[0][2][1][iv]+ kappa1loc*dq[0][2][1][iv]*dq[2][0][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][2][1][iv]*q[1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][0][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[0][2][2][iv]*dq[0][2][2][iv]+ kappa1loc*dq[0][2][2][iv]*dq[2][0][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][2][2][iv]*q[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][0][iv]*h[0][0][iv] - h[0][0][iv]*q[0][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][1][iv]*h[0][1][iv] - h[0][1][iv]*q[0][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][2][iv]*h[0][2][iv] - h[0][2][iv]*q[0][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[0][0][iv] + r3loc);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,0,0)] = -sthtmp[iv];

__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[0][1][iv])*qh[iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][0][iv]*(q[1][0][iv])   -xiloc*(q[0][0][iv]    +r3loc)*h[1][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][1][iv]*(q[1][1][iv] + r3loc)   -xiloc*(q[0][1][iv]    )*h[1][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][2][iv]*(q[1][2][iv])   -xiloc*(q[0][2][iv]    )*h[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[0][0][0][iv]*dq[1][0][0][iv]+ kappa1loc*dq[0][0][0][iv]*dq[0][1][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][0][0][iv]*q[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[0][0][1][iv]*dq[1][0][1][iv]+ kappa1loc*dq[0][0][1][iv]*dq[0][1][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][0][1][iv]*q[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[0][0][2][iv]*dq[1][0][2][iv]+ kappa1loc*dq[0][0][2][iv]*dq[0][1][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][0][2][iv]*q[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[0][1][0][iv]*dq[1][1][0][iv]+ kappa1loc*dq[0][1][0][iv]*dq[1][1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[0][1][1][iv]*dq[1][1][1][iv]+ kappa1loc*dq[0][1][1][iv]*dq[1][1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[0][1][2][iv]*dq[1][1][2][iv]+ kappa1loc*dq[0][1][2][iv]*dq[1][1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[0][2][0][iv]*dq[1][2][0][iv]+ kappa1loc*dq[0][2][0][iv]*dq[2][1][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][2][0][iv]*q[0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[0][2][1][iv]*dq[1][2][1][iv]+ kappa1loc*dq[0][2][1][iv]*dq[2][1][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][2][1][iv]*q[1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][1][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[0][2][2][iv]*dq[1][2][2][iv]+ kappa1loc*dq[0][2][2][iv]*dq[2][1][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][2][2][iv]*q[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][0][iv]*h[1][0][iv] - h[0][0][iv]*q[1][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][1][iv]*h[1][1][iv] - h[0][1][iv]*q[1][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][2][iv]*h[1][2][iv] - h[0][2][iv]*q[1][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[0][1][iv]);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,0,1)] = -sthtmp[iv];

__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[0][2][iv])*qh[iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][0][iv]*(q[2][0][iv])   -xiloc*(q[0][0][iv]    +r3loc)*h[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][1][iv]*(q[2][1][iv])   -xiloc*(q[0][1][iv]    )*h[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[0][2][iv]*(q[2][2][iv] + r3loc)   -xiloc*(q[0][2][iv]    )*h[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[0][0][0][iv]*dq[2][0][0][iv]+ kappa1loc*dq[0][0][0][iv]*dq[0][2][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][0][0][iv]*q[0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[0][0][1][iv]*dq[2][0][1][iv]+ kappa1loc*dq[0][0][1][iv]*dq[0][2][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][0][1][iv]*q[1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[0][0][2][iv]*dq[2][0][2][iv]+ kappa1loc*dq[0][0][2][iv]*dq[0][2][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[0][0][2][iv]*q[2][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[0][1][0][iv]*dq[2][1][0][iv]+ kappa1loc*dq[0][1][0][iv]*dq[1][2][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][1][0][iv]*q[0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[0][1][1][iv]*dq[2][1][1][iv]+ kappa1loc*dq[0][1][1][iv]*dq[1][2][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][1][1][iv]*q[1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[0][1][2][iv]*dq[2][1][2][iv]+ kappa1loc*dq[0][1][2][iv]*dq[1][2][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[0][1][2][iv]*q[2][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[0][2][0][iv]*dq[2][2][0][iv]+ kappa1loc*dq[0][2][0][iv]*dq[2][2][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[0][2][1][iv]*dq[2][2][1][iv]+ kappa1loc*dq[0][2][1][iv]*dq[2][2][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[0][2][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[0][2][2][iv]*dq[2][2][2][iv]+ kappa1loc*dq[0][2][2][iv]*dq[2][2][2][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][0][iv]*h[2][0][iv] - h[0][0][iv]*q[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][1][iv]*h[2][1][iv] - h[0][1][iv]*q[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[0][2][iv]*h[2][2][iv] - h[0][2][iv]*q[2][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[0][2][iv]);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,0,2)] = -sthtmp[iv];

__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[1][0][iv])*qh[iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][0][iv]*(q[0][0][iv] + r3loc)   -xiloc*(q[1][0][iv]    )*h[0][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][1][iv]*(q[0][1][iv])   -xiloc*(q[1][1][iv]    +r3loc)*h[0][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][2][iv]*(q[0][2][iv])   -xiloc*(q[1][2][iv]    )*h[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[1][0][0][iv]*dq[0][0][0][iv]+ kappa1loc*dq[1][0][0][iv]*dq[0][0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[1][0][1][iv]*dq[0][0][1][iv]+ kappa1loc*dq[1][0][1][iv]*dq[0][0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[1][0][2][iv]*dq[0][0][2][iv]+ kappa1loc*dq[1][0][2][iv]*dq[0][0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[1][1][0][iv]*dq[0][1][0][iv]+ kappa1loc*dq[1][1][0][iv]*dq[1][0][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][1][0][iv]*q[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[1][1][1][iv]*dq[0][1][1][iv]+ kappa1loc*dq[1][1][1][iv]*dq[1][0][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][1][1][iv]*q[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[1][1][2][iv]*dq[0][1][2][iv]+ kappa1loc*dq[1][1][2][iv]*dq[1][0][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][1][2][iv]*q[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[1][2][0][iv]*dq[0][2][0][iv]+ kappa1loc*dq[1][2][0][iv]*dq[2][0][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][2][0][iv]*q[0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[1][2][1][iv]*dq[0][2][1][iv]+ kappa1loc*dq[1][2][1][iv]*dq[2][0][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][2][1][iv]*q[1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][0][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[1][2][2][iv]*dq[0][2][2][iv]+ kappa1loc*dq[1][2][2][iv]*dq[2][0][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][2][2][iv]*q[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][0][iv]*h[0][0][iv] - h[1][0][iv]*q[0][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][1][iv]*h[0][1][iv] - h[1][1][iv]*q[0][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][2][iv]*h[0][2][iv] - h[1][2][iv]*q[0][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[1][0][iv]);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,1,0)] = -sthtmp[iv];

__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[1][1][iv]+ r3loc)*qh[iv] -p0[iv] ;

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][0][iv]*(q[1][0][iv])   -xiloc*(q[1][0][iv]    )*h[1][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][1][iv]*(q[1][1][iv] + r3loc)   -xiloc*(q[1][1][iv]    +r3loc)*h[1][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][2][iv]*(q[1][2][iv])   -xiloc*(q[1][2][iv]    )*h[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[1][0][0][iv]*dq[1][0][0][iv]+ kappa1loc*dq[1][0][0][iv]*dq[0][1][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][0][0][iv]*q[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[1][0][1][iv]*dq[1][0][1][iv]+ kappa1loc*dq[1][0][1][iv]*dq[0][1][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][0][1][iv]*q[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[1][0][2][iv]*dq[1][0][2][iv]+ kappa1loc*dq[1][0][2][iv]*dq[0][1][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][0][2][iv]*q[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[1][1][0][iv]*dq[1][1][0][iv]+ kappa1loc*dq[1][1][0][iv]*dq[1][1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[1][1][1][iv]*dq[1][1][1][iv]+ kappa1loc*dq[1][1][1][iv]*dq[1][1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[1][1][2][iv]*dq[1][1][2][iv]+ kappa1loc*dq[1][1][2][iv]*dq[1][1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[1][2][0][iv]*dq[1][2][0][iv]+ kappa1loc*dq[1][2][0][iv]*dq[2][1][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][2][0][iv]*q[0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[1][2][1][iv]*dq[1][2][1][iv]+ kappa1loc*dq[1][2][1][iv]*dq[2][1][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][2][1][iv]*q[1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][1][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[1][2][2][iv]*dq[1][2][2][iv]+ kappa1loc*dq[1][2][2][iv]*dq[2][1][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][2][2][iv]*q[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][0][iv]*h[1][0][iv] - h[1][0][iv]*q[1][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][1][iv]*h[1][1][iv] - h[1][1][iv]*q[1][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][2][iv]*h[1][2][iv] - h[1][2][iv]*q[1][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[1][1][iv] + r3loc);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,1,1)] = -sthtmp[iv];

__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[1][2][iv])*qh[iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][0][iv]*(q[2][0][iv])   -xiloc*(q[1][0][iv]    )*h[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][1][iv]*(q[2][1][iv])   -xiloc*(q[1][1][iv]    +r3loc)*h[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[1][2][iv]*(q[2][2][iv] + r3loc)   -xiloc*(q[1][2][iv]    )*h[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[1][0][0][iv]*dq[2][0][0][iv]+ kappa1loc*dq[1][0][0][iv]*dq[0][2][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][0][0][iv]*q[0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[1][0][1][iv]*dq[2][0][1][iv]+ kappa1loc*dq[1][0][1][iv]*dq[0][2][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][0][1][iv]*q[1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[1][0][2][iv]*dq[2][0][2][iv]+ kappa1loc*dq[1][0][2][iv]*dq[0][2][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[1][0][2][iv]*q[2][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[1][1][0][iv]*dq[2][1][0][iv]+ kappa1loc*dq[1][1][0][iv]*dq[1][2][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][1][0][iv]*q[0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[1][1][1][iv]*dq[2][1][1][iv]+ kappa1loc*dq[1][1][1][iv]*dq[1][2][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][1][1][iv]*q[1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[1][1][2][iv]*dq[2][1][2][iv]+ kappa1loc*dq[1][1][2][iv]*dq[1][2][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[1][1][2][iv]*q[2][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[1][2][0][iv]*dq[2][2][0][iv]+ kappa1loc*dq[1][2][0][iv]*dq[2][2][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[1][2][1][iv]*dq[2][2][1][iv]+ kappa1loc*dq[1][2][1][iv]*dq[2][2][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[1][2][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[1][2][2][iv]*dq[2][2][2][iv]+ kappa1loc*dq[1][2][2][iv]*dq[2][2][2][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][0][iv]*h[2][0][iv] - h[1][0][iv]*q[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][1][iv]*h[2][1][iv] - h[1][1][iv]*q[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[1][2][iv]*h[2][2][iv] - h[1][2][iv]*q[2][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[1][2][iv]);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,1,2)] = -sthtmp[iv];

__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[2][0][iv])*qh[iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][0][iv]*(q[0][0][iv] + r3loc)   -xiloc*(q[2][0][iv]    )*h[0][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][1][iv]*(q[0][1][iv])   -xiloc*(q[2][1][iv]    )*h[0][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][2][iv]*(q[0][2][iv])   -xiloc*(q[2][2][iv]    +r3loc)*h[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[2][0][0][iv]*dq[0][0][0][iv]+ kappa1loc*dq[2][0][0][iv]*dq[0][0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[2][0][1][iv]*dq[0][0][1][iv]+ kappa1loc*dq[2][0][1][iv]*dq[0][0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[2][0][2][iv]*dq[0][0][2][iv]+ kappa1loc*dq[2][0][2][iv]*dq[0][0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[2][1][0][iv]*dq[0][1][0][iv]+ kappa1loc*dq[2][1][0][iv]*dq[1][0][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][1][0][iv]*q[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[2][1][1][iv]*dq[0][1][1][iv]+ kappa1loc*dq[2][1][1][iv]*dq[1][0][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][1][1][iv]*q[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[2][1][2][iv]*dq[0][1][2][iv]+ kappa1loc*dq[2][1][2][iv]*dq[1][0][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][1][2][iv]*q[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[2][2][0][iv]*dq[0][2][0][iv]+ kappa1loc*dq[2][2][0][iv]*dq[2][0][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][2][0][iv]*q[0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[2][2][1][iv]*dq[0][2][1][iv]+ kappa1loc*dq[2][2][1][iv]*dq[2][0][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][2][1][iv]*q[1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][0][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[2][2][2][iv]*dq[0][2][2][iv]+ kappa1loc*dq[2][2][2][iv]*dq[2][0][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][2][2][iv]*q[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][0][iv]*h[0][0][iv] - h[2][0][iv]*q[0][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][1][iv]*h[0][1][iv] - h[2][1][iv]*q[0][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][2][iv]*h[0][2][iv] - h[2][2][iv]*q[0][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[2][0][iv]);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,2,0)] = -sthtmp[iv];

__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[2][1][iv])*qh[iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][0][iv]*(q[1][0][iv])   -xiloc*(q[2][0][iv]    )*h[1][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][1][iv]*(q[1][1][iv] + r3loc)   -xiloc*(q[2][1][iv]    )*h[1][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][2][iv]*(q[1][2][iv])   -xiloc*(q[2][2][iv]    +r3loc)*h[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[2][0][0][iv]*dq[1][0][0][iv]+ kappa1loc*dq[2][0][0][iv]*dq[0][1][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][0][0][iv]*q[0][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[2][0][1][iv]*dq[1][0][1][iv]+ kappa1loc*dq[2][0][1][iv]*dq[0][1][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][0][1][iv]*q[1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[2][0][2][iv]*dq[1][0][2][iv]+ kappa1loc*dq[2][0][2][iv]*dq[0][1][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][0][2][iv]*q[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[2][1][0][iv]*dq[1][1][0][iv]+ kappa1loc*dq[2][1][0][iv]*dq[1][1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[2][1][1][iv]*dq[1][1][1][iv]+ kappa1loc*dq[2][1][1][iv]*dq[1][1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[2][1][2][iv]*dq[1][1][2][iv]+ kappa1loc*dq[2][1][2][iv]*dq[1][1][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[2][2][0][iv]*dq[1][2][0][iv]+ kappa1loc*dq[2][2][0][iv]*dq[2][1][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][2][0][iv]*q[0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[2][2][1][iv]*dq[1][2][1][iv]+ kappa1loc*dq[2][2][1][iv]*dq[2][1][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][2][1][iv]*q[1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][1][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[2][2][2][iv]*dq[1][2][2][iv]+ kappa1loc*dq[2][2][2][iv]*dq[2][1][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][2][2][iv]*q[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][0][iv]*h[1][0][iv] - h[2][0][iv]*q[1][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][1][iv]*h[1][1][iv] - h[2][1][iv]*q[1][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][2][iv]*h[1][2][iv] - h[2][2][iv]*q[1][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[2][1][iv]);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,2,1)] = -sthtmp[iv];

__targetILP__(iv) sthtmp[iv] = 2.0*xiloc*(q[2][2][iv]+ r3loc)*qh[iv] -p0[iv] ;

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][0][iv]*(q[2][0][iv])   -xiloc*(q[2][0][iv]    )*h[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][1][iv]*(q[2][1][iv])   -xiloc*(q[2][1][iv]    )*h[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += -xiloc*h[2][2][iv]*(q[2][2][iv] + r3loc)   -xiloc*(q[2][2][iv]    +r3loc)*h[2][2][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][0][iv]*dq[0][0][0][iv] - kappa1loc*dq[2][0][0][iv]*dq[2][0][0][iv]+ kappa1loc*dq[2][0][0][iv]*dq[0][2][0][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][0][0][iv]*q[0][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][0][iv]*dq[1][0][1][iv] - kappa1loc*dq[2][0][1][iv]*dq[2][0][1][iv]+ kappa1loc*dq[2][0][1][iv]*dq[0][2][1][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][0][1][iv]*q[1][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][0][iv]*dq[2][0][2][iv] - kappa1loc*dq[2][0][2][iv]*dq[2][0][2][iv]+ kappa1loc*dq[2][0][2][iv]*dq[0][2][2][iv];

      __targetILP__(iv) sthtmp[iv] -= 2.0*kappa1loc*q0loc*dq[2][0][2][iv]*q[2][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][1][iv]*dq[0][1][0][iv] - kappa1loc*dq[2][1][0][iv]*dq[2][1][0][iv]+ kappa1loc*dq[2][1][0][iv]*dq[1][2][0][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][1][0][iv]*q[0][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][1][iv]*dq[1][1][1][iv] - kappa1loc*dq[2][1][1][iv]*dq[2][1][1][iv]+ kappa1loc*dq[2][1][1][iv]*dq[1][2][1][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][1][1][iv]*q[1][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][1][iv]*dq[2][1][2][iv] - kappa1loc*dq[2][1][2][iv]*dq[2][1][2][iv]+ kappa1loc*dq[2][1][2][iv]*dq[1][2][2][iv];

      __targetILP__(iv) sthtmp[iv] += 2.0*kappa1loc*q0loc*dq[2][1][2][iv]*q[2][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][2][iv]*dq[0][2][0][iv] - kappa1loc*dq[2][2][0][iv]*dq[2][2][0][iv]+ kappa1loc*dq[2][2][0][iv]*dq[2][2][0][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][2][iv]*dq[1][2][1][iv] - kappa1loc*dq[2][2][1][iv]*dq[2][2][1][iv]+ kappa1loc*dq[2][2][1][iv]*dq[2][2][1][iv];

    __targetILP__(iv) sthtmp[iv] += - kappa0loc*dq[2][2][2][iv]*dq[2][2][2][iv] - kappa1loc*dq[2][2][2][iv]*dq[2][2][2][iv]+ kappa1loc*dq[2][2][2][iv]*dq[2][2][2][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][0][iv]*h[2][0][iv] - h[2][0][iv]*q[2][0][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][1][iv]*h[2][1][iv] - h[2][1][iv]*q[2][1][iv];

  __targetILP__(iv) sthtmp[iv] += q[2][2][iv]*h[2][2][iv] - h[2][2][iv]*q[2][2][iv];

__targetILP__(iv) sthtmp[iv] -= zetaloc*(q[2][2][iv] + r3loc);

 __targetILP__(iv) sth[addr_rank2(tc_nSites,3,3,baseIndex+iv,2,2)] = -sthtmp[iv];

}
#endif


/* KEVIN */

__host__ __target__ void h_loop_unrolled_be(fe_lc_t * fe,
					    double dq[3][3][3][VVL],
					    double dsq[3][3][VVL],
					    double q[3][3][VVL],
					    double h[3][3][VVL],
					    double eq[VVL],
					    double sum[NSIMDVL]);


/*IMPORTANT NOTE*/

/* required to be in scope here for performance reasons on GPU */
/* since otherwise the compiler places the emporary q[][][] etc arrays */
/* in regular off-chip memory rather than registers */
/* which has a huge impact on performance */

__host__ __target__
void fe_lc_compute_h_v(fe_lc_t * fe, double q[3][3][NSIMDVL], 
		       double dq[3][3][3][NSIMDVL],
		       double dsq[3][3][NSIMDVL], 
		       double h[3][3][NSIMDVL]) {

  int iv=0;
  int ia, ib, ic;
  double q0;
  double kappa0, kappa1;

  double q2[VVL];
  double e2[VVL];
  double eq[VVL];
  double sum[VVL];

  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d);
  LEVI_CIVITA_CHAR(e);

  /* Reshifted values */

  q0 = fe->param->rredshift*fe->param->q0;
  kappa0 = fe->param->redshift*fe->param->redshift*fe->param->kappa0;
  kappa1 = kappa0;

  /* From the bulk terms in the free energy... */

  __targetILP__(iv) q2[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) q2[iv] += q[ia][ib][iv]*q[ia][ib][iv];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) sum[iv] = 0.0;
      for (ic = 0; ic < 3; ic++) {
	__targetILP__(iv) sum[iv] += q[ia][ic][iv]*q[ib][ic][iv];
      }
      __targetILP__(iv) h[ia][ib][iv] =
	- fe->param->a0*(1.0 - r3*fe->param->gamma)*q[ia][ib][iv]
	+ fe->param->a0*fe->param->gamma*(sum[iv] - r3*q2[iv]*d[ia][ib])
	- fe->param->a0*fe->param->gamma*q2[iv]*q[ia][ib][iv];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  __targetILP__(iv) eq[iv] = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	__targetILP__(iv) eq[iv] += e[ib][ic][ia]*dq[ib][ic][ia][iv];
      }
    }
  }

  /* Contraction d_c Q_db ... */

  h_loop_unrolled_be(fe, dq, dsq, q, h, eq, sum);

  /* Electric field term */

  __targetILP__(iv) e2[iv] = 0.0;
  for (ia = 0; ia < 3; ia++) {
    __targetILP__(iv) {
      e2[iv] += fe->param->electric[ia]*fe->param->electric[ia];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) {
	h[ia][ib][iv] +=  fe->param->epsilon*
	  (fe->param->electric[ia]*fe->param->electric[ib] - r3*d[ia][ib]*e2[iv]);
      }
    }
  }

  return;

}


/* Unrolled kernels: thes get much beter performance since he multiplications
 by 0 and repeated loading of duplicate coefficients have been eliminated */

__target__ void h_loop_unrolled_be(fe_lc_t * fe, double dq[3][3][3][VVL],
				   double dsq[3][3][VVL],
				   double q[3][3][VVL],
				   double h[3][3][VVL],
				   double eq[VVL], double sum[NSIMDVL]) {

  int iv=0;
  double q0;
  double kappa0, kappa1;
  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d_);

  q0 = fe->param->rredshift*fe->param->q0;
  kappa0 = fe->param->redshift*fe->param->redshift*fe->param->kappa0;
  kappa1 = kappa0;

__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[1][0][2][iv] + dq[1][0][2][iv];
__targetILP__(iv) sum[iv] += -dq[2][0][1][iv] + -dq[2][0][1][iv];
__targetILP__(iv) h[0][0][iv] += kappa0*dsq[0][0][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[0][0] - 4.0*kappa1*q0*q0*q[0][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][0][2][iv];
__targetILP__(iv) sum[iv] += dq[1][1][2][iv] ;
__targetILP__(iv) sum[iv] += dq[2][0][0][iv];
__targetILP__(iv) sum[iv] += -dq[2][1][1][iv] ;
__targetILP__(iv) h[0][1][iv] += kappa0*dsq[0][1][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[0][1]- 4.0*kappa1*q0*q0*q[0][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][0][1][iv];
__targetILP__(iv) sum[iv] += -dq[1][0][0][iv];
__targetILP__(iv) sum[iv] += dq[1][2][2][iv] ;
__targetILP__(iv) sum[iv] += -dq[2][2][1][iv] ;
__targetILP__(iv) h[0][2][iv] += kappa0*dsq[0][2][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[0][2] - 4.0*kappa1*q0*q0*q[0][2][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][0][2][iv] ;
__targetILP__(iv) sum[iv] += dq[1][1][2][iv];
__targetILP__(iv) sum[iv] += dq[2][0][0][iv] ;
__targetILP__(iv) sum[iv] += -dq[2][1][1][iv];
__targetILP__(iv) h[1][0][iv] += kappa0*dsq[1][0][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[1][0] - 4.0*kappa1*q0*q0*q[1][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += -dq[0][1][2][iv] + -dq[0][1][2][iv];
__targetILP__(iv) sum[iv] += dq[2][1][0][iv] + dq[2][1][0][iv];
__targetILP__(iv) h[1][1][iv] += kappa0*dsq[1][1][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[1][1] - 4.0*kappa1*q0*q0*q[1][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][1][1][iv];
__targetILP__(iv) sum[iv] += -dq[0][2][2][iv] ;
__targetILP__(iv) sum[iv] += -dq[1][1][0][iv];
__targetILP__(iv) sum[iv] += dq[2][2][0][iv] ;
__targetILP__(iv) h[1][2][iv] += kappa0*dsq[1][2][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[1][2] - 4.0*kappa1*q0*q0*q[1][2][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][0][1][iv] ;
__targetILP__(iv) sum[iv] += -dq[1][0][0][iv] ;
__targetILP__(iv) sum[iv] += dq[1][2][2][iv];
__targetILP__(iv) sum[iv] += -dq[2][2][1][iv];
__targetILP__(iv) h[2][0][iv] += kappa0*dsq[2][0][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[2][0] - 4.0*kappa1*q0*q0*q[2][0][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][1][1][iv] ;
__targetILP__(iv) sum[iv] += -dq[0][2][2][iv];
__targetILP__(iv) sum[iv] += -dq[1][1][0][iv] ;
__targetILP__(iv) sum[iv] += dq[2][2][0][iv];
__targetILP__(iv) h[2][1][iv] += kappa0*dsq[2][1][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[2][1] - 4.0*kappa1*q0*q0*q[2][1][iv];
__targetILP__(iv) sum[iv] = 0.0;
__targetILP__(iv) sum[iv] += dq[0][2][1][iv] + dq[0][2][1][iv];
__targetILP__(iv) sum[iv] += -dq[1][2][0][iv] + -dq[1][2][0][iv];
__targetILP__(iv) h[2][2][iv] += kappa0*dsq[2][2][iv] - 2.0*kappa1*q0*sum[iv] + 4.0*r3*kappa1*q0*eq[iv]*d_[2][2] - 4.0*kappa1*q0*q0*q[2][2][iv];

 return;
}
