/*****************************************************************************
 *
 *  fe_lc.c
 *
 *  Routines related to blue phase liquid crystal free energy
 *  and molecular field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "physics.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "blue_phase.h"

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
  (fe_str_ft)       fe_lc_str_symm,
  (fe_str_ft)       fe_lc_str_anti,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   fe_lc_mol_field,
  (fe_htensor_v_ft) fe_lc_mol_field_v,
  (fe_stress_v_ft)  fe_lc_stress_v,
  (fe_stress_v_ft)  fe_lc_str_symm_v,
  (fe_stress_v_ft)  fe_lc_str_anti_v
};

static __constant__ fe_vt_t fe_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_lc_fed,
  (fe_mu_ft)        NULL,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_lc_stress,
  (fe_str_ft)       fe_lc_str_symm,
  (fe_str_ft)       fe_lc_str_anti,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   fe_lc_mol_field,
  (fe_htensor_v_ft) fe_lc_mol_field_v,
  (fe_stress_v_ft)  fe_lc_stress_v,
  (fe_stress_v_ft)  fe_lc_str_symm_v,
  (fe_stress_v_ft)  fe_lc_str_anti_v
};


/*****************************************************************************
 *
 *  fe_lc_create
 *
 *  "le" only used in initialisation of field p, so may be NULL
 *
 *****************************************************************************/

__host__ int fe_lc_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			  field_t * q, field_grad_t * dq, fe_lc_t ** pobj) {

  int ndevice;
  int nhalo;
  fe_lc_t * fe = NULL;

  assert(pe);
  assert(cs);
  assert(q);
  assert(dq);
  assert(pobj);

  fe = (fe_lc_t *) calloc(1, sizeof(fe_lc_t));
  assert(fe);
  if (fe == NULL) pe_fatal(pe, "calloc(fe_lc_t) failed\n");

  fe->param = (fe_lc_param_t *) calloc(1, sizeof(fe_lc_param_t));
  assert(fe->param);
  if (fe->param == NULL) pe_fatal(pe, "calloc(fe_lc_param_t) failed\n");

  fe->pe = pe;
  fe->cs = cs;
  fe->q = q;
  fe->dq = dq;

  /* Additional active stress field "p" */

  cs_nhalo(fe->cs, &nhalo);
  field_create(fe->pe, fe->cs, 3, "Active P", &fe->p);
  field_init(fe->p, nhalo, le);
  field_grad_create(fe->pe, fe->p, 2, &fe->dp);

  /* free energy interface functions */
  fe->super.func = &fe_hvt;
  fe->super.id = FE_LC;

  /* Allocate device memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe->target = fe;
  }
  else {
    fe_lc_param_t * tmp;
    fe_vt_t * vt;

    tdpAssert(tdpMalloc((void **) &fe->target, sizeof(fe_lc_t)));
    tdpAssert(tdpMemset(fe->target, 0, sizeof(fe_lc_t)));

    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));

    tdpAssert(tdpMemcpy(&fe->target->param, &tmp, sizeof(fe_lc_param_t *),
			tdpMemcpyHostToDevice));
    tdpGetSymbolAddress((void **) &vt, tdpSymbol(fe_dvt));
    tdpAssert(tdpMemcpy(&fe->target->super.func, &vt, sizeof(fe_vt_t *),
			tdpMemcpyHostToDevice));

    /* Q_ab, gradient */
    tdpAssert(tdpMemcpy(&fe->target->q, &q->target, sizeof(field_t *),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&fe->target->dq, &dq->target, sizeof(field_grad_t *),
			tdpMemcpyHostToDevice));
    /* Active stress */
    tdpAssert(tdpMemcpy(&fe->target->p, &fe->p->target, sizeof(field_t *),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&fe->target->dp, &fe->dp->target,
			sizeof(field_grad_t *), tdpMemcpyHostToDevice));
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

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) tdpAssert(tdpFree(fe->target));

  if (fe->dp) field_grad_free(fe->dp);
  if (fe->p) field_free(fe->p);

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
 *  Includes time-dependent electric field.
 *
 *****************************************************************************/

__host__ int fe_lc_param_commit(fe_lc_t * fe) {

  int ia;
  double e0_freq, t;
  double e0[3];
  physics_t * phys = NULL;
  PI_DOUBLE(pi);

  assert(fe);

  physics_ref(&phys);
  physics_e0(phys, e0);
  physics_e0_frequency(phys, &e0_freq);
  physics_control_time(phys, &t);
  for (ia = 0; ia < 3; ia++) {
    fe->param->e0coswt[ia] = cos(2.0*pi*e0_freq*t)*e0[ia];
  }

  tdpMemcpyToSymbol(tdpSymbol(const_param), fe->param, sizeof(fe_lc_param_t),
		    0, tdpMemcpyHostToDevice);

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
      efield += fe->param->e0coswt[ia]*q[ia][ib]*fe->param->e0coswt[ib];
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
  assert(fe->q);
  assert(fe->dq);

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);
  field_grad_tensor_delsq(fe->dq, index, dsq);

  fe_lc_compute_h(fe, fe->param->gamma, q, dq, dsq, h);
  fe_lc_compute_stress(fe, q, dq, h, sth);

  if (fe->param->is_active) {
    int ia, ib;
    double dp[3][3];
    double sa[3][3];
    field_grad_vector_grad(fe->dp, index, dp);
    fe_lc_compute_stress_active(fe, q, dp, sa);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	sth[ia][ib] += sa[ia][ib];
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_str_symm
 *
 *  Symmetric stress. Easier to compute the total, and take off
 *  the antisymmtric part.
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_str_symm(fe_lc_t * fe, int index,
				       double s[3][3]) {
  int ia, ib, ic;
  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  assert(fe);
  assert(fe->q);
  assert(fe->dq);

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);
  field_grad_tensor_delsq(fe->dq, index, dsq);

  fe_lc_compute_h(fe, fe->param->gamma, q, dq, dsq, h);
  fe_lc_compute_stress(fe, q, dq, h, s);

  if (fe->param->is_active) {
    int ia, ib;
    double dp[3][3];
    double sa[3][3];
    field_grad_vector_grad(fe->dp, index, dp);
    fe_lc_compute_stress_active(fe, q, dp, sa);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib] += sa[ia][ib];
      }
    }
  }

  /* Antisymmetric part is subtracted (added, with the -ve sign) */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	s[ia][ib] += q[ia][ic]*h[ib][ic] - h[ia][ic]*q[ib][ic];
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_str_anti
 *
 *  Antisymmetric part of the stress.
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_str_anti(fe_lc_t * fe, int index,
				       double s[3][3]) {

  int ia, ib, ic;
  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  assert(fe);

  field_tensor(fe->q, index, q);
  field_grad_tensor_grad(fe->dq, index, dq);
  field_grad_tensor_delsq(fe->dq, index, dsq);

  fe_lc_compute_h(fe, fe->param->gamma, q, dq, dsq, h);

  /* The antisymmetric piece q_ac h_cb - h_ac q_cb. We can
   * rewrite it as q_ac h_bc - h_ac q_bc. */
  /* With minus sign */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = 0.0;
      for (ic = 0; ic < 3; ic++) {
	s[ia][ib] -= (q[ia][ic]*h[ib][ic] - h[ia][ic]*q[ib][ic]);
      }
    }
  }

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
 *  fe_lc_compute_stress_active
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_compute_stress_active(fe_lc_t * fe,
						    double q[3][3],
						    double dp[3][3],
						    double s[3][3]) {
  int ia, ib;
  KRONECKER_DELTA_CHAR(d);

  /* Previously comment said: -zeta*(q_ab - 1/3 d_ab)
   * while code was           -zeta*(q[ia][ib] + r3*d[ia][ib])
   * for zeta = zeta1 */
  /* The sign of zeta0 needs to be clarified cf Eq. 36 of notes */
  /* For "backwards compatability" use zeta0 = +1/3 at the moment */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = -fe->param->zeta1*(q[ia][ib] + fe->param->zeta0*d[ia][ib])
                -  fe->param->zeta2*(dp[ia][ib] + dp[ib][ia]);
    }
  }

  /* This is an extra minus sign for the divergance. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	s[ia][ib] = -s[ia][ib];
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
    e2 += fe->param->e0coswt[ia]*fe->param->e0coswt[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] +=  fe->param->epsilon
	*(fe->param->e0coswt[ia]*fe->param->e0coswt[ib] - r3*d[ia][ib]*e2);
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
  double e0[3];
  physics_t * phys = NULL;
  PI_DOUBLE(pi);

  assert(fe);

  physics_ref(&phys);
  physics_e0(phys, e0);

  fieldsq = 0.0;
  for (ia = 0; ia < 3; ia++) {
    fieldsq += e0[ia]*e0[ia];
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

__host__ __device__
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

__host__ int fe_lc_redshift_compute(cs_t * cs, fe_lc_t * fe) {

  int ic, jc, kc, index;
  int ia, ib, id, ig;
  int nlocal[3];

  double q[3][3], dq[3][3][3];

  double dq0, dq1, dq2, dq3, sum;
  double egrad_local[2], egrad[2];    /* Gradient terms for redshift calc. */
  double rnew;

  MPI_Comm comm;
  LEVI_CIVITA_CHAR(e);

  if (fe->param->is_redshift_updated == 0) return 0;

  assert(cs);

  cs_cart_comm(cs, &comm);
  cs_nlocal(cs, nlocal);

  egrad_local[0] = 0.0;
  egrad_local[1] = 0.0;

  /* Accumulate the sums (all fluid) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);

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

  MPI_Allreduce(egrad_local, egrad, 2, MPI_DOUBLE, MPI_SUM, comm);

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

/*****************************************************************************
 *
 *  fe_lc_active_stress
 *
 *  Driver to compute the zeta2 term in the active stress. This must
 *  be done
 *    1. after computation of order parameter gradient d_a Q_bc
 *    2. before the final stress computation
 *
 *  The full active stress is written:
 *
 *  S_ab = zeta_0 d_ab - zeta_1 Q_ab - zeta_2 (d_a P_b + d_b P_a)
 *
 *  where P_a = Q_ak d_m Q_mk. The first two terms are simple to compute.
 *
 *  Computation of the zeta_2 term is split into two stages. Here, we
 *  compute P_a from the existing Q_ab and gradient d_a Q_bc. We then
 *  take the gradient of P_a. This allows the complete term to be
 *  computed as part of the stress.
 *
 *****************************************************************************/

__host__ int fe_lc_active_stress(fe_lc_t * fe) {

  int ic, jc, kc, index;
  int nlocal[3];
  int ia, im, ik;

  double q[3][3];
  double dq[3][3][3];
  double p[3];

  assert(fe);

  if (fe->param->zeta2 == 0.0) return 0;

  assert(fe->p);
  assert(fe->dp);

  cs_nlocal(fe->cs, nlocal);

  /* compute p_a */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(fe->cs, ic, jc, kc);
	field_tensor(fe->q, index, q);
	field_grad_tensor_grad(fe->dq, index, dq);

	for (ia = 0; ia < 3; ia++) {
	  p[ia] = 0.0;
	  for (ik = 0; ik < 3; ik++) {
	    for (im = 0; im < 3; im++) {
	      p[ia] += q[ia][ik]*dq[im][im][ik];
	    }
	  }
	}

	field_vector_set(fe->p, index, p);
        /* Next site */
      }
    }
  }

  field_halo(fe->p);
  fe->dp->d2 = fe->dq->d2; /* Kludge - set same gradient for dp */
  field_grad_compute(fe->dp);

  return 0;
}


/*****************************************************************************
 *
 *  fe_lc_mol_field_v
 *
 *****************************************************************************/

__host__ __device__
void fe_lc_mol_field_v(fe_lc_t * fe, int index, double h[3][3][NSIMDVL]) {

  int ia, iv;

  double q[3][3][NSIMDVL];
  double dq[3][3][3][NSIMDVL];
  double dsq[3][3][NSIMDVL];

  double * __restrict__ data;
  double * __restrict__ grad;
  double * __restrict__ delsq;

  assert(fe);
 
  data = fe->q->data;
  grad = fe->dq->grad;
  delsq = fe->dq->delsq;

  /* Expand various tensors */

  for_simd_v(iv, NSIMDVL) q[X][X][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,XX)];
  for_simd_v(iv, NSIMDVL) q[X][Y][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,XY)];
  for_simd_v(iv, NSIMDVL) q[X][Z][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,XZ)];
  for_simd_v(iv, NSIMDVL) q[Y][X][iv] = q[X][Y][iv];
  for_simd_v(iv, NSIMDVL) q[Y][Y][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,YY)];
  for_simd_v(iv, NSIMDVL) q[Y][Z][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,YZ)];
  for_simd_v(iv, NSIMDVL) q[Z][X][iv] = q[X][Z][iv];
  for_simd_v(iv, NSIMDVL) q[Z][Y][iv] = q[Y][Z][iv];
  for_simd_v(iv, NSIMDVL) q[Z][Z][iv] = 0.0 - q[X][X][iv] - q[Y][Y][iv];


  for (ia = 0; ia < NVECTOR; ia++) {
    for_simd_v(iv, NSIMDVL) dq[ia][X][X][iv] = grad[addr_rank2(fe->q->nsites,NQAB,NVECTOR,index+iv,XX,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][X][Y][iv] = grad[addr_rank2(fe->q->nsites,NQAB,NVECTOR,index+iv,XY,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][X][Z][iv] = grad[addr_rank2(fe->q->nsites,NQAB,NVECTOR,index+iv,XZ,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][Y][X][iv] = dq[ia][X][Y][iv];
    for_simd_v(iv, NSIMDVL) dq[ia][Y][Y][iv] = grad[addr_rank2(fe->q->nsites,NQAB,NVECTOR,index+iv,YY,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][Y][Z][iv] = grad[addr_rank2(fe->q->nsites,NQAB,NVECTOR,index+iv,YZ,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][Z][X][iv] = dq[ia][X][Z][iv];
    for_simd_v(iv, NSIMDVL) dq[ia][Z][Y][iv] = dq[ia][Y][Z][iv];
    for_simd_v(iv, NSIMDVL) dq[ia][Z][Z][iv] = 0.0 - dq[ia][X][X][iv] - dq[ia][Y][Y][iv];
  }

  for_simd_v(iv, NSIMDVL) dsq[X][X][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,XX)];
  for_simd_v(iv, NSIMDVL) dsq[X][Y][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,XY)];
  for_simd_v(iv, NSIMDVL) dsq[X][Z][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,XZ)];
  for_simd_v(iv, NSIMDVL) dsq[Y][X][iv] = dsq[X][Y][iv];
  for_simd_v(iv, NSIMDVL) dsq[Y][Y][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,YY)];
  for_simd_v(iv, NSIMDVL) dsq[Y][Z][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,YZ)];
  for_simd_v(iv, NSIMDVL) dsq[Z][X][iv] = dsq[X][Z][iv];
  for_simd_v(iv, NSIMDVL) dsq[Z][Y][iv] = dsq[Y][Z][iv];
  for_simd_v(iv, NSIMDVL) dsq[Z][Z][iv] = 0.0 - dsq[X][X][iv] - dsq[Y][Y][iv];


  fe_lc_compute_h_v(fe, q, dq, dsq, h);

  return;
}

/*****************************************************************************
 *
 *  fe_lc_stress_v
 *
 *  Vectorised version of fe_lc_stress
 *
 *****************************************************************************/

__host__ __device__
void fe_lc_stress_v(fe_lc_t * fe, int index, double s[3][3][NSIMDVL]) { 

  int iv;
  int ia;
 
  double q[3][3][NSIMDVL];
  double h[3][3][NSIMDVL];
  double dq[3][3][3][NSIMDVL];
  double dsq[3][3][NSIMDVL];

  double * __restrict__ data;
  double * __restrict__ grad;
  double * __restrict__ delsq;

  data = fe->q->data;
  grad = fe->dq->grad;
  delsq = fe->dq->delsq;

  for_simd_v(iv, NSIMDVL) q[X][X][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,XX)];
  for_simd_v(iv, NSIMDVL) q[X][Y][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,XY)];
  for_simd_v(iv, NSIMDVL) q[X][Z][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,XZ)];
  for_simd_v(iv, NSIMDVL) q[Y][X][iv] = q[X][Y][iv];
  for_simd_v(iv, NSIMDVL) q[Y][Y][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,YY)];
  for_simd_v(iv, NSIMDVL) q[Y][Z][iv] = data[addr_rank1(fe->q->nsites,NQAB,index+iv,YZ)];
  for_simd_v(iv, NSIMDVL) q[Z][X][iv] = q[X][Z][iv];
  for_simd_v(iv, NSIMDVL) q[Z][Y][iv] = q[Y][Z][iv];
  for_simd_v(iv, NSIMDVL) q[Z][Z][iv] = 0.0 - q[X][X][iv] - q[Y][Y][iv];

  for (ia = 0; ia < NVECTOR; ia++) {
    for_simd_v(iv, NSIMDVL) dq[ia][X][X][iv] = grad[addr_rank2(fe->q->nsites,NQAB,3,index+iv,XX,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][X][Y][iv] = grad[addr_rank2(fe->q->nsites,NQAB,3,index+iv,XY,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][X][Z][iv] = grad[addr_rank2(fe->q->nsites,NQAB,3,index+iv,XZ,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][Y][X][iv] = dq[ia][X][Y][iv];
    for_simd_v(iv, NSIMDVL) dq[ia][Y][Y][iv] = grad[addr_rank2(fe->q->nsites,NQAB,3,index+iv,YY,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][Y][Z][iv] = grad[addr_rank2(fe->q->nsites,NQAB,3,index+iv,YZ,ia)];
    for_simd_v(iv, NSIMDVL) dq[ia][Z][X][iv] = dq[ia][X][Z][iv];
    for_simd_v(iv, NSIMDVL) dq[ia][Z][Y][iv] = dq[ia][Y][Z][iv];
    for_simd_v(iv, NSIMDVL) dq[ia][Z][Z][iv] = 0.0 - dq[ia][X][X][iv] - dq[ia][Y][Y][iv];
  }

  for_simd_v(iv, NSIMDVL) dsq[X][X][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,XX)];
  for_simd_v(iv, NSIMDVL) dsq[X][Y][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,XY)];
  for_simd_v(iv, NSIMDVL) dsq[X][Z][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,XZ)];
  for_simd_v(iv, NSIMDVL) dsq[Y][X][iv] = dsq[X][Y][iv];
  for_simd_v(iv, NSIMDVL) dsq[Y][Y][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,YY)];
  for_simd_v(iv, NSIMDVL) dsq[Y][Z][iv] = delsq[addr_rank1(fe->q->nsites,NQAB,index+iv,YZ)];
  for_simd_v(iv, NSIMDVL) dsq[Z][X][iv] = dsq[X][Z][iv];
  for_simd_v(iv, NSIMDVL) dsq[Z][Y][iv] = dsq[Y][Z][iv];
  for_simd_v(iv, NSIMDVL) dsq[Z][Z][iv] = 0.0 - dsq[X][X][iv] - dsq[Y][Y][iv];

  fe_lc_compute_h_v(fe, q, dq, dsq, h);
  fe_lc_compute_stress_v(fe, q, dq, h, s);

  if (fe->param->is_active) {
    int ib;
    double dp[3][3];
    double sa[3][3];
    double q1[3][3];
    for (iv = 0; iv < NSIMDVL; iv++) {
      field_grad_vector_grad(fe->dp, index + iv, dp);
      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  q1[ia][ib] = q[ia][ib][iv];
	}
      }
      fe_lc_compute_stress_active(fe, q1, dp, sa);
      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  s[ia][ib][iv] += sa[ia][ib];
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_str_symm_v
 *
 *****************************************************************************/

__host__ __device__ void fe_lc_str_symm_v(fe_lc_t * fe, int index,
			 		  double s[3][3][NSIMDVL]) {

  int ia, ib, iv;
  double s1[3][3];

  assert(fe);

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_lc_str_symm(fe, index + iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_str_anti_v
 *
 *****************************************************************************/

__host__ __device__ void fe_lc_str_anti_v(fe_lc_t * fe, int index,
			 		  double s[3][3][NSIMDVL]) {
  assert(fe);

  int ia, ib, iv;
  double s1[3][3];

  assert(fe);

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_lc_str_anti(fe, index + iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_compute_fed_v
 *
 *  Vectorised version of fe_lc_dompute_fed().
 *
 *  NO gamma = gamma(r) at the moment.
 *
 ****************************************************************************/

__host__ __device__
void fe_lc_compute_fed_v(fe_lc_t * fe,
			 double q[3][3][NSIMDVL], 
			 double dq[3][3][3][NSIMDVL],
			 double fed[NSIMDVL]) {
  int iv;
  int ia, ib, ic;

  double q0;
  double kappa0;
  double kappa1;

  double sum[NSIMDVL];
  double q2[NSIMDVL], q3[NSIMDVL];
  double dq0[NSIMDVL], dq1[NSIMDVL];
  double efield[NSIMDVL];
  const double r3 = 1.0/3.0;

  assert(fe);

  /* Redshifted values */
  q0 = fe->param->rredshift*fe->param->q0;
  kappa0 = fe->param->redshift*fe->param->redshift*fe->param->kappa0;
  kappa1 = kappa0;

  for_simd_v(iv, NSIMDVL) q2[iv] = 0.0;

  /* Q_ab^2 */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL)  q2[iv] += q[ia][ib][iv]*q[ia][ib][iv];
    }
  }

  /* Q_ab Q_bc Q_ca */

  for_simd_v(iv, NSIMDVL) q3[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	/* We use here the fact that q[ic][ia] = q[ia][ic] */
	for_simd_v(iv, NSIMDVL)  q3[iv] += q[ia][ib][iv]*q[ib][ic][iv]*q[ia][ic][iv];
      }
    }
  }

  /* (d_b Q_ab)^2 */

  for_simd_v(iv, NSIMDVL)  dq0[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for_simd_v(iv, NSIMDVL)  sum[iv] = 0.0;
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL)  sum[iv] += dq[ib][ia][ib][iv];
    }
    for_simd_v(iv, NSIMDVL)  dq0[iv] += sum[iv]*sum[iv];
  }

  /* (e_acd d_c Q_db + 2q_0 Q_ab)^2 */
  /* With symmetric Q_db write Q_bd */

  for_simd_v(iv, NSIMDVL)  dq1[iv] = 0.0;

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[1][0][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[2][0][1][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[1][1][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[2][1][1][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[1][2][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[2][2][1][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[0][2][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[0][0][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[2][0][0][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[0][1][2][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += dq[2][1][0][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[0][2][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[2][2][0][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[1][2][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[0][0][1][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[1][0][0][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[0][1][1][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[1][1][0][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[0][2][1][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] -= dq[1][2][0][iv];

  for_simd_v(iv, NSIMDVL) sum[iv] += 2.0*q0*q[2][2][iv];

  for_simd_v(iv, NSIMDVL) dq1[iv] += sum[iv]*sum[iv];


  /* Electric field term (epsilon_ includes the factor 1/12pi) */

  for_simd_v(iv, NSIMDVL)  efield[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL) {
	efield[iv] += fe->param->e0coswt[ia]*q[ia][ib][iv]*fe->param->e0coswt[ib];
      }
    }
  }


  for_simd_v(iv, NSIMDVL) {
    fed[iv] = 0.5*fe->param->a0*(1.0 - r3*fe->param->gamma)*q2[iv]
      - r3*fe->param->a0*fe->param->gamma*q3[iv]
      + 0.25*fe->param->a0*fe->param->gamma*q2[iv]*q2[iv]
      + 0.5*kappa0*dq0[iv] + 0.5*kappa1*dq1[iv]
      - fe->param->epsilon*efield[iv];
  }

  return;
}


/*****************************************************************************
 *
 *  fe_lc_compute_h_v
 *
 *  Vectorised version of the molecular field computation.
 *
 *  Alan's note for GPU version.
 *
 *  To get temperary q[][][] etc arrays into registers really requires
 *  inlining to caller file scope.
 *
 *  NO gamma = gamma(r) at the mooment.
 *
 *****************************************************************************/

__host__ __device__ __inline__
void fe_lc_compute_h_v(fe_lc_t * fe,
		       double q[3][3][NSIMDVL], 
		       double dq[3][3][3][NSIMDVL],
		       double dsq[3][3][NSIMDVL], 
		       double h[3][3][NSIMDVL]) {

  int iv;
  int ia, ib, ic;
  double q0;
  double gamma;
  double kappa0;
  double kappa1;

  double q2[NSIMDVL];
  double e2[NSIMDVL];
  double edq[NSIMDVL];
  double sum[NSIMDVL];

  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d);
  LEVI_CIVITA_CHAR(e);

  /* Redshifted values */
  q0 = fe->param->rredshift*fe->param->q0;
  kappa0 = fe->param->redshift*fe->param->redshift*fe->param->kappa0;
  kappa1 = kappa0;

  gamma = fe->param->gamma;

  /* From the bulk terms in the free energy... */

  for_simd_v(iv, NSIMDVL) q2[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL) q2[iv] += q[ia][ib][iv]*q[ia][ib][iv];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for_simd_v(iv, NSIMDVL) sum[iv] += q[ia][ic][iv]*q[ib][ic][iv];
      }
      for_simd_v(iv, NSIMDVL) h[ia][ib][iv] =
	- fe->param->a0*(1.0 - r3*gamma)*q[ia][ib][iv]
	+ fe->param->a0*gamma*(sum[iv] - r3*q2[iv]*d[ia][ib])
	- fe->param->a0*gamma*q2[iv]*q[ia][ib][iv];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  for_simd_v(iv, NSIMDVL) edq[iv] = 0.0;

  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	for_simd_v(iv, NSIMDVL) edq[iv] += e[ib][ic][ia]*dq[ib][ic][ia][iv];
      }
    }
  }

  /* Contraction d_c Q_db ... */

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[1][0][2][iv] + dq[1][0][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[2][0][1][iv] + -dq[2][0][1][iv];
  for_simd_v(iv, NSIMDVL) {
    h[0][0][iv] += kappa0*dsq[0][0][iv] - 2.0*kappa1*q0*sum[iv]
      + 4.0*r3*kappa1*q0*edq[iv] - 4.0*kappa1*q0*q0*q[0][0][iv];
  }

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[0][0][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[1][1][2][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[2][0][0][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[2][1][1][iv] ;
  for_simd_v(iv, NSIMDVL) {
    h[0][1][iv] += kappa0*dsq[0][1][iv] - 2.0*kappa1*q0*sum[iv]
      - 4.0*kappa1*q0*q0*q[0][1][iv];
  }

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[0][0][1][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[1][0][0][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[1][2][2][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[2][2][1][iv] ;
  for_simd_v(iv, NSIMDVL) {
    h[0][2][iv] += kappa0*dsq[0][2][iv] - 2.0*kappa1*q0*sum[iv]
      - 4.0*kappa1*q0*q0*q[0][2][iv];
    }

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[0][0][2][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[1][1][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[2][0][0][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[2][1][1][iv];
  for_simd_v(iv, NSIMDVL) {
    h[1][0][iv] += kappa0*dsq[1][0][iv] - 2.0*kappa1*q0*sum[iv]
      - 4.0*kappa1*q0*q0*q[1][0][iv];
  }

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[0][1][2][iv] + -dq[0][1][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[2][1][0][iv] + dq[2][1][0][iv];
  for_simd_v(iv, NSIMDVL) {
    h[1][1][iv] += kappa0*dsq[1][1][iv] - 2.0*kappa1*q0*sum[iv]
      + 4.0*r3*kappa1*q0*edq[iv] - 4.0*kappa1*q0*q0*q[1][1][iv];
  }

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[0][1][1][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[0][2][2][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[1][1][0][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[2][2][0][iv] ;
  for_simd_v(iv, NSIMDVL) {
    h[1][2][iv] += kappa0*dsq[1][2][iv] - 2.0*kappa1*q0*sum[iv]
      - 4.0*kappa1*q0*q0*q[1][2][iv];
  }

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[0][0][1][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[1][0][0][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[1][2][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[2][2][1][iv];
  for_simd_v(iv, NSIMDVL) {
    h[2][0][iv] += kappa0*dsq[2][0][iv] - 2.0*kappa1*q0*sum[iv]
      - 4.0*kappa1*q0*q0*q[2][0][iv];
  }

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[0][1][1][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[0][2][2][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[1][1][0][iv] ;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[2][2][0][iv];
  for_simd_v(iv, NSIMDVL) {
    h[2][1][iv] += kappa0*dsq[2][1][iv] - 2.0*kappa1*q0*sum[iv]
      - 4.0*kappa1*q0*q0*q[2][1][iv];
  }

  for_simd_v(iv, NSIMDVL) sum[iv] = 0.0;
  for_simd_v(iv, NSIMDVL) sum[iv] += dq[0][2][1][iv] + dq[0][2][1][iv];
  for_simd_v(iv, NSIMDVL) sum[iv] += -dq[1][2][0][iv] + -dq[1][2][0][iv];
  for_simd_v(iv, NSIMDVL) {
    h[2][2][iv] += kappa0*dsq[2][2][iv] - 2.0*kappa1*q0*sum[iv]
      + 4.0*r3*kappa1*q0*edq[iv] - 4.0*kappa1*q0*q0*q[2][2][iv];
  }

  /* Electric field term */

  for_simd_v(iv, NSIMDVL) e2[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for_simd_v(iv, NSIMDVL) {
      e2[iv] += fe->param->e0coswt[ia]*fe->param->e0coswt[ia];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL) {
	h[ia][ib][iv] +=  fe->param->epsilon*
	  (fe->param->e0coswt[ia]*fe->param->e0coswt[ib] - r3*d[ia][ib]*e2[iv]);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_compute_stress_v
 *
 *  Vectorised version of fe_lc_compute_stress()
 *
 *****************************************************************************/

__host__ __device__
void fe_lc_compute_stress_v(fe_lc_t * fe,
			    double q[3][3][NSIMDVL],
			    double dq[3][3][3][NSIMDVL],
			    double h[3][3][NSIMDVL],
			    double s[3][3][NSIMDVL]) {
  int ia, ib;
  int iv;

  double kappa0;
  double kappa1;
  double q0;
  double xi;

  double qh[NSIMDVL];
  double p0[NSIMDVL];
  double sthtmp[NSIMDVL];

  const double r3 = (1.0/3.0);

  /* Redshifted values */

  q0 = fe->param->q0*fe->param->rredshift;
  kappa0 = fe->param->kappa0*fe->param->redshift*fe->param->redshift;
  kappa1 = fe->param->kappa1*fe->param->redshift*fe->param->redshift;

  xi = fe->param->xi;

  /* We have ignored the rho T term at the moment, assumed to be zero
     (in particular, it has no divergence if rho = const). */

  fe_lc_compute_fed_v(fe, q, dq, p0);

  for_simd_v(iv, NSIMDVL) p0[iv] = 0.0 - p0[iv]; 

  /* The contraction Q_ab H_ab */

  for_simd_v(iv, NSIMDVL) qh[iv] = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL) qh[iv] += q[ia][ib][iv]*h[ia][ib][iv];
    }
  }

  /* The rest is automatically generated following
   * fe_lc_compute_stress() */


  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[0][0][iv]+ r3)*qh[iv] -p0[iv] ;

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][0][iv]*(q[0][0][iv] + r3)   -xi*(q[0][0][iv]    +r3)*h[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][1][iv]*(q[0][1][iv])   -xi*(q[0][1][iv]    )*h[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][2][iv]*(q[0][2][iv])   -xi*(q[0][2][iv]    )*h[0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][0][iv]*dq[0][0][0][iv] - kappa1*dq[0][0][0][iv]*dq[0][0][0][iv]+ kappa1*dq[0][0][0][iv]*dq[0][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][0][iv]*dq[1][0][1][iv] - kappa1*dq[0][0][1][iv]*dq[0][0][1][iv]+ kappa1*dq[0][0][1][iv]*dq[0][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][0][iv]*dq[2][0][2][iv] - kappa1*dq[0][0][2][iv]*dq[0][0][2][iv]+ kappa1*dq[0][0][2][iv]*dq[0][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][1][iv]*dq[0][1][0][iv] - kappa1*dq[0][1][0][iv]*dq[0][1][0][iv]+ kappa1*dq[0][1][0][iv]*dq[1][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][1][0][iv]*q[0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][1][iv]*dq[1][1][1][iv] - kappa1*dq[0][1][1][iv]*dq[0][1][1][iv]+ kappa1*dq[0][1][1][iv]*dq[1][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][1][1][iv]*q[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][1][iv]*dq[2][1][2][iv] - kappa1*dq[0][1][2][iv]*dq[0][1][2][iv]+ kappa1*dq[0][1][2][iv]*dq[1][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][1][2][iv]*q[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][2][iv]*dq[0][2][0][iv] - kappa1*dq[0][2][0][iv]*dq[0][2][0][iv]+ kappa1*dq[0][2][0][iv]*dq[2][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][2][0][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][2][iv]*dq[1][2][1][iv] - kappa1*dq[0][2][1][iv]*dq[0][2][1][iv]+ kappa1*dq[0][2][1][iv]*dq[2][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][2][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][0][2][iv]*dq[2][2][2][iv] - kappa1*dq[0][2][2][iv]*dq[0][2][2][iv]+ kappa1*dq[0][2][2][iv]*dq[2][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][2][2][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][0][iv]*h[0][0][iv] - h[0][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][1][iv]*h[0][1][iv] - h[0][1][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][2][iv]*h[0][2][iv] - h[0][2][iv]*q[0][2][iv];


  /* XX -ve sign */
  for_simd_v(iv, NSIMDVL) s[X][X][iv] = -sthtmp[iv];


  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[0][1][iv])*qh[iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][0][iv]*(q[1][0][iv])   -xi*(q[0][0][iv]    +r3)*h[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][1][iv]*(q[1][1][iv] + r3)   -xi*(q[0][1][iv]    )*h[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][2][iv]*(q[1][2][iv])   -xi*(q[0][2][iv]    )*h[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][0][iv]*dq[0][0][0][iv] - kappa1*dq[0][0][0][iv]*dq[1][0][0][iv]+ kappa1*dq[0][0][0][iv]*dq[0][1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][0][0][iv]*q[0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][0][iv]*dq[1][0][1][iv] - kappa1*dq[0][0][1][iv]*dq[1][0][1][iv]+ kappa1*dq[0][0][1][iv]*dq[0][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][0][1][iv]*q[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][0][iv]*dq[2][0][2][iv] - kappa1*dq[0][0][2][iv]*dq[1][0][2][iv]+ kappa1*dq[0][0][2][iv]*dq[0][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][0][2][iv]*q[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][1][iv]*dq[0][1][0][iv] - kappa1*dq[0][1][0][iv]*dq[1][1][0][iv]+ kappa1*dq[0][1][0][iv]*dq[1][1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][1][iv]*dq[1][1][1][iv] - kappa1*dq[0][1][1][iv]*dq[1][1][1][iv]+ kappa1*dq[0][1][1][iv]*dq[1][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][1][iv]*dq[2][1][2][iv] - kappa1*dq[0][1][2][iv]*dq[1][1][2][iv]+ kappa1*dq[0][1][2][iv]*dq[1][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][2][iv]*dq[0][2][0][iv] - kappa1*dq[0][2][0][iv]*dq[1][2][0][iv]+ kappa1*dq[0][2][0][iv]*dq[2][1][0][iv];
    
  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][2][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][2][iv]*dq[1][2][1][iv] - kappa1*dq[0][2][1][iv]*dq[1][2][1][iv]+ kappa1*dq[0][2][1][iv]*dq[2][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][2][1][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][1][2][iv]*dq[2][2][2][iv] - kappa1*dq[0][2][2][iv]*dq[1][2][2][iv]+ kappa1*dq[0][2][2][iv]*dq[2][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][2][2][iv]*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][0][iv]*h[1][0][iv] - h[0][0][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][1][iv]*h[1][1][iv] - h[0][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][2][iv]*h[1][2][iv] - h[0][2][iv]*q[1][2][iv];


  /* XY with minus sign */
  for_simd_v(iv, NSIMDVL) s[X][Y][iv] = -sthtmp[iv];


  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[0][2][iv])*qh[iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][0][iv]*(q[2][0][iv])   -xi*(q[0][0][iv]    +r3)*h[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][1][iv]*(q[2][1][iv])   -xi*(q[0][1][iv]    )*h[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[0][2][iv]*(q[2][2][iv] + r3)   -xi*(q[0][2][iv]    )*h[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][0][iv]*dq[0][0][0][iv] - kappa1*dq[0][0][0][iv]*dq[2][0][0][iv]+ kappa1*dq[0][0][0][iv]*dq[0][2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][0][0][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][0][iv]*dq[1][0][1][iv] - kappa1*dq[0][0][1][iv]*dq[2][0][1][iv]+ kappa1*dq[0][0][1][iv]*dq[0][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][0][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][0][iv]*dq[2][0][2][iv] - kappa1*dq[0][0][2][iv]*dq[2][0][2][iv]+ kappa1*dq[0][0][2][iv]*dq[0][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[0][0][2][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][1][iv]*dq[0][1][0][iv] - kappa1*dq[0][1][0][iv]*dq[2][1][0][iv]+ kappa1*dq[0][1][0][iv]*dq[1][2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][1][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][1][iv]*dq[1][1][1][iv] - kappa1*dq[0][1][1][iv]*dq[2][1][1][iv]+ kappa1*dq[0][1][1][iv]*dq[1][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][1][1][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][1][iv]*dq[2][1][2][iv] - kappa1*dq[0][1][2][iv]*dq[2][1][2][iv]+ kappa1*dq[0][1][2][iv]*dq[1][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[0][1][2][iv]*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][2][iv]*dq[0][2][0][iv] - kappa1*dq[0][2][0][iv]*dq[2][2][0][iv]+ kappa1*dq[0][2][0][iv]*dq[2][2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][2][iv]*dq[1][2][1][iv] - kappa1*dq[0][2][1][iv]*dq[2][2][1][iv]+ kappa1*dq[0][2][1][iv]*dq[2][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[0][2][2][iv]*dq[2][2][2][iv] - kappa1*dq[0][2][2][iv]*dq[2][2][2][iv]+ kappa1*dq[0][2][2][iv]*dq[2][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][0][iv]*h[2][0][iv] - h[0][0][iv]*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][1][iv]*h[2][1][iv] - h[0][1][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[0][2][iv]*h[2][2][iv] - h[0][2][iv]*q[2][2][iv];


  /* XZ with -ve sign*/
  for_simd_v(iv, NSIMDVL) s[X][Z][iv] = -sthtmp[iv];



  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[1][0][iv])*qh[iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][0][iv]*(q[0][0][iv] + r3)   -xi*(q[1][0][iv]    )*h[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][1][iv]*(q[0][1][iv])   -xi*(q[1][1][iv]    +r3)*h[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][2][iv]*(q[0][2][iv])   -xi*(q[1][2][iv]    )*h[0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][0][iv]*dq[0][0][0][iv] - kappa1*dq[1][0][0][iv]*dq[0][0][0][iv]+ kappa1*dq[1][0][0][iv]*dq[0][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][0][iv]*dq[1][0][1][iv] - kappa1*dq[1][0][1][iv]*dq[0][0][1][iv]+ kappa1*dq[1][0][1][iv]*dq[0][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][0][iv]*dq[2][0][2][iv] - kappa1*dq[1][0][2][iv]*dq[0][0][2][iv]+ kappa1*dq[1][0][2][iv]*dq[0][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][1][iv]*dq[0][1][0][iv] - kappa1*dq[1][1][0][iv]*dq[0][1][0][iv]+ kappa1*dq[1][1][0][iv]*dq[1][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][1][0][iv]*q[0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][1][iv]*dq[1][1][1][iv] - kappa1*dq[1][1][1][iv]*dq[0][1][1][iv]+ kappa1*dq[1][1][1][iv]*dq[1][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][1][1][iv]*q[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][1][iv]*dq[2][1][2][iv] - kappa1*dq[1][1][2][iv]*dq[0][1][2][iv]+ kappa1*dq[1][1][2][iv]*dq[1][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][1][2][iv]*q[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][2][iv]*dq[0][2][0][iv] - kappa1*dq[1][2][0][iv]*dq[0][2][0][iv]+ kappa1*dq[1][2][0][iv]*dq[2][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][2][0][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][2][iv]*dq[1][2][1][iv] - kappa1*dq[1][2][1][iv]*dq[0][2][1][iv]+ kappa1*dq[1][2][1][iv]*dq[2][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][2][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][0][2][iv]*dq[2][2][2][iv] - kappa1*dq[1][2][2][iv]*dq[0][2][2][iv]+ kappa1*dq[1][2][2][iv]*dq[2][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][2][2][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][0][iv]*h[0][0][iv] - h[1][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][1][iv]*h[0][1][iv] - h[1][1][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][2][iv]*h[0][2][iv] - h[1][2][iv]*q[0][2][iv];


  /* YX -ve sign */
  for_simd_v(iv, NSIMDVL) s[Y][X][iv] = -sthtmp[iv];



  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[1][1][iv]+ r3)*qh[iv] -p0[iv] ;

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][0][iv]*(q[1][0][iv])   -xi*(q[1][0][iv]    )*h[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][1][iv]*(q[1][1][iv] + r3)   -xi*(q[1][1][iv]    +r3)*h[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][2][iv]*(q[1][2][iv])   -xi*(q[1][2][iv]    )*h[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][0][iv]*dq[0][0][0][iv] - kappa1*dq[1][0][0][iv]*dq[1][0][0][iv]+ kappa1*dq[1][0][0][iv]*dq[0][1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][0][0][iv]*q[0][2][iv];
  
  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][0][iv]*dq[1][0][1][iv] - kappa1*dq[1][0][1][iv]*dq[1][0][1][iv]+ kappa1*dq[1][0][1][iv]*dq[0][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][0][1][iv]*q[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][0][iv]*dq[2][0][2][iv] - kappa1*dq[1][0][2][iv]*dq[1][0][2][iv]+ kappa1*dq[1][0][2][iv]*dq[0][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][0][2][iv]*q[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][1][iv]*dq[0][1][0][iv] - kappa1*dq[1][1][0][iv]*dq[1][1][0][iv]+ kappa1*dq[1][1][0][iv]*dq[1][1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][1][iv]*dq[1][1][1][iv] - kappa1*dq[1][1][1][iv]*dq[1][1][1][iv]+ kappa1*dq[1][1][1][iv]*dq[1][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][1][iv]*dq[2][1][2][iv] - kappa1*dq[1][1][2][iv]*dq[1][1][2][iv]+ kappa1*dq[1][1][2][iv]*dq[1][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][2][iv]*dq[0][2][0][iv] - kappa1*dq[1][2][0][iv]*dq[1][2][0][iv]+ kappa1*dq[1][2][0][iv]*dq[2][1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][2][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][2][iv]*dq[1][2][1][iv] - kappa1*dq[1][2][1][iv]*dq[1][2][1][iv]+ kappa1*dq[1][2][1][iv]*dq[2][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][2][1][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][1][2][iv]*dq[2][2][2][iv] - kappa1*dq[1][2][2][iv]*dq[1][2][2][iv]+ kappa1*dq[1][2][2][iv]*dq[2][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][2][2][iv]*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][0][iv]*h[1][0][iv] - h[1][0][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][1][iv]*h[1][1][iv] - h[1][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][2][iv]*h[1][2][iv] - h[1][2][iv]*q[1][2][iv];


  /* YY -ve sign */
  for_simd_v(iv, NSIMDVL) s[Y][Y][iv] = -sthtmp[iv];



  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[1][2][iv])*qh[iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][0][iv]*(q[2][0][iv])   -xi*(q[1][0][iv]    )*h[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][1][iv]*(q[2][1][iv])   -xi*(q[1][1][iv]    +r3)*h[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[1][2][iv]*(q[2][2][iv] + r3)   -xi*(q[1][2][iv]    )*h[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][0][iv]*dq[0][0][0][iv] - kappa1*dq[1][0][0][iv]*dq[2][0][0][iv]+ kappa1*dq[1][0][0][iv]*dq[0][2][0][iv];
  
  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][0][0][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][0][iv]*dq[1][0][1][iv] - kappa1*dq[1][0][1][iv]*dq[2][0][1][iv]+ kappa1*dq[1][0][1][iv]*dq[0][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][0][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][0][iv]*dq[2][0][2][iv] - kappa1*dq[1][0][2][iv]*dq[2][0][2][iv]+ kappa1*dq[1][0][2][iv]*dq[0][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[1][0][2][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][1][iv]*dq[0][1][0][iv] - kappa1*dq[1][1][0][iv]*dq[2][1][0][iv]+ kappa1*dq[1][1][0][iv]*dq[1][2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][1][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][1][iv]*dq[1][1][1][iv] - kappa1*dq[1][1][1][iv]*dq[2][1][1][iv]+ kappa1*dq[1][1][1][iv]*dq[1][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][1][1][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][1][iv]*dq[2][1][2][iv] - kappa1*dq[1][1][2][iv]*dq[2][1][2][iv]+ kappa1*dq[1][1][2][iv]*dq[1][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[1][1][2][iv]*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][2][iv]*dq[0][2][0][iv] - kappa1*dq[1][2][0][iv]*dq[2][2][0][iv]+ kappa1*dq[1][2][0][iv]*dq[2][2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][2][iv]*dq[1][2][1][iv] - kappa1*dq[1][2][1][iv]*dq[2][2][1][iv]+ kappa1*dq[1][2][1][iv]*dq[2][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[1][2][2][iv]*dq[2][2][2][iv] - kappa1*dq[1][2][2][iv]*dq[2][2][2][iv]+ kappa1*dq[1][2][2][iv]*dq[2][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][0][iv]*h[2][0][iv] - h[1][0][iv]*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][1][iv]*h[2][1][iv] - h[1][1][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[1][2][iv]*h[2][2][iv] - h[1][2][iv]*q[2][2][iv];


  /* YZ -ve sign */
  for_simd_v(iv, NSIMDVL) s[Y][Z][iv] = -sthtmp[iv];



  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[2][0][iv])*qh[iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][0][iv]*(q[0][0][iv] + r3)   -xi*(q[2][0][iv]    )*h[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][1][iv]*(q[0][1][iv])   -xi*(q[2][1][iv]    )*h[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][2][iv]*(q[0][2][iv])   -xi*(q[2][2][iv]    +r3)*h[0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][0][iv]*dq[0][0][0][iv] - kappa1*dq[2][0][0][iv]*dq[0][0][0][iv]+ kappa1*dq[2][0][0][iv]*dq[0][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][0][iv]*dq[1][0][1][iv] - kappa1*dq[2][0][1][iv]*dq[0][0][1][iv]+ kappa1*dq[2][0][1][iv]*dq[0][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][0][iv]*dq[2][0][2][iv] - kappa1*dq[2][0][2][iv]*dq[0][0][2][iv]+ kappa1*dq[2][0][2][iv]*dq[0][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][1][iv]*dq[0][1][0][iv] - kappa1*dq[2][1][0][iv]*dq[0][1][0][iv]+ kappa1*dq[2][1][0][iv]*dq[1][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][1][0][iv]*q[0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][1][iv]*dq[1][1][1][iv] - kappa1*dq[2][1][1][iv]*dq[0][1][1][iv]+ kappa1*dq[2][1][1][iv]*dq[1][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][1][1][iv]*q[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][1][iv]*dq[2][1][2][iv] - kappa1*dq[2][1][2][iv]*dq[0][1][2][iv]+ kappa1*dq[2][1][2][iv]*dq[1][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][1][2][iv]*q[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][2][iv]*dq[0][2][0][iv] - kappa1*dq[2][2][0][iv]*dq[0][2][0][iv]+ kappa1*dq[2][2][0][iv]*dq[2][0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][2][0][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][2][iv]*dq[1][2][1][iv] - kappa1*dq[2][2][1][iv]*dq[0][2][1][iv]+ kappa1*dq[2][2][1][iv]*dq[2][0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][2][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][0][2][iv]*dq[2][2][2][iv] - kappa1*dq[2][2][2][iv]*dq[0][2][2][iv]+ kappa1*dq[2][2][2][iv]*dq[2][0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][2][2][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][0][iv]*h[0][0][iv] - h[2][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][1][iv]*h[0][1][iv] - h[2][1][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][2][iv]*h[0][2][iv] - h[2][2][iv]*q[0][2][iv];


  /* ZX -ve sign */
  for_simd_v(iv, NSIMDVL) s[Z][X][iv] = -sthtmp[iv];




  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[2][1][iv])*qh[iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][0][iv]*(q[1][0][iv])   -xi*(q[2][0][iv]    )*h[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][1][iv]*(q[1][1][iv] + r3)   -xi*(q[2][1][iv]    )*h[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][2][iv]*(q[1][2][iv])   -xi*(q[2][2][iv]    +r3)*h[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][0][iv]*dq[0][0][0][iv] - kappa1*dq[2][0][0][iv]*dq[1][0][0][iv]+ kappa1*dq[2][0][0][iv]*dq[0][1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][0][0][iv]*q[0][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][0][iv]*dq[1][0][1][iv] - kappa1*dq[2][0][1][iv]*dq[1][0][1][iv]+ kappa1*dq[2][0][1][iv]*dq[0][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][0][1][iv]*q[1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][0][iv]*dq[2][0][2][iv] - kappa1*dq[2][0][2][iv]*dq[1][0][2][iv]+ kappa1*dq[2][0][2][iv]*dq[0][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][0][2][iv]*q[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][1][iv]*dq[0][1][0][iv] - kappa1*dq[2][1][0][iv]*dq[1][1][0][iv]+ kappa1*dq[2][1][0][iv]*dq[1][1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][1][iv]*dq[1][1][1][iv] - kappa1*dq[2][1][1][iv]*dq[1][1][1][iv]+ kappa1*dq[2][1][1][iv]*dq[1][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][1][iv]*dq[2][1][2][iv] - kappa1*dq[2][1][2][iv]*dq[1][1][2][iv]+ kappa1*dq[2][1][2][iv]*dq[1][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][2][iv]*dq[0][2][0][iv] - kappa1*dq[2][2][0][iv]*dq[1][2][0][iv]+ kappa1*dq[2][2][0][iv]*dq[2][1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][2][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][2][iv]*dq[1][2][1][iv] - kappa1*dq[2][2][1][iv]*dq[1][2][1][iv]+ kappa1*dq[2][2][1][iv]*dq[2][1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][2][1][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][1][2][iv]*dq[2][2][2][iv] - kappa1*dq[2][2][2][iv]*dq[1][2][2][iv]+ kappa1*dq[2][2][2][iv]*dq[2][1][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][2][2][iv]*q[2][0][iv];
  
  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][0][iv]*h[1][0][iv] - h[2][0][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][1][iv]*h[1][1][iv] - h[2][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][2][iv]*h[1][2][iv] - h[2][2][iv]*q[1][2][iv];


  /* ZY -ve sign */
  for_simd_v(iv, NSIMDVL) s[Z][Y][iv] = -sthtmp[iv];



  for_simd_v(iv, NSIMDVL) sthtmp[iv] = 2.0*xi*(q[2][2][iv]+ r3)*qh[iv] -p0[iv] ;

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][0][iv]*(q[2][0][iv])   -xi*(q[2][0][iv]    )*h[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][1][iv]*(q[2][1][iv])   -xi*(q[2][1][iv]    )*h[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += -xi*h[2][2][iv]*(q[2][2][iv] + r3)   -xi*(q[2][2][iv]    +r3)*h[2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][0][iv]*dq[0][0][0][iv] - kappa1*dq[2][0][0][iv]*dq[2][0][0][iv]+ kappa1*dq[2][0][0][iv]*dq[0][2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][0][0][iv]*q[0][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][0][iv]*dq[1][0][1][iv] - kappa1*dq[2][0][1][iv]*dq[2][0][1][iv]+ kappa1*dq[2][0][1][iv]*dq[0][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][0][1][iv]*q[1][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][0][iv]*dq[2][0][2][iv] - kappa1*dq[2][0][2][iv]*dq[2][0][2][iv]+ kappa1*dq[2][0][2][iv]*dq[0][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] -= 2.0*kappa1*q0*dq[2][0][2][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][1][iv]*dq[0][1][0][iv] - kappa1*dq[2][1][0][iv]*dq[2][1][0][iv]+ kappa1*dq[2][1][0][iv]*dq[1][2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][1][0][iv]*q[0][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][1][iv]*dq[1][1][1][iv] - kappa1*dq[2][1][1][iv]*dq[2][1][1][iv]+ kappa1*dq[2][1][1][iv]*dq[1][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][1][1][iv]*q[1][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][1][iv]*dq[2][1][2][iv] - kappa1*dq[2][1][2][iv]*dq[2][1][2][iv]+ kappa1*dq[2][1][2][iv]*dq[1][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += 2.0*kappa1*q0*dq[2][1][2][iv]*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][2][iv]*dq[0][2][0][iv] - kappa1*dq[2][2][0][iv]*dq[2][2][0][iv]+ kappa1*dq[2][2][0][iv]*dq[2][2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][2][iv]*dq[1][2][1][iv] - kappa1*dq[2][2][1][iv]*dq[2][2][1][iv]+ kappa1*dq[2][2][1][iv]*dq[2][2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += - kappa0*dq[2][2][2][iv]*dq[2][2][2][iv] - kappa1*dq[2][2][2][iv]*dq[2][2][2][iv]+ kappa1*dq[2][2][2][iv]*dq[2][2][2][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][0][iv]*h[2][0][iv] - h[2][0][iv]*q[2][0][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][1][iv]*h[2][1][iv] - h[2][1][iv]*q[2][1][iv];

  for_simd_v(iv, NSIMDVL) sthtmp[iv] += q[2][2][iv]*h[2][2][iv] - h[2][2][iv]*q[2][2][iv];


  /* ZZ -ve sign */
  for_simd_v(iv, NSIMDVL) s[Z][Z][iv] = -sthtmp[iv];

  return;
}

