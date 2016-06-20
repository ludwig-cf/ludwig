/****************************************************************************
 *
 *  symmetric.c
 *
 *  Implementation of the symmetric \phi^4 free energy functional:
 *
 *  F[\phi] = (1/2) A \phi^2 + (1/4) B \phi^4 + (1/2) \kappa (\nabla\phi)^2
 *
 *  The first two terms represent the bulk free energy, while the
 *  final term penalises curvature in the interface. For a complete
 *  description see Kendon et al., J. Fluid Mech., 440, 147 (2001).
 *
 *  The usual mode of operation is to take a = -b < 0 and k > 0.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "leesedwards.h"
#include "symmetric.h"

#ifndef OLD_SHIT

/****************************************************************************
 *
 *  fe_symm_create
 *
 *  fe is the "superclass" free energy pointer.
 *
 ****************************************************************************/

__host__ int fe_symm_create(field_t * phi, field_grad_t * dphi,
			    fe_symm_t ** p) {

  fe_symm_t * obj = NULL;

  assert(phi);
  assert(dphi);

  obj = (fe_symm_t *) calloc(1, sizeof(fe_symm_t));
  if (obj == NULL) fatal("calloc(fe_symm_t) failed\n");

  obj->phi = phi;
  obj->dphi = dphi;

  *p = obj;

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_free
 *
 *****************************************************************************/

__host__ int fe_symm_free(fe_symm_t * fe) {

  assert(fe);

  free(fe);

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_param_set
 *
 *  No constraints are placed on the parameters here, so the caller is
 *  responsible for making sure they are sensible.
 *
 ****************************************************************************/

__host__ int fe_symm_param_set(fe_symm_t * fe, fe_symm_param_t values) {

  assert(fe);

  fe->param.a = values.a;
  fe->param.b = values.b;
  fe->param.kappa = values.kappa;

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_param
 *
 *****************************************************************************/

__host__ __device__
int fe_symm_param(fe_symm_t * fe, fe_symm_param_t * values) {

  assert(fe);

  values->a = fe->param.a;
  values->b = fe->param.b;
  values->kappa = fe->param.kappa;

  return 0;
}
/****************************************************************************
 *
 *  fe_symm_interfacial_tension
 *
 *  Assumes phi^* = (-a/b)^1/2 and a < 0.
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_interfacial_tension(fe_symm_t * fe, double * sigma) {

  double a, b, kappa;

  assert(fe);

  a = fe->param.a;
  b = fe->param.b;
  kappa = fe->param.kappa;

  *sigma = sqrt(-8.0*kappa*a*a*a/(9.0*b*b));

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_interfacial_width
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_interfacial_width(fe_symm_t * fe, double * xi) {

  assert(fe);

  *xi = sqrt(-2.0*fe->param.kappa/fe->param.a);

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_free_energy_density
 *
 *  The free energy density is as above.
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_fed(fe_symm_t * fe, int index, double * fed) {

  double phi;
  double dphi[3];

  assert(fe);

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_grad(fe->dphi, index, dphi);

  *fed = (0.5*fe->param.a + 0.25*fe->param.b*phi*phi)*phi*phi
    + 0.5*fe->param.kappa*dot_product(dphi, dphi);

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_str
 *
 *  Return the chemical stress tensor for given position index.
 *
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_str(fe_symm_t * fe, int index,  double s[3][3]) {

  int ia, ib;
  double kappa;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;
  double d_ab;

  assert(fe);

  kappa = fe->param.kappa;

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_grad(fe->dphi, index, grad_phi);
  field_grad_scalar_delsq(fe->dphi, index, &delsq_phi);

  p0 = 0.5*fe->param.a*phi*phi + 0.75*fe->param.b*phi*phi*phi*phi
    - kappa*phi*delsq_phi - 0.5*kappa*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      d_ab = (ia == ib);
      s[ia][ib] = p0*d_ab + kappa*grad_phi[ia]*grad_phi[ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_chemical_potential_target
 *
 *****************************************************************************/

__target__
void fe_symm_chemical_potential_target(fe_symm_t * fe, int index, double * mu) {

  double phi;
  double delsq;

  phi = fe->phi->data[addr_rank0(le_nsites(), index)];
  delsq = fe->dphi->delsq[addr_rank0(le_nsites(), index)];

  *mu = fe->param.a*phi + fe->param.b*phi*phi*phi - fe->param.kappa*delsq;

  return;
}

/*****************************************************************************
 *
 *  fe_symm_chemical_stress_target
 *
 *****************************************************************************/

__target__
void fe_symm_chemical_stress_target(fe_symm_t * fe, int index,
				    double s[3][3*NSIMDVL]) {
  int ia, ib;
  int iv;
  double phi;
  double delsq;
  double grad[3];
  double p0;
  fe_symm_param_t param;

  param = fe->param;

  __targetILP__(iv) {
    for (ia = 0; ia < 3; ia++) {
      grad[ia] = fe->dphi->grad[vaddr_rank2(le_nsites(), 1, 3, index, 0, ia, iv)];
    }

    phi = fe->phi->data[vaddr_rank1(le_nsites(), 1, index, 0, iv)];
    delsq = fe->dphi->delsq[vaddr_rank1(le_nsites(), 1, index, 0, iv)];

    p0 = 0.5*param.a*phi*phi + 0.75*param.b*phi*phi*phi*phi
      - param.kappa*phi*delsq 
      - 0.5*param.kappa*(grad[X]*grad[X] + grad[Y]*grad[Y] + grad[Z]*grad[Z]);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib*NSIMDVL+iv] = p0*tc_d_[ia][ib] + param.kappa*grad[ia]*grad[ib];
      }
    }
  }

  return;
}

#else

static double a_     = -0.003125;
static double b_     = +0.003125;
static double kappa_ = +0.002;

static __targetConst__ double t_a_     = -0.003125;
static __targetConst__ double t_b_     = +0.003125;
static __targetConst__ double t_kappa_ = +0.002;

static field_t * phi_ = NULL;
static field_grad_t * grad_phi_ = NULL;

/****************************************************************************
 *
 *  symmetric_phi_set
 *
 *  Attach a reference to the order parameter field object, and the
 *  associated gradient object.
 *
 ****************************************************************************/

__targetHost__ int symmetric_phi_set(field_t * phi, field_grad_t * dphi) {

  assert(phi);
  assert(dphi);

  phi_ = phi;
  grad_phi_ = dphi;

  return 0;
}

/****************************************************************************
 *
 *  symmetric_free_energy_parameters_set
 *
 *  No constraints are placed on the parameters here, so the caller is
 *  responsible for making sure they are sensible.
 *
 ****************************************************************************/

__targetHost__
void symmetric_free_energy_parameters_set(double a, double b, double kappa) {

  a_ = a;
  b_ = b;
  kappa_ = kappa;

  copyConstToTarget(&t_a_, &a, sizeof(double));
  copyConstToTarget(&t_b_, &b, sizeof(double));
  copyConstToTarget(&t_kappa_, &kappa, sizeof(double));

  fe_kappa_set(kappa);

  return;
}

/****************************************************************************
 *
 *  symmetric_a
 *
 ****************************************************************************/

__targetHost__ double symmetric_a(void) {

  return a_;

}

/****************************************************************************
 *
 *  symmetric_b
 *
 ****************************************************************************/

__targetHost__ double symmetric_b(void) {

  return b_;

}

/****************************************************************************
 *
 *  symmetric_interfacial_tension
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

__targetHost__ double symmetric_interfacial_tension(void) {

  double sigma;


  sigma = sqrt(-8.0*kappa_*a_*a_*a_/(9.0*b_*b_));

  return sigma;
}

/****************************************************************************
 *
 *  symmetric_interfacial_width
 *
 ****************************************************************************/

__targetHost__ double symmetric_interfacial_width(void) {

  double xi;

  xi = sqrt(-2.0*kappa_/a_);

  return xi;
}

/****************************************************************************
 *
 *  symmetric_free_energy_density
 *
 *  The free energy density is as above.
 *
 ****************************************************************************/

__targetHost__ double symmetric_free_energy_density(const int index) {

  double phi;
  double dphi[3];
  double e;

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, dphi);

  e = 0.5*a_*phi*phi + 0.25*b_*phi*phi*phi*phi
	+ 0.5*kappa_*dot_product(dphi, dphi);

  return e;
}

/****************************************************************************
 *
 *  symmetric_chemical_potential
 *
 *  The chemical potential \mu = \delta F / \delta \phi
 *                             = a\phi + b\phi^3 - \kappa\nabla^2 \phi
 *
 ****************************************************************************/

__targetHost__
double symmetric_chemical_potential(const int index, const int nop) {

  double phi;
  double delsq_phi;
  double mu;

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  mu = a_*phi + b_*phi*phi*phi - kappa_*delsq_phi;

  return mu;
}

__target__
double symmetric_chemical_potential_target(const int index, const int nop,
					   const double * t_phi,
					   const double * t_delsqphi) {
  double phi;
  double delsq_phi;
  double mu;

  phi = t_phi[addr_rank0(le_nsites(), index)];
  delsq_phi = t_delsqphi[addr_rank0(le_nsites(), index)];

  mu = t_a_*phi + t_b_*phi*phi*phi - t_kappa_*delsq_phi;

  return mu;
}

/****************************************************************************
 *
 *  symmetric_isotropic_pressure
 *
 *  This ignores the term in the density (assumed to be uniform).
 *
 ****************************************************************************/

__targetHost__ double symmetric_isotropic_pressure(const int index) {

  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, grad_phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);

  return p0;
}

/****************************************************************************
 *
 *  symmetric_chemical_stress
 *
 *  Return the chemical stress tensor for given position index.
 *
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 ****************************************************************************/

__targetHost__ void symmetric_chemical_stress(const int index, double s[3][3]) {

  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, grad_phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]	+ kappa_*grad_phi[ia]*grad_phi[ib];
    }
  }

  return;
}

__target__
void symmetric_chemical_stress_target(const int index,
				      double s[3][3*NSIMDVL],
				      const double* t_phi, 
				      const double* t_gradphi,
				      const double* t_delsqphi) {
  int ia, ib;
  int iv;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  __targetILP__(iv) {
    
    for (ia = 0; ia < 3; ia++) {
      grad_phi[ia] = t_gradphi[vaddr_rank2(le_nsites(), 1, 3, index, 0, ia, iv)];
    }

    phi = t_phi[vaddr_rank1(le_nsites(), 1, index, 0, iv)];
    delsq_phi = t_delsqphi[vaddr_rank1(le_nsites(), 1, index, 0, iv)];

    p0 = 0.5*t_a_*phi*phi + 0.75*t_b_*phi*phi*phi*phi
      - t_kappa_*phi*delsq_phi 
      - 0.5*t_kappa_*(grad_phi[X]*grad_phi[X] + grad_phi[Y]*grad_phi[Y]
		      + grad_phi[Z]*grad_phi[Z]);
    
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib*VVL+ iv] = p0*tc_d_[ia][ib]
	  + t_kappa_*grad_phi[ia]*grad_phi[ib];
      }
    }
  }

  return;
}

#endif
