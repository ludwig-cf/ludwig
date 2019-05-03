/****************************************************************************
 *
 *  surfactant.c
 *
 *  Implementation of the surfactant free energy described by
 *  van der Graff and van der Sman [REFERENCE].
 *
 *  Two order parameters are required:
 *
 *  [0] \phi is compositional order parameter (cf symmetric free energy)
 *  [1] \psi is surfactant concentration (strictly 0 < psi < 1)
 *
 *  The free energy density is:
 *
 *    F = F_\phi + F_\psi + F_surf + F_add
 *
 *  with
 *
 *    F_phi  = symmetric phi^4 free energy
 *    F_psi  = kT [\psi ln \psi + (1 - \psi) ln (1 - \psi)] 
 *    F_surf = - (1/2)\epsilon\psi (grad \phi)^2
 *             - (1/2)\beta \psi^2 (grad \phi)^2
 *    F_add  = + (1/2) W \psi \phi^2
 *
 *  The beta term allows one to get at the Frumkin isotherm and has
 *  been added here.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "surfactant.h"

/* Some values might be
 * a_       = -0.0208333;
 * b_       = +0.0208333;
 * kappa_   = +0.12;
 * 
 * kt_      = 0.00056587;
 * epsilon_ = 0.03;
 * beta_    = 0.0;
 * w_       = 0.0;
 */

struct fe_surfactant1_s {
  fe_t super;                       /* "Superclass" block */
  pe_t * pe;                        /* Parallel environment */
  cs_t * cs;                        /* Coordinate system */
  fe_surf1_param_t * param;         /* Parameters */
  field_t * phi;                    /* Single field with {phi,psi} */
  field_grad_t * dphi;              /* gradients thereof */
  fe_surf1_t * target;              /* Device copy */
};

/* Virtual function table (host) */

static fe_vt_t fe_surf1_hvt = {
  (fe_free_ft)      fe_surf1_free,     /* Virtual destructor */
  (fe_target_ft)    fe_surf1_target,   /* Return target pointer */
  (fe_fed_ft)       fe_surf1_fed,      /* Free energy density */
  (fe_mu_ft)        fe_surf1_mu,       /* Chemical potential */
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_surf1_str,      /* Total stress */
  (fe_str_ft)       fe_surf1_str,      /* Symmetric stress */
  (fe_str_ft)       NULL,              /* Antisymmetric stress (not relevant */
  (fe_hvector_ft)   NULL,              /* Not relevant */
  (fe_htensor_ft)   NULL,              /* Not relevant */
  (fe_htensor_v_ft) NULL,              /* Not reelvant */
  (fe_stress_v_ft)  fe_surf1_str_v,    /* Total stress (vectorised version) */
  (fe_stress_v_ft)  fe_surf1_str_v,    /* Symmetric part (vectorised) */
  (fe_stress_v_ft)  NULL               /* Antisymmetric part */
};


static __constant__ fe_surf1_param_t const_param;

/****************************************************************************
 *
 *  fe_surf1_create
 *
 ****************************************************************************/

int fe_surf1_create(pe_t * pe, cs_t * cs, field_t * phi,
		    field_grad_t * dphi, fe_surf1_param_t param,
		    fe_surf1_t ** fe) {
  int ndevice;
  fe_surf1_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(fe);
  assert(phi);
  assert(dphi);

  obj = (fe_surf1_t *) calloc(1, sizeof(fe_surf1_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(fe_surf1_t) failed\n");

  obj->param = (fe_surf1_param_t *) calloc(1, sizeof(fe_surf1_param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(pe, "calloc(fe_surf1_param_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->phi = phi;
  obj->dphi = dphi;
  obj->super.func = &fe_surf1_hvt;
  obj->super.id = FE_SURFACTANT1;

  /* Allocate target memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe_surf1_param_set(obj, param);
    obj->target = obj;
  }
  else {
    fe_surf1_param_t * tmp;
    tdpMalloc((void **) &obj->target, sizeof(fe_surf1_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpMemcpy(&obj->target->param, tmp, sizeof(fe_surf1_param_t *),
	      tdpMemcpyHostToDevice);
    /* Now copy. */
    assert(0); /* No implementation */
  }

  *fe = obj;

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_free
 *
 ****************************************************************************/

__host__ int fe_surf1_free(fe_surf1_t * fe) {

  int ndevice;

  assert(fe);

  tdpGetDeviceCount(&ndevice);
  if (ndevice > 0) tdpFree(fe->target);

  free(fe->param);
  free(fe);

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_info
 *
 *  Some information on parameters etc.
 *
 ****************************************************************************/

__host__ int fe_surf1_info(fe_surf1_t * fe) {

  double sigma, xi0;
  double psi_c;
  pe_t * pe = NULL;

  assert(fe);

  pe = fe->pe;

  fe_surf1_sigma(fe, &sigma);
  fe_surf1_xi0(fe, &xi0);
  fe_surf1_langmuir_isotherm(fe, &psi_c);

  pe_info(pe, "Surfactant free energy parameters:\n");
  pe_info(pe, "Bulk parameter A      = %12.5e\n", fe->param->a);
  pe_info(pe, "Bulk parameter B      = %12.5e\n", fe->param->b);
  pe_info(pe, "Surface penalty kappa = %12.5e\n", fe->param->kappa);
  pe_info(pe, "Scale energy kT       = %12.5e\n", fe->param->kt);
  pe_info(pe, "Surface adsorption e  = %12.5e\n", fe->param->epsilon);
  pe_info(pe, "Surface psi^2 beta    = %12.5e\n", fe->param->beta);
  pe_info(pe, "Enthalpic term W      = %12.5e\n", fe->param->w);

  pe_info(pe, "\n");
  pe_info(pe, "Derived quantities\n");
  pe_info(pe, "Interfacial tension   = %12.5e\n", sigma);
  pe_info(pe, "Interfacial width     = %12.5e\n", xi0);
  pe_info(pe, "Langmuir isotherm     = %12.5e\n", psi_c);

}

/****************************************************************************
 *
 *  fe_surf1_target
 *
 ****************************************************************************/

__host__ int fe_surf1_target(fe_surf1_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_param_set
 *
 ****************************************************************************/

__host__ int fe_surf1_param_set(fe_surf1_t * fe, fe_surf1_param_t vals) {

  assert(fe);

  *fe->param = vals;

  return 0;
}

/*****************************************************************************
 *
 *  fe_surf1_param
 *
 *****************************************************************************/

__host__ int fe_surf1_param(fe_surf1_t * fe, fe_surf1_param_t * values) {
  assert(fe);

  *values = *fe->param;

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_sigma
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

__host__ int fe_surf1_sigma(fe_surf1_t * fe,  double * sigma0) {

  double a, b, kappa;

  assert(fe);
  assert(sigma0);

  a = fe->param->a;
  b = fe->param->b;
  kappa = fe->param->kappa;

  *sigma0 = sqrt(-8.0*kappa*a*a*a/(9.0*b*b));

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_xi0
 *
 *  Interfacial width.
 *
 ****************************************************************************/

__host__ int fe_surf1_xi0(fe_surf1_t * fe, double * xi0) {

  assert(fe);
  assert(xi0);

  *xi0 = sqrt(-2.0*fe->param->kappa/fe->param->a);

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_langmuir_isotherm
 *
 *  The Langmuir isotherm psi_c is given by
 *  
 *  ln psi_c = (1/2) epsilon / (kT xi_0^2)
 *
 *  and can be a useful reference. The situation is more complex if
 *  beta is not zero (Frumpkin isotherm).
 *
 ****************************************************************************/ 

__host__ int fe_surf1_langmuir_isotherm(fe_surf1_t * fe, double * psi_c) {

  double xi0;

  assert(fe);

  fe_surf1_xi0(fe, &xi0);
  *psi_c = exp(0.5*fe->param->epsilon / (fe->param->kt*xi0*xi0));

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_fed
 *
 *  This is:
 *     (1/2)A \phi^2 + (1/4)B \phi^4 + (1/2) kappa (\nabla\phi)^2
 *   + kT [ \psi ln \psi + (1 - \psi) ln (1 - \psi) ]
 *   - (1/2) \epsilon\psi (\nabla\phi)^2 - (1/2) \beta \psi^2 (\nabla\phi)^2
 *   + (1/2)W\psi\phi^2
 *
 ****************************************************************************/

__host__ int fe_surf1_fed(fe_surf1_t * fe, int index, double * fed) {

  double field[2];
  double phi;
  double psi;
  double dphi[2][3];
  double dphisq;

  assert(fe);

  field_scalar_array(fe->phi, index, field);
  field_grad_pair_grad(fe->dphi, index, dphi);

  phi = field[0];
  psi = field[1];

  dphisq = dphi[0][X]*dphi[0][X] + dphi[0][Y]*dphi[0][Y]
         + dphi[0][Z]*dphi[0][Z];

  /* We have the symmetric piece followed by terms in psi */

  *fed = 0.5*fe->param->a*phi*phi + 0.25*fe->param->b*phi*phi*phi*phi
    + 0.5*fe->param->kappa*dphisq;

  assert(psi > 0.0);
  assert(psi < 1.0);

  *fed += fe->param->kt*(psi*log(psi) + (1.0 - psi)*log(1.0 - psi))
    - 0.5*fe->param->epsilon*psi*dphisq
    - 0.5*fe->param->beta*psi*psi*dphisq
    + 0.5*fe->param->w*psi*phi*phi;

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_mu
 * 
 *  Two chemical potentials are present:
 *
 *  \mu_\phi = A\phi + B\phi^3 - kappa \nabla^2 \phi
 *           + W\phi \psi
 *           + \epsilon (\psi \nabla^2\phi + \nabla\phi . \nabla\psi)
 *           + \beta (\psi^2 \nabla^2\phi + 2\psi \nabla\phi . \nabla\psi) 
 * 
 *  \mu_\psi = kT (ln \psi - ln (1 - \psi) + (1/2) W \phi^2
 *           - (1/2) \epsilon (\nabla \phi)^2
 *           - \beta \psi (\nabla \phi)^2
 *
 ****************************************************************************/

__host__ int fe_surf1_mu(fe_surf1_t * fe, int index, double * mu) {

  double phi;
  double psi;
  double field[2];
  double grad[2][3];
  double delsq[2];

  assert(fe);
  assert(mu); assert(mu + 1);

  field_scalar_array(fe->phi, index, field);
  field_grad_pair_grad(fe->dphi, index, grad);
  field_grad_pair_delsq(fe->dphi, index, delsq);

  phi = field[0];
  psi = field[1];

  /* mu_phi */

  mu[0] = fe->param->a*phi + fe->param->b*phi*phi*phi
    - fe->param->kappa*delsq[0]
    + fe->param->w*phi*psi
    + fe->param->epsilon*(psi*delsq[0] + dot_product(grad[0], grad[1]))
    + fe->param->beta*psi*(psi*delsq[0] + 2.0*dot_product(grad[0], grad[1]));

  /* mu_psi */

  assert(psi > 0.0);
  assert(psi < 1.0);

  mu[1] = fe->param->kt*(log(psi) - log(1.0 - psi))
    + 0.5*fe->param->w*phi*phi
    - 0.5*fe->param->epsilon*dot_product(grad[0], grad[0])
    - fe->param->beta*psi*dot_product(grad[0], grad[0]);

  return 0;
}

/****************************************************************************
 *
 *  fe_surf1_str
 *
 *  Thermodynamic stress S_ab = p0 delta_ab + P_ab
 *
 *  p0 = (1/2) A \phi^2 + (3/4) B \phi^4 - (1/2) \kappa \nabla^2 \phi
 *     - (1/2) kappa (\nabla phi)^2
 *     - kT ln(1 - \psi)
 *     + W \psi \phi^2
 *     + \epsilon \phi \nabla_a \phi \nabla_a \psi
 *     + \epsilon \phi \psi \nabla^2 \phi
 *     + 2 \beta \phi \psi \nabla_a\phi \nabla_a\psi
 *     + \beta\phi\psi^2 \nabla^2 \phi
 *     - (1/2) \beta\psi^2 (\nabla\phi)^2  
 *
 *  P_ab = (\kappa - \epsilon\psi - \beta\psi^2) \nabla_a \phi \nabla_b \phi
 *
 ****************************************************************************/

__host__ int fe_surf1_str(fe_surf1_t * fe, int index, double s[3][3]) {

  int ia, ib;
  double field[2];
  double phi;
  double psi;
  double delsq[2];
  double grad[2][3];
  double p0;
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  field_scalar_array(fe->phi, index, field);
  field_grad_pair_grad(fe->dphi, index, grad);
  field_grad_pair_delsq(fe->dphi, index, delsq);

  phi = field[0];
  psi = field[1];

  assert(psi < 1.0);

  p0 = 0.5*fe->param->a*phi*phi + 0.75*fe->param->b*phi*phi*phi*phi
    - fe->param->kappa*(phi*delsq[0] - 0.5*dot_product(grad[0], grad[0]))
    - fe->param->kt*log(1.0 - psi)
    + fe->param->w*psi*phi*phi
    + fe->param->epsilon*phi*(dot_product(grad[0], grad[1]) + psi*delsq[0])
    + fe->param->beta*psi*(2.0*phi*dot_product(grad[0], grad[1])
			   + phi*psi*delsq[0]
			   - 0.5*psi*dot_product(grad[0], grad[0]));

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d[ia][ib]	+
	(fe->param->kappa - fe->param->epsilon*psi - fe->param->beta*psi*psi)*
	grad[0][ia]*grad[0][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_surf1_str_v
 *
 *  Stress (vectorised version) Currently a patch-up.
 *
 *****************************************************************************/

int fe_surf1_str_v(fe_surf1_t * fe, int index, double s[3][3][NSIMDVL]) {

  int ia, ib;
  int iv;
  double s1[3][3];

  assert(fe);

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_surf1_str(fe, index + iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return 0;
}
