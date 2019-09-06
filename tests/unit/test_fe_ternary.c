/*****************************************************************************
 *
 *  test_fe_tenery.c
 *
 *  Unit tests for ternary free energy.
 *
 *  Edinburgh Soft Matter and Statistical Phsyics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Shan Chen (shan.chen@epfl.ch)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "fe_ternary.h"
#include "tests.h"

__host__ int test_fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_ternary_fed(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_ternary_mu(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_ternary_str(pe_t * pe, cs_t * cs, field_t * phi);

__host__ int shan_ternary_fed(fe_ternary_t * fe, int index, double * fed);
__host__ int shan_ternary_mu(fe_ternary_t * fe, int index, double * mu);
__host__ int shan_ternary_str(fe_ternary_t * fe, int index, double s[3][3]);

/*****************************************************************************
 *
 *  test_fe_tenrary_suite
 *
 *****************************************************************************/

__host__ int test_fe_ternary_suite(void) {

  const int nf2 = 2;
  const int nhalo = 2;

  int ndevice;
  pe_t * pe = NULL;
  cs_t * cs = NULL;
  field_t * phi = NULL;

  tdpGetDeviceCount(&ndevice);

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  if (ndevice) {
    pe_info(pe, "SKIP     ./unit/test_fe_ternary\n");
  }
  else {

    cs_create(pe, &cs);
    cs_init(cs);

    field_create(pe, cs, nf2, "ternary", &phi);
    field_init(phi, nhalo, NULL);

    test_fe_ternary_create(pe, cs, phi);
    test_fe_ternary_fed(pe, cs, phi);
    test_fe_ternary_mu(pe, cs, phi);
    test_fe_ternary_str(pe, cs, phi);

    field_free(phi);
    cs_free(cs);
  }

  pe_info(pe, "PASS     ./unit/test_fe_ternary\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_create
 *
 *****************************************************************************/

__host__ int test_fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  fe_ternary_param_t pref = {0.1, 0.2, 0.3, 0.4};
  fe_ternary_param_t p    = {0};
  field_grad_t * dphi = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  assert(fe);

  fe_ternary_param(fe, &p);
  test_assert((pref.alpha  - p.alpha)  < DBL_EPSILON);
  test_assert((pref.kappa1 - p.kappa1) < DBL_EPSILON);
  test_assert((pref.kappa2 - p.kappa2) < DBL_EPSILON);
  test_assert((pref.kappa3 - p.kappa3) < DBL_EPSILON);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_fed
 *
 *****************************************************************************/

__host__ int test_fe_ternary_fed(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  field_grad_t * dphi = NULL;
  fe_ternary_param_t pref = {0.5, 0.6, 0.7, 0.8};

  int index = 1;
  double phi0[2] = {-0.3, 0.7};
  double grad[2][3] = {{0.1, -0.2, 0.3}, {-0.4, 0.5, -0.7}};
  double fed;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients */

  field_scalar_array_set(phi, index, phi0);
  fe_ternary_fed(fe, index, &fed);
  test_assert(fabs(fed - 3.3075000e-02) < DBL_EPSILON);

  /* With gradients */

  field_grad_pair_grad_set(dphi, index, grad);
  fe_ternary_fed(fe, index, &fed);
  test_assert(fabs(fed - 1.6313750e-01) < DBL_EPSILON);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_mu
 *
 *****************************************************************************/

__host__ int test_fe_ternary_mu(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  field_grad_t * dphi = NULL;
  fe_ternary_param_t pref = {0.5, 0.6, 0.7, 0.8};

  int index = 1;
  double phi0[2] = {-0.3, 0.7};
  double d2phi[2] = {0.1, 0.4};
  double mu[3];

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients */

  field_scalar_array_set(phi, index, phi0);
  fe_ternary_mu(fe, index, mu);
  test_assert(fabs(mu[0] - -2.9400000e-02) < DBL_EPSILON);
  test_assert(fabs(mu[1] - -9.6600000e-02) < DBL_EPSILON);
  test_assert(fabs(mu[2] - -2.9400000e-02) < DBL_EPSILON);

  /* With delsq */

  field_grad_pair_delsq_set(dphi, index, d2phi);
  fe_ternary_mu(fe, index, mu);
  test_assert(fabs(mu[0] - -4.0025000e-02) < DBL_EPSILON);
  test_assert(fabs(mu[1] - -2.0972500e-01) < DBL_EPSILON);
  test_assert(fabs(mu[2] - -5.0250000e-03) < DBL_EPSILON);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_str
 *
 *****************************************************************************/

__host__ int test_fe_ternary_str(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  field_grad_t * dphi = NULL;
  fe_ternary_param_t pref = {0.5, 0.6, 0.7, 0.8};

  int index = 1;
  double phi0[2] = {-0.3, 0.7};
  double d2phi[2] = {0.1, 0.4};
  double grad[2][3] = {{0.1, -0.2, 0.3}, {-0.4, 0.5, -0.7}};
  double s[3][3];

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients */

  field_scalar_array_set(phi, index, phi0);
  fe_ternary_str(fe, index, s);

  test_assert(fabs(s[0][0] - 5.2552500e-01) < DBL_EPSILON);
  test_assert(fabs(s[0][1] - 0.0000000e+00) < DBL_EPSILON);
  test_assert(fabs(s[0][2] - 0.0000000e+00) < DBL_EPSILON);
  test_assert(fabs(s[1][0] - 0.0000000e+00) < DBL_EPSILON);
  test_assert(fabs(s[1][1] - 5.2552500e-01) < DBL_EPSILON);
  test_assert(fabs(s[1][2] - 0.0000000e+00) < DBL_EPSILON);
  test_assert(fabs(s[2][0] - 0.0000000e+00) < DBL_EPSILON);
  test_assert(fabs(s[2][1] - 0.0000000e+00) < DBL_EPSILON);
  test_assert(fabs(s[2][2] - 5.2552500e-01) < DBL_EPSILON);

  /* With grad */

  field_grad_pair_grad_set(dphi, index, grad);
  fe_ternary_str(fe, index, s);

  test_assert(fabs(s[0][0] -  4.4077500e-01) < DBL_EPSILON);
  test_assert(fabs(s[0][1] - -5.7062500e-02) < DBL_EPSILON);
  test_assert(fabs(s[0][2] -  8.0000000e-02) < DBL_EPSILON);
  test_assert(fabs(s[1][0] - -5.7062500e-02) < DBL_EPSILON);
  test_assert(fabs(s[1][1] -  4.6777500e-01) < DBL_EPSILON);
  test_assert(fabs(s[1][2] - -1.0150000e-01) < DBL_EPSILON);
  test_assert(fabs(s[2][0] -  8.0000000e-02) < DBL_EPSILON);
  test_assert(fabs(s[2][1] - -1.0150000e-01) < DBL_EPSILON);
  test_assert(fabs(s[2][2] -  5.3796250e-01) < DBL_EPSILON);

  /* With delsq */

  field_grad_pair_delsq_set(dphi, index, d2phi);
  fe_ternary_str(fe, index, s);

  test_assert(fabs(s[0][0] -  3.9790000e-01) < DBL_EPSILON);
  test_assert(fabs(s[0][1] - -5.7062500e-02) < DBL_EPSILON);
  test_assert(fabs(s[0][2] -  8.0000000e-02) < DBL_EPSILON);
  test_assert(fabs(s[1][0] - -5.7062500e-02) < DBL_EPSILON);
  test_assert(fabs(s[1][1] -  4.2490000e-01) < DBL_EPSILON);
  test_assert(fabs(s[1][2] - -1.0150000e-01) < DBL_EPSILON);
  test_assert(fabs(s[2][0] -  8.0000000e-02) < DBL_EPSILON);
  test_assert(fabs(s[2][1] - -1.0150000e-01) < DBL_EPSILON);
  test_assert(fabs(s[2][2] -  4.9508750e-01) < DBL_EPSILON);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}


/****************************************************************************
 *
 ****************************************************************************/

__host__ int shan_ternary_fed(fe_ternary_t * fe, int index, double * fed) {
    
    int ia,ic;
    double a[3], b[3];
    double field[2];
    double phi;
    double psi;
    double rho;
    double grad[2][3];
    double grho[3];  /* the gradient of rho*/
    
    assert(fe);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_grad(fe->dphi, index, grad);
    
    phi = field[0];
    psi = field[1];
    assert(abs(phi)+abs(psi) == 1);
    rho = 1.0;
    
    for (ic=0;ic<3; ic++){
        grho[ic] = 0.0;
        a[ic] = 0.0;
        b[ic] = 0.0;
    }
    
    for (ia = 0; ia < 3; ia++){
        a[ia] = grho[ia] + grad[0][ia] - grad[1][ia];
        b[ia] = grho[ia] - grad[0][ia] - grad[1][ia];
    }
    
    *fed = 0.03125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi)*(2 + psi - rho - phi)*(2 + psi - rho - phi)
    + 0.125*fe->param->kappa1*fe->param->alpha*fe->param->alpha*dot_product(a,a)
    + 0.03125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi)*(2 + psi - rho + phi)*(2 + psi - rho + phi)
    + 0.125*fe->param->kappa2*fe->param->alpha*fe->param->alpha*dot_product(b,b)
    + 0.5*fe->param->kappa3*psi*psi*(1 - psi)*(1 - psi)
    + 0.5*fe->param->kappa3*fe->param->alpha*fe->param->alpha*dot_product(grad[1], grad[1]);
    
    return 0;
}

/****************************************************************************
 *
 ****************************************************************************/

__host__ int shan_ternary_mu(fe_ternary_t * fe, int index, double * mu) {
    
    double phi;
    double psi;
    double rho;
    double field[2];
    double delsq[2];
    double delsq_rho; /* laplace operator*/
    
    assert(fe);
    assert(mu); assert(mu + 1);//assert(mu + 2);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_delsq(fe->dphi, index, delsq);
    
    phi = field[0];
    psi = field[1];
    assert(abs(phi)+abs(psi) == 1);
    rho = 1.0;
    delsq_rho = 0.0;
    
    //mu_phi
    mu[0] = 0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi - 2)*(rho - phi - psi - 1)
    + 0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa2 - fe->param->kappa1)*(delsq_rho - delsq[1])
    - (fe->param->kappa1 + fe->param->kappa2)*delsq[0]);
    
    //mu_psi
    mu[1] = -0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi - 2)*(rho - phi - psi - 1)
    + fe->param->kappa3*psi*(psi - 1)*(2*psi - 1)
    +0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa1 + fe->param->kappa2)*delsq_rho
    -(fe->param->kappa2 - fe->param->kappa1)*delsq[0]
    - (fe->param->kappa2 + fe->param->kappa1 + 4*fe->param->kappa3)*delsq[1]);
    //mu_rho
    mu[2] = 0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi - 2)*(rho - phi - psi -1)
    + 0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa1 + fe->param->kappa2)*(delsq[1] - delsq[0])
    + (fe->param->kappa2 - fe->param->kappa1)*delsq_rho);
    
    return 0;
}

/****************************************************************************
 *
 ****************************************************************************/

__host__ int shan_ternary_str(fe_ternary_t * fe, int index, double s[3][3]) {
    
    int ia, ib,ic;
    double field[2];
    double phi;
    double psi;
    double rho;
    double delsq[2];
    double grad[2][3];
    double p0;
    double grho[3];
    double delsq_rho;
    KRONECKER_DELTA_CHAR(d);
    
    assert(fe);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_grad(fe->dphi, index, grad);
    field_grad_pair_delsq(fe->dphi, index, delsq);
    rho = 1.0;
    phi = field[0];
    psi = field[1];
    assert(abs(phi)+abs(psi) == 1);
    
    for (ic=0;ic<3; ic++){
        grho[ic] = 0.0;
    }
    
    delsq_rho = 0.0;
    
   
    
    p0= (fe->param->kappa1 + fe->param->kappa2)*(0.09375*rho*rho*rho*rho + 0.09375*phi*phi*phi*phi + 0.5625*rho*rho*phi*phi
    + 0.5625*rho*rho*psi*psi + 0.5625*phi*phi*psi*psi - 0.375*rho*rho*rho*psi - 0.375*rho*psi*psi*psi + 0.75*rho*rho*psi
    - 0.75*rho*phi*phi - 0.75*rho*psi*psi + 0.75*phi*phi*psi - 0.25*rho*rho*rho + 0.125*rho*rho + 0.125*phi*phi
    - 0.25*rho*psi- 1.125*rho*phi*phi*psi)
    + (fe->param->kappa1 - fe->param->kappa2)*(0.375*rho*rho*rho*phi + 0.375*rho*phi*phi*phi - 0.375*phi*phi*phi*psi
    - 0.375*phi*psi*psi*psi -0.25*phi*phi*phi -0.75*rho*rho*phi - 0.75*phi*psi*psi + 0.25*rho*phi - 0.25*phi*psi
    + 1.125*rho*phi*psi*psi - 1.125*rho*rho*phi*psi + 1.5*rho*phi*psi)
    + 0.25*(fe->param->kappa1 + fe->param->kappa2 - 8*fe->param->kappa3)*psi*psi*psi
    +(fe->param->kappa1 + fe->param->kappa2 + 16*fe->param->kappa3)*(0.09375*psi*psi*psi*psi + 0.125*psi*psi)
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(-0.5*(dot_product(grho,grho)) - rho*delsq_rho)
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(-0.5*(dot_product(grad[0],grad[0])) - phi*delsq[0])
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2 + 4*fe->param->kappa3)*(-0.5*(dot_product(grad[1],grad[1])) - psi*delsq[1])
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(-dot_product(grho,grad[0])-rho*delsq[0]-phi*delsq_rho)
    - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(-dot_product(grho,grad[1])-rho*delsq[1]-psi*delsq_rho)
    - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(-dot_product(grad[0],grad[1])-phi*delsq[1]-psi*delsq[0]);
    
    
    for (ia = 0; ia < 3; ia++) {
        for (ib = 0; ib < 3; ib++) {
            s[ia][ib] = p0*d[ia][ib] + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grho[ia]*grho[ib])
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grad[0][ia]*grad[0][ib])
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2 + 4*fe->param->kappa3)*(grad[1][ia]*grad[1][ib])
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(grho[ia]*grad[0][ib] + grad[0][ia]*grho[ib])
            - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grho[ia]*grad[1][ib] + grad[1][ia]*grho[ib])
            - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(grad[0][ia]*grad[1][ib] + grad[1][ia]*grad[0][ib]);
        }
    }
        
    return 0;
}

